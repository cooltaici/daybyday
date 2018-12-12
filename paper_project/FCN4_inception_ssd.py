from keras.applications.vgg16 import VGG16
from keras.models import *
from keras.layers import *
from keras_layer_L2Normalization import *
from fcn_inception import *

def fcn4_inception3_ssd(weightpath=r"",
                        isnorm = True,
                        image_size = (224,224),
                        n_classes = 2,
                        min_scale=None,
                        max_scale=None,
                        scale = [ 0.1, 0.23, 0.46, 0.61],
                        aspect_ratios_global=None,
                        aspect_ratios_per_layer=[[0.5, 1.0, 2.0],
                                                [1.0/3.0, 0.5, 1.0, 2.0, 3.0],
                                                [0.5, 1.0, 2.0]],
                        two_boxes_for_ar1=True,
                        limit_boxes=False,
                        variances=[0.1, 0.1, 0.2, 0.2],
                        coords='centroids',
                        normalize_coords=False):
    if scale:
        scales = np.array(scale)
    else:
        raise ValueError("scale 不能为空")
    if aspect_ratios_per_layer:
        n_boxes = []
        for aspect_ratios in aspect_ratios_per_layer:
            if (1 in aspect_ratios) & two_boxes_for_ar1:
                n_boxes.append(len(aspect_ratios) + 1) # +1 for the second box for aspect ratio 1
            else:
                n_boxes.append(len(aspect_ratios))
        n_boxes_conv3_3 = n_boxes[0] # 4 boxes per cell for the original implementation
        n_boxes_conv4_3 = n_boxes[1]
        n_boxes_conv5_3 = n_boxes[2]
    #加载分割网络，初始化权值，并固定
    input_shape = (image_size[0],image_size[1],3)
    model_Seg = fcn4_inception3(True)
    if os.path.exists(weightpath):
        model_Seg.load_weights(filepath = weightpath)
    else:
        print("分割网络权重不存在！")
    img_input = model_Seg.input
    for layers in model_Seg.layers:
        layers.trainable = False
    #SSD：使用低层特征
    # Conv3_3 = model_Seg.get_layer('block3_conv3').output
    # Conv4_3 = model_Seg.get_layer('block4_conv3').output
    # Conv5_3 = model_Seg.get_layer('block5_conv3').output
    #Conv3_3_norm = L2Normalization(gamma_init=20, name='conv4_3_norm')(Conv3_3)
    #SSD反卷积：使用高层特征
    Conv3_3 = model_Seg.get_layer('UnConv28').output
    Conv4_3 = model_Seg.get_layer('UnConv14').output
    Conv5_3 = model_Seg.get_layer('UnConv7').output
    Conv3_3 = Conv2D(128 , (3 ,3) , activation='relu' , padding='same')(Conv3_3)
    Conv4_3 = Conv2D(128 , (3 ,3) , activation='relu' , padding='same')(Conv4_3)
    Conv5_3 = Conv2D(128 , (3 ,3) , activation='relu' , padding='same')(Conv5_3)
    Conv3_3_norm = L2Normalization(gamma_init=20, name='conv4_3_norm')(Conv3_3)

    #output shape is (batch, nbox_total, 2class)
    Conv3_3_norm_mbox_conf = Conv2D(n_boxes_conv3_3 * n_classes, (3, 3), padding='same', name='Conv3_3_norm_mbox_conf')(Conv3_3_norm)
    Conv4_3_mbox_conf = Conv2D(n_boxes_conv4_3 * n_classes, (3, 3), padding='same', name='Conv4_3_mbox_conf')(Conv4_3)
    Conv5_3_mbox_conf = Conv2D(n_boxes_conv5_3 * n_classes, (3, 3), padding='same', name='Conv5_3_mbox_conf')(Conv5_3)

    Conv3_3_norm_mbox_conf_reshape = Reshape((-1, n_classes), name='Conv3_3_norm_mbox_conf_reshape')(Conv3_3_norm_mbox_conf)
    Conv4_3_mbox_conf_reshape = Reshape((-1, n_classes), name='Conv4_3_mbox_conf_reshape')(Conv4_3_mbox_conf)
    Conv5_3_mbox_conf_reshape = Reshape((-1, n_classes), name='Conv5_3_mbox_conf_reshape')(Conv5_3_mbox_conf)

    mbox_conf = Concatenate(axis=1, name='mbox_conf')([Conv3_3_norm_mbox_conf_reshape,
                                                       Conv4_3_mbox_conf_reshape,
                                                       Conv5_3_mbox_conf_reshape])
    mbox_conf_softmax = Activation('softmax', name='mbox_conf_softmax')(mbox_conf)
    model= Model(inputs=img_input,outputs=mbox_conf_softmax)

    predictor_sizes = np.array([Conv3_3_norm_mbox_conf._keras_shape[1:3],
                                 Conv4_3_mbox_conf._keras_shape[1:3],
                                 Conv5_3_mbox_conf._keras_shape[1:3]])

    return model,predictor_sizes

#测试输出没问题
if __name__ == '__main__':
    model = fcn4_inception3_ssd(False)
    model.summary()
    print('over!')