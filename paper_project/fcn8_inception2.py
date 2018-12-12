from keras.applications.vgg16 import VGG16
from keras.models import *
from keras.layers import *
from keras_layer_L2Normalization import *
from fcn_inception import *

def conv2d_bn(x,
              filters,
              num_row,
              num_col,
              isnorm = True,
              padding='same',
              strides=(1, 1),
              name=None):
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None
    if K.image_data_format() == 'channels_first':
        bn_axis = 1
    else:
        bn_axis = 3
    if isnorm:
        x = Conv2D(
                    filters, (num_row, num_col),
                    strides=strides,
                    kernel_initializer='he_normal',
                    padding=padding,
                    name=conv_name)(x)
        x = BatchNormalization(axis=bn_axis, scale=False, name=bn_name)(x)
        x = Activation('relu', name=name)(x)
        return x
    else:
        x = Conv2D(filters, (num_row, num_col), strides=strides, kernel_initializer='he_normal', padding='same',name = conv_name)(x)
        return x
#这里我们使用了inception,resnet等网络的思想，如果isnorm为真的话，我们将在一些卷积层用上batchnorm
def fcn8_inception2(isnorm = True):
    nclass = 128
    input_shape = (224,224,3)
    base_model = VGG16(include_top=None, pooling=None, weights='imagenet', input_shape = input_shape)
    img_input = base_model.input
    #f1 = base_model.get_layer('block1_pool').output    #112*112*64
    #f2 = base_model.get_layer('block2_pool').output    #56*56*128 从这里开始可以增加分辨率
    f3 = base_model.get_layer('block3_pool').output    #28*28*256
    f4 = base_model.get_layer('block4_pool').output    #14*14*512
    f5 = base_model.get_layer('block5_pool').output    #7*7*512

    o = f5
    #o = Conv2D( 256 , ( 7 , 7 ) , activation='relu' , padding='same')(o)  # 原来的值4096
    o = conv2d_bn(o, 512, 7, 7, isnorm)
    o = Dropout(0.5)(o)
    #o = Conv2D( 256 , ( 1 , 1 ) , activation='relu' , padding='same')(o)  # 原来的值4096
    o = conv2d_bn(o, 512, 1, 1, isnorm)
    o = Dropout(0.5)(o)

    o = conv2d_bn(o, nclass, 1, 1,name="UnConv7")
    #o = Conv2D( nclass ,  ( 1 , 1 ) ,kernel_initializer='he_normal', padding='same')(o)
    o = Conv2DTranspose( nclass , kernel_size=(4,4) ,  strides=(2,2) , padding='same', use_bias=False)(o)

    ##################################################
    o4 = f4
    #f4branch1x1 = Conv2D(nclass, (1, 1), kernel_initializer='he_normal', padding='same')(o4)
    f4branch1x1 = conv2d_bn(o4, nclass, 1, 1)

    f4branch3x3db = conv2d_bn(o4, nclass, 5, 5)
    f4branch3x3db = conv2d_bn(f4branch3x3db, nclass, 1, 1)

    o = Add()([o,f4branch1x1,f4branch3x3db],name="UnConv14")

    o = Conv2DTranspose( nclass , kernel_size=(4,4) ,  strides=(2,2) , padding='same', use_bias=False)(o)
    ####################################################
    o3 = f3
    #f3branch1x1 = Conv2D(nclass, (1, 1), kernel_initializer='he_normal', padding='same')(o3)
    f3branch1x1 = conv2d_bn(o3, nclass, 1, 1)
    # 要不要卷积分解？卷积分解可以减小训练参数，加快计算时间
    f3branch3x3db = conv2d_bn(o3, nclass, 5, 5)
    f3branch3x3db = conv2d_bn(f3branch3x3db, nclass, 1, 1)

    o = Add()([o, f3branch1x1, f3branch3x3db],name="UnConv28")
    o = Conv2DTranspose(1, kernel_size=(16, 16), strides=(8, 8), padding='same', name="Seg_output", use_bias=False,
                        activation='sigmoid')(o)
    ####################################################
    model = Model(img_input , o )
    midname = 'block1_conv2'
    for layers in model.layers:
        layers.trainable = False
        if layers.name == midname:
            break
    return model

#fcn8_inception2的基础上增加SSD
def fcn8_inception2_ssd(weightpath=r"",
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
    nclass = 128
    input_shape = (224,224,3)
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
    model_Seg = fcn8_inception2(True)
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
    print('over!')