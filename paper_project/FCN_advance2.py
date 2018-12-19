# encoding:utf-8
from keras.applications.vgg16 import VGG16
from keras.models import *
from keras.layers import *
from keras_layer_L2Normalization import *
from fcn_inception import *
import os

def FCN8(nClasses,iflevel = False):
    input_shape = (224,224,3)
    base_model = VGG16(include_top=None, pooling=None, weights='imagenet', input_shape = input_shape)
    img_input = base_model.input
    # f1 = base_model.get_layer('block1_pool').output    #112*112
    # f2 = base_model.get_layer('block2_pool').output    #56*56
    f3 = base_model.get_layer('block3_pool').output
    f4 = base_model.get_layer('block4_pool').output    #14*14
    f5 = base_model.get_layer('block5_pool').output    #7*7

    o = f5
    o = Conv2D( 1024 , ( 7 , 7 ) , activation='relu' , padding='same')(o)
    o = Dropout(0.5)(o)
    o = Conv2D( 1024 , ( 1 , 1 ) , activation='relu' , padding='same')(o)
    o = Dropout(0.5)(o)

    o = Conv2D( nClasses ,  ( 1 , 1 ) ,kernel_initializer='he_normal', padding='same',name="UnConv7")(o)     #1层
    o = Conv2DTranspose( nClasses , kernel_size=(4,4) ,  strides=(2,2) , padding='same', use_bias=False)(o)

    o2 = f4
    o2 = Conv2D(nClasses , (1 , 1 ) ,kernel_initializer='he_normal' , padding='same')(o2)
    o = Add()([ o , o2 ])                                                                       #2层

    o = Conv2DTranspose(nClasses , kernel_size=(4,4) ,  strides=(2,2), padding='same' , use_bias=False )(o)
    o3 = f3
    o3 = Conv2D(nClasses ,  (1 , 1 ) ,kernel_initializer='he_normal', padding='same')(o3)
    o  = Add()([ o , o3 ])                                                                      #3层

    if iflevel:
        o = Conv2DTranspose(nClasses, kernel_size=(16, 16), name='out_image', strides=(8, 8), padding='same',
                            use_bias=False,activation='relu')(o)
    else:
        o = Conv2DTranspose(1, kernel_size=(16, 16), name='out_image', strides=(8, 8), padding='same',
                            use_bias=False,activation='sigmoid')(o)

    model = Model(img_input , o )
    midname = 'block2_conv1'  #block3_conv1
    for layers in model.layers:
        layers.trainable = False
        if layers.name == midname:
            break
    return model

#FCN8基础上在加SSD
def FCN8_ssd(weightpath=r"",
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
    nClasses = 128
    iflevel = False
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
        n_boxes_conv6_3 = n_boxes[3]
        n_boxes_conv7_3 = n_boxes[4]
    #加载分割网络，初始化权值，并固定
    input_shape = (image_size[0],image_size[1],3)
    model_Seg = FCN8(128)
    if os.path.exists(weightpath):
        model_Seg.load_weights(filepath = weightpath)
    else:
        raise  ValueError(weightpath+"分割网络权重不存在！")
    img_input = model_Seg.input
    for layers in model_Seg.layers:
        layers.trainable = False
    f2 = model_Seg.get_layer('block4_conv3').output  #28*28

    # conv5_1 = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_1')(f2)
    # conv5_2 = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_2')(conv5_1)
    # conv5_3 = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_3')(conv5_2)
    # pool5 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='pool5')(conv5_3)
    #
    # fc6 = Conv2D(1024, (3, 3), dilation_rate=(6, 6), activation='relu', padding='same', name='fc6')(pool5)
    #
    # fc7 = Conv2D(1024, (1, 1), activation='relu', padding='same', name='fc7')(fc6)
    #
    # conv6_1 = Conv2D(256, (1, 1), activation='relu', padding='same', name='conv6_1')(fc7)
    # conv6_2 = Conv2D(512, (3, 3), strides=(2, 2), activation='relu', padding='same', name='conv6_2')(conv6_1)
    #
    # conv7_1 = Conv2D(128, (1, 1), activation='relu', padding='same', name='conv7_1')(conv6_2)
    # conv7_2 = Conv2D(256, (3, 3), strides=(2, 2), activation='relu', padding='same', name='conv7_2')(conv7_1)
    #
    # conv8_1 = Conv2D(128, (1, 1), activation='relu', padding='same', name='conv8_1')(conv7_2)
    # conv8_2 = Conv2D(256, (3, 3), strides=(1, 1), activation='relu', padding='same', name='conv8_2')(conv8_1)
    #
    # conv9_1 = Conv2D(128, (1, 1), activation='relu', padding='same', name='conv9_1')(conv8_2)
    # conv9_2 = Conv2D(256, (3, 3), strides=(1, 1), activation='relu', padding='same', name='conv9_2')(conv9_1)

    conv5_1 = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_1')(f2)
    conv5_2 = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_2')(conv5_1)
    conv5_3 = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_3')(conv5_2)
    pool5 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='pool5')(conv5_3)

    fc6 = Conv2D(512, (3, 3), dilation_rate=(6, 6), activation='relu', padding='same', name='fc6')(pool5)

    fc7 = Conv2D(512, (1, 1), activation='relu', padding='same', name='fc7')(fc6)

    conv6_1 = Conv2D(128, (1, 1), activation='relu', padding='same', name='conv6_1')(fc7)
    conv6_2 = Conv2D(256, (3, 3), strides=(2, 2), activation='relu', padding='same', name='conv6_2')(conv6_1)

    conv7_1 = Conv2D(64, (1, 1), activation='relu', padding='same', name='conv7_1')(conv6_2)
    conv7_2 = Conv2D(128, (3, 3), strides=(2, 2), activation='relu', padding='same', name='conv7_2')(conv7_1)

    conv8_1 = Conv2D(64, (1, 1), activation='relu', padding='same', name='conv8_1')(conv7_2)
    conv8_2 = Conv2D(128, (3, 3), strides=(1, 1), activation='relu', padding='same', name='conv8_2')(conv8_1)

    conv9_1 = Conv2D(64, (1, 1), activation='relu', padding='same', name='conv9_1')(conv8_2)
    conv9_2 = Conv2D(128, (3, 3), strides=(1, 1), activation='relu', padding='same', name='conv9_2')(conv9_1)


    Conv3_3_norm = conv5_3
    Conv4_3 = conv6_2
    Conv5_3 = conv7_2
    Conv6_3 = conv8_2
    Conv7_3 = conv9_2

    #output shape is (batch, nbox_total, 2class)
    Conv3_3_norm_mbox_conf = Conv2D(n_boxes_conv3_3 * n_classes, (3, 3), padding='same', name='Conv3_3_norm_mbox_conf')(Conv3_3_norm)
    Conv4_3_mbox_conf = Conv2D(n_boxes_conv4_3 * n_classes, (3, 3), padding='same', name='Conv4_3_mbox_conf')(Conv4_3)
    Conv5_3_mbox_conf = Conv2D(n_boxes_conv5_3 * n_classes, (3, 3), padding='same', name='Conv5_3_mbox_conf')(Conv5_3)
    Conv6_3_mbox_conf = Conv2D(n_boxes_conv6_3 * n_classes, (3, 3), padding='same', name='Conv6_3_mbox_conf')(Conv6_3)
    Conv7_3_mbox_conf = Conv2D(n_boxes_conv7_3 * n_classes, (3, 3), padding='same', name='Conv7_3_mbox_conf')(Conv7_3)

    Conv3_3_norm_mbox_conf_reshape = Reshape((-1, n_classes), name='Conv3_3_norm_mbox_conf_reshape')(Conv3_3_norm_mbox_conf)
    Conv4_3_mbox_conf_reshape = Reshape((-1, n_classes), name='Conv4_3_mbox_conf_reshape')(Conv4_3_mbox_conf)
    Conv5_3_mbox_conf_reshape = Reshape((-1, n_classes), name='Conv5_3_mbox_conf_reshape')(Conv5_3_mbox_conf)
    Conv6_3_mbox_conf_reshape = Reshape((-1, n_classes), name='Conv6_3_mbox_conf_reshape')(Conv6_3_mbox_conf)
    Conv7_3_mbox_conf_reshape = Reshape((-1, n_classes), name='Conv7_3_mbox_conf_reshape')(Conv7_3_mbox_conf)

    mbox_conf = Concatenate(axis=1, name='mbox_conf')([Conv3_3_norm_mbox_conf_reshape,
                                                       Conv4_3_mbox_conf_reshape,
                                                       Conv5_3_mbox_conf_reshape,
                                                       Conv6_3_mbox_conf_reshape,
                                                       Conv7_3_mbox_conf_reshape])

    mbox_conf_softmax = Activation('softmax', name='mbox_conf_softmax')(mbox_conf)
    model= Model(inputs=img_input,outputs=mbox_conf_softmax)

    predictor_sizes = np.array([Conv3_3_norm_mbox_conf._keras_shape[1:3],
                                 Conv4_3_mbox_conf._keras_shape[1:3],
                                 Conv5_3_mbox_conf._keras_shape[1:3],
                                Conv6_3_mbox_conf._keras_shape[1:3],
                                Conv7_3_mbox_conf._keras_shape[1:3]])

    return model,predictor_sizes

def FCN8_ssd_combine(Segpath=r"",
                     SSDpath = r"",
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
    nClasses = 128
    iflevel = False
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
        n_boxes_conv6_3 = n_boxes[3]
        n_boxes_conv7_3 = n_boxes[4]
    #加载分割网络
    model_Seg = FCN8(128)
    if os.path.exists(Segpath):
        model_Seg.load_weights(Segpath)
    else:
        raise ValueError(Segpath + " is not exist!")
    #加载SSD
    img_input = model_Seg.input
    for layers in model_Seg.layers:
        layers.trainable = False
    f2 = model_Seg.get_layer('block4_conv3').output  #28*28
    # f3 = model_Seg.get_layer('add_2').output    #28*28
    # f4 = model_Seg.get_layer('add_1').output    #14*14
    # f6 = model_Seg.get_layer('dropout_2').output    #7*7

    #f2 = Conv2D(128, (3, 3), kernel_initializer='he_normal', activation="relu", padding='same')(f2)  # 28*28
    # f3 = Conv2D(128, (3, 3), kernel_initializer='he_normal', activation="relu",padding='same')(f3) #28*28
    # f4 = Conv2D(128, (3, 3), kernel_initializer='he_normal', activation="relu", padding='same')(f4) #14*14
    # f6 = Conv2D(128, (3, 3), kernel_initializer='he_normal', activation="relu", padding='same')(f6) #7*7

    conv5_1 = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_1')(f2)
    conv5_2 = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_2')(conv5_1)
    conv5_3 = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_3')(conv5_2)
    pool5 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='pool5')(conv5_3)

    fc6 = Conv2D(512, (3, 3), dilation_rate=(6, 6), activation='relu', padding='same', name='fc6')(pool5)

    fc7 = Conv2D(512, (1, 1), activation='relu', padding='same', name='fc7')(fc6)

    conv6_1 = Conv2D(128, (1, 1), activation='relu', padding='same', name='conv6_1')(fc7)
    conv6_2 = Conv2D(256, (3, 3), strides=(2, 2), activation='relu', padding='same', name='conv6_2')(conv6_1)

    conv7_1 = Conv2D(64, (1, 1), activation='relu', padding='same', name='conv7_1')(conv6_2)
    conv7_2 = Conv2D(128, (3, 3), strides=(2, 2), activation='relu', padding='same', name='conv7_2')(conv7_1)

    conv8_1 = Conv2D(64, (1, 1), activation='relu', padding='same', name='conv8_1')(conv7_2)
    conv8_2 = Conv2D(128, (3, 3), strides=(1, 1), activation='relu', padding='same', name='conv8_2')(conv8_1)

    conv9_1 = Conv2D(64, (1, 1), activation='relu', padding='same', name='conv9_1')(conv8_2)
    conv9_2 = Conv2D(128, (3, 3), strides=(1, 1), activation='relu', padding='same', name='conv9_2')(conv9_1)


    Conv3_3_norm = conv5_3
    Conv4_3 = conv6_2
    Conv5_3 = conv7_2
    Conv6_3 = conv8_2
    Conv7_3 = conv9_2

    #output shape is (batch, nbox_total, 2class)
    Conv3_3_norm_mbox_conf = Conv2D(n_boxes_conv3_3 * n_classes, (3, 3), padding='same', name='Conv3_3_norm_mbox_conf')(Conv3_3_norm)
    Conv4_3_mbox_conf = Conv2D(n_boxes_conv4_3 * n_classes, (3, 3), padding='same', name='Conv4_3_mbox_conf')(Conv4_3)
    Conv5_3_mbox_conf = Conv2D(n_boxes_conv5_3 * n_classes, (3, 3), padding='same', name='Conv5_3_mbox_conf')(Conv5_3)
    Conv6_3_mbox_conf = Conv2D(n_boxes_conv6_3 * n_classes, (3, 3), padding='same', name='Conv6_3_mbox_conf')(Conv6_3)
    Conv7_3_mbox_conf = Conv2D(n_boxes_conv7_3 * n_classes, (3, 3), padding='same', name='Conv7_3_mbox_conf')(Conv7_3)

    Conv3_3_norm_mbox_conf_reshape = Reshape((-1, n_classes), name='Conv3_3_norm_mbox_conf_reshape')(Conv3_3_norm_mbox_conf)
    Conv4_3_mbox_conf_reshape = Reshape((-1, n_classes), name='Conv4_3_mbox_conf_reshape')(Conv4_3_mbox_conf)
    Conv5_3_mbox_conf_reshape = Reshape((-1, n_classes), name='Conv5_3_mbox_conf_reshape')(Conv5_3_mbox_conf)
    Conv6_3_mbox_conf_reshape = Reshape((-1, n_classes), name='Conv6_3_mbox_conf_reshape')(Conv6_3_mbox_conf)
    Conv7_3_mbox_conf_reshape = Reshape((-1, n_classes), name='Conv7_3_mbox_conf_reshape')(Conv7_3_mbox_conf)

    mbox_conf = Concatenate(axis=1, name='mbox_conf')([Conv3_3_norm_mbox_conf_reshape,
                                                       Conv4_3_mbox_conf_reshape,
                                                       Conv5_3_mbox_conf_reshape,
                                                       Conv6_3_mbox_conf_reshape,
                                                       Conv7_3_mbox_conf_reshape])

    mbox_conf_softmax = Activation('softmax', name='mbox_conf_softmax')(mbox_conf)
    model_ssd= Model(inputs=img_input,outputs=mbox_conf_softmax)


    if os.path.exists(SSDpath):
        model_ssd.load_weights(SSDpath)
    else:
        raise TypeError("SSDpath" + "is not exist!")
    predictor_sizes = np.array([Conv3_3_norm_mbox_conf._keras_shape[1:3],
                                 Conv4_3_mbox_conf._keras_shape[1:3],
                                 Conv5_3_mbox_conf._keras_shape[1:3],
                                Conv6_3_mbox_conf._keras_shape[1:3],
								Conv7_3_mbox_conf._keras_shape[1:3]])
    #网络结合
    model = Model(inputs=img_input,outputs=[model_Seg.output,model_ssd.output])
    return model,predictor_sizes


if __name__ == '__main__':
    model = FCN8(128)
    model.summary()
    print('load successfully')
    # from keras.utils import plot_model
    # plot_model( model , show_shapes=True , to_file='model.png')
