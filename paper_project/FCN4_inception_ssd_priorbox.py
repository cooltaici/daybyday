from keras.applications.vgg16 import VGG16
from keras.models import *
from keras.layers import *
from keras_layer_L2Normalization import *
from keras_layer_AnchorBoxes import *
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
def fcn4_inception3(isnorm = True,
                    image_size = (224,224),
                    n_classes = 2,
                    min_scale=0.1,
                    max_scale=0.8,
                    aspect_ratios_global=None,
                    aspect_ratios_per_layer=[[0.5, 1.0, 2.0],
                                            [1.0/3.0, 0.5, 1.0, 2.0, 3.0],
                                            [0.5, 1.0, 2.0]],
                    two_boxes_for_ar1=True,
                    limit_boxes=False,
                    variances=[0.1, 0.1, 0.2, 0.2],
                    coords='centroids',
                    normalize_coords=False):
    scales = np.linspace(min_scale, max_scale, len(aspect_ratios_per_layer)+1)
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

    nclass = 128
    input_shape = (224,224,3)
    base_model = VGG16(include_top=None, pooling=None, weights='imagenet', input_shape = input_shape)
    img_input = base_model.input
    #f1 = base_model.get_layer('block1_pool').output    #112*112*64
    f2 = base_model.get_layer('block2_pool').output    #56*56*128 从这里开始可以增加分辨率
    f3 = base_model.get_layer('block3_pool').output    #28*28*256
    f4 = base_model.get_layer('block4_pool').output    #14*14*512
    f5 = base_model.get_layer('block5_pool').output    #7*7*512
    #SSD
    Conv3_3 = base_model.get_layer('block3_conv3').output
    Conv4_3 = base_model.get_layer('block4_conv3').output
    Conv5_3 = base_model.get_layer('block5_conv3').output
    Conv3_3_norm = L2Normalization(gamma_init=20, name='conv4_3_norm')(Conv3_3)

    #output shape is (batch, nbox_total, 2class)
    Conv3_3_norm_mbox_conf = Conv2D(n_boxes_conv3_3 * n_classes, (3, 3), padding='same', name='conv4_3_norm_mbox_conf')(Conv3_3_norm)
    Conv4_3_mbox_conf = Conv2D(n_boxes_conv4_3 * n_classes, (3, 3), padding='same', name='fc7_mbox_conf')(Conv4_3)
    Conv5_3_mbox_conf = Conv2D(n_boxes_conv5_3 * n_classes, (3, 3), padding='same', name='conv6_2_mbox_conf')(Conv5_3)

    Conv3_3_norm_mbox_loc = Conv2D(n_boxes_conv4_3 * 4, (3, 3), padding='same', name='conv4_3_norm_mbox_loc')(Conv3_3_norm)
    Conv4_3_mbox_loc = Conv2D(n_boxes_conv4_3 * 4, (3, 3), padding='same', name='fc7_mbox_loc')(Conv4_3)
    Conv5_3_mbox_loc = Conv2D(n_boxes_conv5_3 * 4, (3, 3), padding='same', name='conv6_2_mbox_loc')(Conv5_3)

    #output shape is (batch, nbox_total, 8)   4cord+4varience，先验框，这里其实不需要吧？

    Conv3_3_norm_mbox_priorbox = AnchorBoxes(224, 224,
                                             this_scale=scales[0], next_scale=scales[1],
                                             aspect_ratios=aspect_ratios_per_layer[0],
                                             two_boxes_for_ar1=two_boxes_for_ar1,
                                             limit_boxes=limit_boxes,
                                             variances=variances,
                                             coords=coords,
                                             normalize_coords=normalize_coords,
                                             name='conv4_3_norm_mbox_priorbox')(Conv3_3_norm_mbox_loc)
    Conv4_3_mbox_priorbox = AnchorBoxes(224, 224,
                                        this_scale=scales[1], next_scale=scales[2],
                                        aspect_ratios=aspect_ratios_per_layer[1],
                                        two_boxes_for_ar1=two_boxes_for_ar1,
                                        limit_boxes=limit_boxes,
                                        variances=variances,
                                        coords=coords,
                                        normalize_coords=normalize_coords,
                                        name='fc7_mbox_priorbox')(Conv4_3_mbox_loc)
    Conv5_3_mbox_priorbox = AnchorBoxes(224, 224,
                                        this_scale=scales[2], next_scale=scales[3],
                                        aspect_ratios=aspect_ratios_per_layer[2],
                                        two_boxes_for_ar1=two_boxes_for_ar1,
                                        limit_boxes=limit_boxes,
                                        variances=variances,
                                        coords=coords,
                                        normalize_coords=normalize_coords,
                                        name='conv6_2_mbox_priorbox')(Conv5_3_mbox_loc)

    o = f5
    #o = Conv2D( 256 , ( 7 , 7 ) , activation='relu' , padding='same')(o)  # 原来的值4096
    o = conv2d_bn(o, 256, 7, 7)
    o = Dropout(0.5)(o)
    #o = Conv2D( 256 , ( 1 , 1 ) , activation='relu' , padding='same')(o)  # 原来的值4096
    o = conv2d_bn(o, 256, 1, 1)
    o = Dropout(0.5)(o)

    o = conv2d_bn(o, nclass, 1, 1, isnorm)
    #o = Conv2D( nclass ,  ( 1 , 1 ) ,kernel_initializer='he_normal', padding='same')(o)
    o = Conv2DTranspose( nclass , kernel_size=(4,4) ,  strides=(2,2) , padding='same', use_bias=False)(o)

    ##################################################
    o4 = f4
    #f4branch1x1 = Conv2D(nclass, (1, 1), kernel_initializer='he_normal', padding='same')(o4)
    f4branch1x1 = conv2d_bn(o4, nclass, 1, 1, isnorm)

    f4branch3x3db = conv2d_bn(o4, 64, 3, 3)
    f4branch3x3db = conv2d_bn(f4branch3x3db, nclass, 1, 1)
    f4branch5x5db = conv2d_bn(o4, 64, 5, 5)
    f4branch5x5db = conv2d_bn(f4branch5x5db, nclass, 1, 1)

    # o4 = concatenate([f4branch1x1, f4branch3x3db, f4branch5x5db],axis = 3,name = 'mixed0')
    # o4 = conv2d_bn(o4, nclass, 1, 1, isnorm)
    o = Add()([o,f4branch1x1,f4branch3x3db,f4branch5x5db])

    o = Conv2DTranspose( nclass , kernel_size=(4,4) ,  strides=(2,2) , padding='same', use_bias=False)(o)
    ####################################################
    o3 = f3
    #f3branch1x1 = Conv2D(nclass, (1, 1), kernel_initializer='he_normal', padding='same')(o3)
    f3branch1x1 = conv2d_bn(o3, nclass, 1, 1, isnorm)
    # 要不要卷积分解？卷积分解可以减小训练参数，加快计算时间
    f3branch3x3db = conv2d_bn(o3, 64, 3, 3, isnorm)
    f3branch3x3db = conv2d_bn(f3branch3x3db, nclass, 1, 1, isnorm)
    f3branch5x5db = conv2d_bn(o3, 64, 5, 5)
    f3branch5x5db = conv2d_bn(f3branch5x5db, nclass, 1, 1)

    # o3 = concatenate([branch1x1, branch3x3db, branch5x5db, branch_pool],axis = 3,name = 'mixed1')
    # o3 = conv2d_bn(o3, nclass, 1, 1, isnorm)
    o = Add()([o, f3branch1x1, f3branch3x3db,f3branch5x5db])

    o = Conv2DTranspose(nclass, kernel_size=(4,4) ,  strides=(2,2) , padding='same', use_bias=False)(o)
    ####################################################
    o2 = f2
    f2branch1x1 = conv2d_bn(o2, nclass, 1, 1, isnorm)
    # 要不要卷积分解？卷积分解可以减小训练参数，加快计算时间
    f2branch3x3db = conv2d_bn(o2, 64, 3, 3, isnorm)
    f2branch3x3db = conv2d_bn(f2branch3x3db, nclass, 1, 1, isnorm)
    f2branch5x5db = conv2d_bn(o2, 64, 5, 5)
    f2branch5x5db = conv2d_bn(f2branch5x5db, nclass, 1, 1)

    # o2 = concatenate([f2branch1x1, f2branch3x3db, f2branch5x5db, branch_pool],axis = 3,name = 'mixed2')
    # o2 = conv2d_bn(o2, nclass, 1, 1, isnorm)
    o = Add()([o, f2branch1x1, f2branch3x3db,f2branch5x5db])

    o = Conv2DTranspose( 1 , kernel_size=(8,8) ,  strides=(4,4) , padding='same', use_bias=False,activation='sigmoid')(o)
    ####################################################
    model = Model(img_input , o )
    midname = 'block1_conv2'
    for layers in model.layers:
        layers.trainable = False
        if layers.name == midname:
            break
    return model

#测试输出没问题
if __name__ == '__main__':
    model = fcn4_inception3(False)
    model.summary()
    print('over!')