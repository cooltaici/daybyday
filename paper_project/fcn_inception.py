from keras.applications.vgg16 import VGG16
from keras.models import *
from keras.layers import *

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
def fcn4_inception3(isnorm = True):
    nclass = 128
    input_shape = (224,224,3)
    base_model = VGG16(include_top=None, pooling=None, weights='imagenet', input_shape = input_shape)
    img_input = base_model.input
    #f1 = base_model.get_layer('block1_pool').output    #112*112*64
    f2 = base_model.get_layer('block2_pool').output    #56*56*128 从这里开始可以增加分辨率
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

    o = conv2d_bn(o, nclass, 1, 1, isnorm)
    #o = Conv2D( nclass ,  ( 1 , 1 ) ,kernel_initializer='he_normal', padding='same')(o)
    o = Conv2DTranspose( nclass , kernel_size=(4,4) ,  strides=(2,2) , padding='same', use_bias=False)(o)

    ##################################################
    o4 = f4
    #f4branch1x1 = Conv2D(nclass, (1, 1), kernel_initializer='he_normal', padding='same')(o4)
    f4branch1x1 = conv2d_bn(o4, nclass, 1, 1, isnorm)

    f4branch3x3db = conv2d_bn(o4, 64, 3, 3, isnorm)
    f4branch3x3db = conv2d_bn(f4branch3x3db, nclass, 1, 1, isnorm)
    f4branch5x5db = conv2d_bn(o4, 64, 5, 5, isnorm)
    f4branch5x5db = conv2d_bn(f4branch5x5db, nclass, 1, 1, isnorm)

    # o4 = concatenate([f4branch1x1, f4branch3x3db, f4branch5x5db],axis = 3,name = 'mixed0')
    # o4 = conv2d_bn(o4, nclass, 1, 1, isnorm)
    o = Add()([o,f4branch1x1,f4branch3x3db,f4branch5x5db])

    o = Conv2DTranspose( nclass , kernel_size=(4,4) ,  strides=(2,2) , padding='same', use_bias=False)(o)
    ####################################################
    o3 = f3
    #f3branch1x1 = Conv2D(nclass, (1, 1), kernel_initializer='he_normal', padding='same')(o3)
    f3branch1x1 = conv2d_bn(o3, nclass, 1, 1)
    # 要不要卷积分解？卷积分解可以减小训练参数，加快计算时间
    f3branch3x3db = conv2d_bn(o3, 64, 3, 3)
    f3branch3x3db = conv2d_bn(f3branch3x3db, nclass, 1, 1)
    f3branch5x5db = conv2d_bn(o3, 64, 5, 5,isnorm)
    f3branch5x5db = conv2d_bn(f3branch5x5db, nclass, 1, 1,isnorm)

    #o3 = concatenate([branch1x1, branch3x3db, branch5x5db, branch_pool],axis = 3,name = 'mixed1')
    #o3 = conv2d_bn(o3, nclass, 1, 1, isnorm)
    o = Add()([o, f3branch1x1, f3branch3x3db,f3branch5x5db])
    #o = Add()([o, f3branch1x1, f3branch3x3db])

    o = Conv2DTranspose(nclass, kernel_size=(4,4) ,  strides=(2,2) , padding='same', use_bias=False)(o)
    ####################################################
    o2 = f2
    f2branch1x1 = conv2d_bn(o2, nclass, 1, 1, isnorm)
    # 要不要卷积分解？卷积分解可以减小训练参数，加快计算时间
    f2branch3x3db = conv2d_bn(o2, 64, 3, 3, isnorm)
    f2branch3x3db = conv2d_bn(f2branch3x3db, nclass, 1, 1, isnorm)
    f2branch5x5db = conv2d_bn(o2, 64, 5, 5, isnorm)
    f2branch5x5db = conv2d_bn(f2branch5x5db, nclass, 1, 1, isnorm)

    # o2 = concatenate([f2branch1x1, f2branch3x3db, f2branch5x5db, branch_pool],axis = 3,name = 'mixed2')
    # o2 = conv2d_bn(o2, nclass, 1, 1, isnorm)
    o = Add()([o, f2branch1x1, f2branch3x3db,f2branch5x5db])

    o = Conv2DTranspose( 1 , kernel_size=(8,8) ,  strides=(4,4) , padding='same',name="Seg_output", use_bias=False,activation='sigmoid')(o)
    ####################################################
    model = Model(img_input , o )
    midname = 'block1_conv2'
    for layers in model.layers:
        layers.trainable = False
        if layers.name == midname:
            break
    return model

#事实证明fcn8>fcn4
def fcn4_inception3_more_fullnet(isnorm = True):
    nclass = 128
    input_shape = (224,224,3)
    base_model = VGG16(include_top=None, pooling=None, weights='imagenet', input_shape = input_shape)
    img_input = base_model.input
    #f1 = base_model.get_layer('block1_pool').output    #112*112*64
    f2 = base_model.get_layer('block2_pool').output    #56*56*128 从这里开始可以增加分辨率
    f3 = base_model.get_layer('block3_pool').output    #28*28*256
    f4 = base_model.get_layer('block4_pool').output    #14*14*512
    f5 = base_model.get_layer('block5_pool').output    #7*7*512

    o = f5
    #o = Conv2D( 256 , ( 7 , 7 ) , activation='relu' , padding='same')(o)  # 原来的值4096
    o = conv2d_bn(o, 1024, 7, 7, isnorm)
    o = Dropout(0.5)(o)
    #o = Conv2D( 256 , ( 1 , 1 ) , activation='relu' , padding='same')(o)  # 原来的值4096
    o = conv2d_bn(o, 1024, 1, 1, isnorm)
    o = Dropout(0.5)(o)

    o = conv2d_bn(o, nclass, 1, 1)
    #o = Conv2D( nclass ,  ( 1 , 1 ) ,kernel_initializer='he_normal', padding='same')(o)
    o = Conv2DTranspose( nclass , kernel_size=(4,4) ,  strides=(2,2) , padding='same', use_bias=False)(o)

    ##################################################
    o4 = f4
    #f4branch1x1 = Conv2D(nclass, (1, 1), kernel_initializer='he_normal', padding='same')(o4)
    f4branch1x1 = conv2d_bn(o4, nclass, 1, 1)

    f4branch3x3db = conv2d_bn(o4, nclass, 3, 3, isnorm)
    f4branch3x3db = conv2d_bn(f4branch3x3db, nclass, 1, 1)

    o = Add()([o,f4branch1x1,f4branch3x3db])

    o = Conv2DTranspose( nclass , kernel_size=(4,4) ,  strides=(2,2) , padding='same', use_bias=False)(o)
    ####################################################
    o3 = f3
    #f3branch1x1 = Conv2D(nclass, (1, 1), kernel_initializer='he_normal', padding='same')(o3)
    f3branch1x1 = conv2d_bn(o3, nclass, 1, 1)
    # 要不要卷积分解？卷积分解可以减小训练参数，加快计算时间
    f3branch3x3db = conv2d_bn(o3, nclass, 3, 3, isnorm)
    f3branch3x3db = conv2d_bn(f3branch3x3db, nclass, 1, 1)

    o = Add()([o, f3branch1x1, f3branch3x3db])

    o = Conv2DTranspose(nclass, kernel_size=(4,4) ,  strides=(2,2) , padding='same', use_bias=False)(o)
    ####################################################
    o2 = f2
    f2branch1x1 = conv2d_bn(o2, nclass, 1, 1)
    # 要不要卷积分解？卷积分解可以减小训练参数，加快计算时间
    f2branch3x3db = conv2d_bn(o2, nclass, 5, 5, isnorm)
    f2branch3x3db = conv2d_bn(f2branch3x3db, nclass, 1, 1)

    # o2 = concatenate([f2branch1x1, f2branch3x3db, f2branch5x5db, branch_pool],axis = 3,name = 'mixed2')
    # o2 = conv2d_bn(o2, nclass, 1, 1, isnorm)
    o = Add()([o, f2branch1x1, f2branch3x3db])

    o = Conv2DTranspose( 1 , kernel_size=(8,8) ,  strides=(4,4) , padding='same',name="Seg_output", use_bias=False,activation='sigmoid')(o)
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