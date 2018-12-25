# encoding:utf-8
from keras.applications.mobilenet import *
from keras.models import *
from keras.layers import *
from keras_layer_L2Normalization import *
import os

def FCN8_MobileNet(nClasses=128,iflevel = False):
    input_shape = (224,224,3)
    base_model = MobileNet(include_top=False, input_shape=input_shape)
    img_input = base_model.input
    # f1 = base_model.get_layer('block1_pool').output    #112*112
    # f2 = base_model.get_layer('block2_pool').output    #56*56
    f3 = base_model.get_layer('conv_pw_5_relu').output    #28*28
    f4 = base_model.get_layer('conv_pw_11_relu').output    #14*14
    f5 = base_model.get_layer('conv_pw_13_relu').output    #7*7

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
    # midname = 'block2_conv1'  #block3_conv1
    # for layers in model.layers:
    #     layers.trainable = False
    #     if layers.name == midname:
    #         break
    return model


if __name__ == '__main__':
    model = FCN8_MobileNet(128)
    model.summary()
    print('load successfully')
    # from keras.utils import plot_model
    # plot_model( model , show_shapes=True , to_file='model.png')
