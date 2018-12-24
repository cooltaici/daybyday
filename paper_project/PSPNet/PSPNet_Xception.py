from keras.applications.xception import *
from keras.models import *
from keras.layers import *
from keras import layers
from keras_layer_L2Normalization import *
from keras.backend import tf as ktf
import os
from PSPNet.utils.utils import *
r"https://github.com/fchollet/deep-learning-models/releases/download/v0.4/xception_weights_tf_dim_ordering_tf_kernels_notop.h5"
def PSPNet_Xception(input_shape = (224,224,3)):
    base_model = Xception(input_shape=input_shape,weights="imagenet",include_top=False)
    # back_bone can all be fixed
    for layers in base_model.layers:
        layers.trainable = False
    block2_pool = base_model.get_layer("block2_pool")
    # block4, output_stride=8
    x = atro_conv_block(block2_pool, 3, [256, 256, 1024], stage=4, block='a', dilation_rate=(2, 2))
    x = atro_identity_block(x, 3, [256, 256, 1024], stage=4, block='b', dilation_rate=(2, 2))
    x = atro_identity_block(x, 3, [256, 256, 1024], stage=4, block='c', dilation_rate=(2, 2))
    x = atro_identity_block(x, 3, [256, 256, 1024], stage=4, block='d', dilation_rate=(2, 2))
    x = atro_identity_block(x, 3, [256, 256, 1024], stage=4, block='e', dilation_rate=(2, 2))
    x = atro_identity_block(x, 3, [256, 256, 1024], stage=4, block='f', dilation_rate=(2, 2))
    # block5, output_stride=8
    x = atro_conv_block(x, 3, [512, 512, 2048], stage=5, block='a', dilation_rate=(4, 4))
    x = atro_identity_block(x, 3, [512, 512, 2048], stage=5, block='b', dilation_rate=(4, 4))
    x = atro_identity_block(x, 3, [512, 512, 2048], stage=5, block='c', dilation_rate=(4, 4))
    pspout = build_pyramid_pooling_module(x, input_shape)

    output = Conv2D(512, (3, 3), strides=(1, 1), padding="same", name="conv5_4", use_bias=False)(pspout)
    output = BatchNormalization(momentum=0.95, epsilon=1e-5)(output)
    output = Activation('relu')(output)
    output = Dropout(0.1)(output)
    output = Conv2D(56, (1, 1), padding='same', name="conv6")(output)
    output = Conv2DTranspose(1, (8, 8), strides=(4, 4), padding="same", use_bias=False, activation='sigmoid')(output)

    model = Model(input=base_model.inputs, outputs=output)
    # fixed weights
    # layername = r"activation_4"
    # for layers in model.layers:
    #     layers.trainable = False
    #     if layers.name == layername:
    #         break
    return model

if __name__ == "__main__":
    model = PSPNet_Xception()
    model.summary()
    print("over")