from __future__ import print_function
from keras.applications.resnet50 import *
from math import ceil
from keras.models import Model
from keras.layers import *
from keras import layers
from keras_layer_L2Normalization import *
from keras.backend import tf as ktf
import os
from PSPNet.utils.utils import *

def PSPNet_ResNet50(input_shape = (224,224,3)):
	base_model = ResNet50(include_top=False,weights="imagenet",input_tensor=None,input_shape=input_shape)
	activation_22 = base_model.get_layer("activation_22")
	#block4, output_stride=8
	x = atro_conv_block(activation_22, 3, [256, 256, 1024], stage=4, block='a',dilation_rate= (2,2))
	x = atro_identity_block(x, 3, [256, 256, 1024], stage=4, block='b',dilation_rate= (2,2))
	x = atro_identity_block(x, 3, [256, 256, 1024], stage=4, block='c',dilation_rate= (2,2))
	x = atro_identity_block(x, 3, [256, 256, 1024], stage=4, block='d',dilation_rate= (2,2))
	x = atro_identity_block(x, 3, [256, 256, 1024], stage=4, block='e',dilation_rate= (2,2))
	x = atro_identity_block(x, 3, [256, 256, 1024], stage=4, block='f',dilation_rate= (2,2))
	#block5, output_stride=8
	x = atro_conv_block(x, 3, [512, 512, 2048], stage=5, block='a',dilation_rate= (4,4))
	x = atro_identity_block(x, 3, [512, 512, 2048], stage=5, block='b',dilation_rate= (4,4))
	x = atro_identity_block(x, 3, [512, 512, 2048], stage=5, block='c',dilation_rate= (4,4))
	pspout = build_pyramid_pooling_module(x, input_shape)

	output = Conv2D(512, (3, 3), strides=(1, 1), padding="same", name="conv5_4",use_bias=False)(pspout)
	output = BatchNormalization(momentum=0.95, epsilon=1e-5)(output)
	output = Activation('relu')(output)
	output = Dropout(0.1)(output)
	output = Conv2D(56, (1, 1), padding='same', name="conv6")(output)
	output = Conv2DTranspose(1,(8,8),strides=(4,4),padding="same",use_bias=False,activation='sigmoid')(output)

	model = Model(input=base_model.inputs, outputs=output)
	#fixed weights
	layername = r"activation_4"
	for layers in model.layers:
		layers.trainable = False
		if layers.name == layername:
			break
	return model

if __name__ == "__main__":
	model = PSPNet_ResNet50()
	model.summary()
	print("over")