from __future__ import print_function
from keras.applications.resnet50 import *
from math import ceil
from keras.models import Model
from keras.layers import *
from keras import layers
from keras_layer_L2Normalization import *
from keras.backend import tf as ktf
import os
from RefineNet.utils.utils import *

def RefineNet_ResNet50(input_shape = (224,224,3), upscaling_method="bilinear"):
	base_model = ResNet50(include_top=False,weights="imagenet",input_tensor=None,input_shape=input_shape)
	base_model.summary()
	#model_base.trainable = False
	high = [base_model.get_layer('activation_49').output,
			base_model.get_layer('activation_40').output,
			base_model.get_layer('activation_22').output,
			base_model.get_layer('activation_10').output]
	low = [None, None, None]

	# Get the feature maps to the proper size with bottleneck
	high[0] = Conv2D(512, (1, 1), padding='same')(high[0]) #7*7
	high[1] = Conv2D(256, (1, 1), padding='same')(high[1])
	high[2] = Conv2D(256, (1, 1), padding='same')(high[2])
	high[3] = Conv2D(256, (1, 1), padding='same')(high[3])

	# RefineNet
	low[0] = RefineBlock(high_inputs=high[0], low_inputs=None)  # Only input ResNet 1/32
	low[1] = RefineBlock(high[1], low[0])  # High input = ResNet 1/16, Low input = Previous 1/16
	low[2] = RefineBlock(high[2], low[1])  # High input = ResNet 1/8, Low input = Previous 1/8
	net = RefineBlock(high[3], low[2])  # High input = ResNet 1/4, Low input = Previous 1/4.

	# g[3]=Upsampling(g[3],scale=4)

	net = ResidualConvUnit(net)
	net = ResidualConvUnit(net)

	if upscaling_method.lower() == "conv":
		net = ConvUpscaleBlock(net, 128, kernel_size=[3, 3], scale=2)
		net = ConvBlock(net, 128, padding='same')
		net = ConvUpscaleBlock(net, 64, kernel_size=[3, 3], scale=2)
		net = ConvBlock(net, 64, padding='same')

	elif upscaling_method.lower() == "bilinear":
		net = Upsampling(scale=4)(net)

	net = Conv2D(1, 1, activation='sigmoid')(net)  # 可以使用softmax
	model = Model(base_model.input, net)
	#fixed weights
	layername = r"activation_7"  #activation_4， activation_10
	# for layers in model.layers:
	# 	layers.trainable = False
	# 	if layers.name == layername:
	# 		break
	return model

if __name__ == "__main__":
	model = RefineNet_ResNet50(input_shape = (224,224,3), upscaling_method="conv")
	model.summary()
	print("over")