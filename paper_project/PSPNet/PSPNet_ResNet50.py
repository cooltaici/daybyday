from __future__ import print_function
from keras.applications.resnet50 import *
from math import ceil
from keras.models import Model
from keras.layers import *
from keras import layers
from keras_layer_L2Normalization import *
from keras.backend import tf as ktf
import os

#孔洞卷积恒等映射映射
def atro_identity_block(input_tensor, kernel_size, filters, stage, block,dilation_rate= (1,1)):
	nb_filter1, nb_filter2, nb_filter3 = filters
	if K.image_dim_ordering() == 'tf':
		bn_axis = 3
	else:
		bn_axis = 1
	conv_name_base = 'res' + str(stage) + block + '_branch'
	bn_name_base = 'bn' + str(stage) + block + '_branch'

	x = Convolution2D(nb_filter1, 1, 1, name=conv_name_base + '2a')(input_tensor)
	x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
	x = Activation('relu')(x)

	x = Convolution2D(nb_filter2, kernel_size, kernel_size,
					  border_mode='same', name=conv_name_base + '2b',dilation_rate=dilation_rate)(x)  #改为空洞卷积
	x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
	x = Activation('relu')(x)

	x = Convolution2D(nb_filter3, 1, 1, name=conv_name_base + '2c')(x)
	x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

	x = merge([x, input_tensor], mode='sum')
	x = Activation('relu')(x)
	return x

#孔洞卷积卷积映射
def atro_conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2),dilation_rate= (1,1)):
	nb_filter1, nb_filter2, nb_filter3 = filters
	if K.image_dim_ordering() == 'tf':
		bn_axis = 3
	else:
		bn_axis = 1
	conv_name_base = 'res' + str(stage) + block + '_branch'
	bn_name_base = 'bn' + str(stage) + block + '_branch'

	x = Convolution2D(nb_filter1, 1, 1, subsample=strides,
					  name=conv_name_base + '2a')(input_tensor)
	x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
	x = Activation('relu')(x)

	x = Convolution2D(nb_filter2, kernel_size, kernel_size, border_mode='same',
					  name=conv_name_base + '2b',dilation_rate=dilation_rate)(x)   #改为空洞卷积
	x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
	x = Activation('relu')(x)

	x = Convolution2D(nb_filter3, 1, 1, name=conv_name_base + '2c')(x)
	x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

	shortcut = Convolution2D(nb_filter3, 1, 1, subsample=strides,
							 name=conv_name_base + '1')(input_tensor)
	shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

	x = merge([x, shortcut], mode='sum')
	x = Activation('relu')(x)
	return x

#
def interp_block(prev_layer, level, feature_map_shape, input_shape, output_stride=8.0):

	kernel_strides = (int(i/output_stride/level) for i in input_shape)
	names = [
		"conv5_3_pool" + str(level) + "_conv",
		"conv5_3_pool" + str(level) + "_conv_bn"
	]
	prev_layer = AveragePooling2D(kernel_strides, strides=kernel_strides)(prev_layer)

	prev_layer = Convolution2D(512, (1, 1), strides=(1, 1), name=names[0],use_bias=False)(prev_layer)
	prev_layer = BatchNormalization(momentum=0.95, name=names[1], epsilon=1e-5)(prev_layer)
	prev_layer = Activation('relu')(prev_layer)

	#这里可以改为卷积映射
	prev_layer = ktf.image.resize_images(prev_layer, feature_map_shape,align_corners=True)

	return prev_layer

def build_pyramid_pooling_module(res, input_shape, output_stride=8.0):
	"""Build the Pyramid Pooling Module."""
	# ---PSPNet concat layers with Interpolation
	feature_map_size = tuple(int(ceil(input_dim / output_stride))
							 for input_dim in input_shape)
	print("PSP module will interpolate to a final feature map size of %s" %
		  (feature_map_size, ))

	interp_block1 = interp_block(res, 1, feature_map_size, input_shape,output_stride=output_stride)
	interp_block2 = interp_block(res, 2, feature_map_size, input_shape,output_stride=output_stride)
	interp_block3 = interp_block(res, 3, feature_map_size, input_shape,output_stride=output_stride)
	interp_block6 = interp_block(res, 6, feature_map_size, input_shape,output_stride=output_stride)

	# concat all these layers. resulted
	# shape=(1,feature_map_size_x,feature_map_size_y,4096)
	res = merge([res,
				 interp_block6,
				 interp_block3,
				 interp_block2,
				 interp_block1],mode="concat")
	return res

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
	#金字塔池化
    ouput = Conv2D(512, (3, 3), strides=(1, 1), padding="same", name="conv5_4",use_bias=False)(pspout)
	ouput = BatchNormalization(momentum=0.95, epsilon=1e-5)(ouput)
	ouput = Activation('relu')(ouput)
	ouput = Dropout(0.1)(ouput)
	ouput = Conv2D(56, (1, 1), padding='same', name="conv6")(ouput)
	#ouput = Conv2DTranspose(1,(8,8),strides=(4,4),padding="same",use_bias=False,activation='sigmoid')(ouput)

	model = Model(input=base_model.inputs)

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