from keras.applications.xception import *
from keras.models import *
from keras.layers import *
from keras import layers
from keras_layer_L2Normalization import *
from keras.backend import tf as ktf
import os

#简单的双线性插值上采样
class Interp(layers.Layer):
	def __init__(self, new_size, **kwargs):
		self.new_size = new_size
		super(Interp, self).__init__(**kwargs)

	def build(self, input_shape):
		super(Interp, self).build(input_shape)

	def call(self, inputs, **kwargs):
		new_height, new_width = self.new_size
		resized = ktf.image.resize_images(inputs, [new_height, new_width],
										  align_corners=True)
		return resized

	def compute_output_shape(self, input_shape):
		return tuple([None, self.new_size[0], self.new_size[1], input_shape[3]])

	def get_config(self):
		config = super(Interp, self).get_config()
		config['new_size'] = self.new_size
		return config

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