from __future__ import print_function
from math import ceil
from keras import layers
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers import BatchNormalization, Activation, Input, Dropout, ZeroPadding2D, Lambda, Permute, Reshape, Conv2DTranspose
from keras.layers.merge import Concatenate, Add
from keras.models import Model
from keras.optimizers import SGD
from keras.backend import tf as ktf
import keras.backend as K
import tensorflow as tf

learning_rate = 1e-3  # Layer specific learning rate
# Weight decay not implemented


def BN(name=""):
    return BatchNormalization(momentum=0.95, name=name, epsilon=1e-5)

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

#重排上采样：
def duc(x, factor=8, output_shape=(512, 512, 1),n_labels=21,output_stride=8):
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    H, W, c, r = output_shape[0], output_shape[1], output_shape[2], factor
    h = H / r
    w = W / r
    x = Conv2D(
            c*r*r,
            (3, 3),
            padding='same',
            name='conv_duc_%s' % factor)(x)
    x = BatchNormalization(axis=bn_axis, name='bn_duc_%s' % factor)(x)
    x = Activation('relu')(x)
    x = Permute((3, 1, 2))(x)
    x = Reshape((c, r, r, h, w))(x)
    x = Permute((1, 4, 2, 5, 3))(x)
    x = Reshape((c, H, W))(x)
    x = Permute((2, 3, 1))(x)

    out = Conv2D(
            filters=n_labels,
            kernel_size=(1, 1),
            padding='same',
            name='out_duc_%s' % output_stride)(x)
    return out

#反卷积上采样
def DeConv2D(input,fileters,output_stride=8,name="upscore"):
    out = Conv2DTranspose(
                    filters=fileters,
                    kernel_size=(output_stride*2, output_stride*2),
                    strides=(output_stride, output_stride),
                    padding='same',
                    kernel_initializer='he_normal',
                    kernel_regularizer=None,
                    use_bias=False,
                    name=name)(input)
    return out

#短连接：主干分支
def residual_conv(prev, level, pad=1, lvl=1, sub_lvl=1, modify_stride=False):
    lvl = str(lvl)
    sub_lvl = str(sub_lvl)
    names = ["conv" + lvl + "_" + sub_lvl + "_1x1_reduce",
             "conv" + lvl + "_" + sub_lvl + "_1x1_reduce_bn",
             "conv" + lvl + "_" + sub_lvl + "_3x3",
             "conv" + lvl + "_" + sub_lvl + "_3x3_bn",
             "conv" + lvl + "_" + sub_lvl + "_1x1_increase",
             "conv" + lvl + "_" + sub_lvl + "_1x1_increase_bn"]
    if modify_stride is False:
        prev = Conv2D(64 * level, (1, 1), strides=(1, 1), name=names[0],
                      use_bias=False)(prev)
    elif modify_stride is True:
        prev = Conv2D(64 * level, (1, 1), strides=(2, 2), name=names[0],
                      use_bias=False)(prev)

    prev = BN(name=names[1])(prev)
    prev = Activation('relu')(prev)

    prev = ZeroPadding2D(padding=(pad, pad))(prev)
    prev = Conv2D(64 * level, (3, 3), strides=(1, 1), dilation_rate=pad,
                  name=names[2], use_bias=False)(prev)

    prev = BN(name=names[3])(prev)
    prev = Activation('relu')(prev)
    prev = Conv2D(256 * level, (1, 1), strides=(1, 1), name=names[4],
                  use_bias=False)(prev)
    prev = BN(name=names[5])(prev)
    return prev

#短连接：卷积分支
def short_convolution_branch(prev, level, lvl=1, sub_lvl=1, modify_stride=False):
    lvl = str(lvl)
    sub_lvl = str(sub_lvl)
    names = ["conv" + lvl + "_" + sub_lvl + "_1x1_proj",
             "conv" + lvl + "_" + sub_lvl + "_1x1_proj_bn"]

    if modify_stride is False:
        prev = Conv2D(256 * level, (1, 1), strides=(1, 1), name=names[0],
                      use_bias=False)(prev)
    elif modify_stride is True:
        prev = Conv2D(256 * level, (1, 1), strides=(2, 2), name=names[0],
                      use_bias=False)(prev)

    prev = BN(name=names[1])(prev)
    return prev


def empty_branch(prev):
    return prev

#残差连接：卷积映射
def residual_short(prev_layer, level, pad=1, lvl=1, sub_lvl=1, modify_stride=False):
    prev_layer = Activation('relu')(prev_layer)
    block_1 = residual_conv(prev_layer, level,
                            pad=pad, lvl=lvl, sub_lvl=sub_lvl,
                            modify_stride=modify_stride)

    block_2 = short_convolution_branch(prev_layer, level,
                                       lvl=lvl, sub_lvl=sub_lvl,
                                       modify_stride=modify_stride)
    added = Add()([block_1, block_2])
    return added

#残差连接：恒等映射
def residual_empty(prev_layer, level, pad=1, lvl=1, sub_lvl=1):
    prev_layer = Activation('relu')(prev_layer)

    block_1 = residual_conv(prev_layer, level, pad=pad,
                            lvl=lvl, sub_lvl=sub_lvl)
    block_2 = empty_branch(prev_layer)
    added = Add()([block_1, block_2])
    return added


def ResNet(inp, layers):
    # Names for the first couple layers of model
    names = ["conv1_1_3x3_s2",
             "conv1_1_3x3_s2_bn",
             "conv1_2_3x3",
             "conv1_2_3x3_bn",
             "conv1_3_3x3",
             "conv1_3_3x3_bn"]
    # compute input shape
    if K.backend() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    begin_wtih_lager_kernel = False
    if begin_wtih_lager_kernel:
        cnv1 = Conv2D(
            64,
            (7, 7),
            strides=(2, 2),
            padding='same',
            name='conv1')(inp)
        bn1 = BatchNormalization(axis=bn_axis, name='bn_conv1')(cnv1)
        relu1 = Activation('relu')(bn1)
        res = MaxPooling2D((3, 3), strides=(2, 2))(relu1)
    else:
        # Short branch(only start of network)
        cnv1 = Conv2D(64, (3, 3), strides=(2, 2), padding='same', name=names[0],
                      use_bias=False)(inp)  # "conv1_1_3x3_s2"
        bn1 = BN(name=names[1])(cnv1)  # "conv1_1_3x3_s2/bn"
        relu1 = Activation('relu')(bn1)  # "conv1_1_3x3_s2/relu"

        cnv1 = Conv2D(64, (3, 3), strides=(1, 1), padding='same', name=names[2],
                      use_bias=False)(relu1)     # "conv1_2_3x3"
        bn1 = BN(name=names[3])(cnv1)            # "conv1_2_3x3/bn"
        relu1 = Activation('relu')(bn1)         # "conv1_2_3x3/relu"

        cnv1 = Conv2D(128, (3, 3), strides=(1, 1), padding='same', name=names[4],
                      use_bias=False)(relu1)          # "conv1_3_3x3"
        bn1 = BN(name=names[5])(cnv1)                 # "conv1_3_3x3/bn"
        relu1 = Activation('relu')(bn1)              # "conv1_3_3x3/relu"

        res = MaxPooling2D(pool_size=(3, 3), padding='same',
                           strides=(2, 2))(relu1)  # "pool1_3x3_s2"

    # ---Residual layers(body of network)
    """
    Modify_stride --Used only once in first 3_1 convolutions block.
    changes stride of first convolution from 1 -> 2
    """

    # 2_1- 2_3
    res = residual_short(res, 1, pad=1, lvl=2, sub_lvl=1)
    for i in range(2):
        res = residual_empty(res, 1, pad=1, lvl=2, sub_lvl=i + 2)

    # 3_1 - 3_3
    res = residual_short(res, 2, pad=1, lvl=3, sub_lvl=1, modify_stride=True)
    for i in range(3):
        res = residual_empty(res, 2, pad=1, lvl=3, sub_lvl=i + 2)
    if layers is 50:
        # 4_1 - 4_6
        res = residual_short(res, 4, pad=2, lvl=4, sub_lvl=1)
        for i in range(5):
            res = residual_empty(res, 4, pad=2, lvl=4, sub_lvl=i + 2)
    elif layers is 101:
        # 4_1 - 4_23
        res = residual_short(res, 4, pad=2, lvl=4, sub_lvl=1)
        for i in range(22):
            res = residual_empty(res, 4, pad=2, lvl=4, sub_lvl=i + 2)
    else:
        print("This ResNet is not implemented")

    # 5_1 - 5_3
    res = residual_short(res, 8, pad=4, lvl=5, sub_lvl=1)
    for i in range(2):
        res = residual_empty(res, 8, pad=4, lvl=5, sub_lvl=i + 2)

    res = Activation('relu')(res)
    return res


def interp_block(prev_layer, level, feature_map_shape, input_shape, output_stride=8.0):

    kernel_strides = (int(i/output_stride/level) for i in input_shape)
    names = [
        "conv5_3_pool" + str(level) + "_conv",
        "conv5_3_pool" + str(level) + "_conv_bn"
    ]
    prev_layer = AveragePooling2D(kernel_strides, strides=kernel_strides)(prev_layer)

    prev_layer = Conv2D(512, (1, 1), strides=(1, 1), name=names[0],
                        use_bias=False)(prev_layer)
    prev_layer = BN(name=names[1])(prev_layer)
    prev_layer = Activation('relu')(prev_layer)

    prev_layer = Interp(feature_map_shape)(prev_layer)  #上采样
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
    res = Concatenate()([res,
                         interp_block6,
                         interp_block3,
                         interp_block2,
                         interp_block1])
    return res


def build_pspnet(nb_classes, resnet_layers, input_shape, out_activation='softmax'):
    """Build PSPNet."""
    print("Building a PSPNet based on ResNet %i expecting inputs of shape %s predicting %i classes" % (
        resnet_layers, input_shape, nb_classes))

    inp = Input((input_shape[0], input_shape[1], 3))
    res = ResNet(inp, layers=resnet_layers)
    psp = build_pyramid_pooling_module(res, input_shape)

    x = Conv2D(512, (3, 3), strides=(1, 1), padding="same", name="conv5_4",
               use_bias=False)(psp)
    x = BN(name="conv5_4_bn")(x)
    x = Activation('relu')(x)
    x = Dropout(0.1)(x)

    #x = Conv2D(nb_classes, (1, 1), strides=(1, 1), name="conv6")(x)
    # x = Interp([input_shape[0], input_shape[1]])(x)
    # x = Activation('softmax')(x)

    x = Conv2D(56, (1, 1), padding='same', name="conv6")(x)
    x = Conv2DTranspose(1,(8,8),strides=(4,4),padding="same",use_bias=False,activation='sigmoid')(x)

    model = Model(inputs=inp, outputs=x)

    # Solver
    sgd = SGD(lr=learning_rate, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

if __name__ == '__main__':
    model = build_pspnet(1, 101, (224,224))   #(473,473)（713,713）（640,480）
    #model.load_weights("weights/keras/pspnet101_voc2012.h5")
    model.summary()
    print('load successfully')