from keras.models import Model
from RefineNet.resnet101 import resnet101_model
from keras.layers import Conv2D, Conv2DTranspose, BatchNormalization, Add, MaxPooling2D,Activation

from keras.layers import Layer, InputSpec
import keras.backend as K
import tensorflow as tf


class Upsampling(Layer):
    def __init__(self, scale=1, **kwargs):
        self.scale = scale
        self.input_spec = [InputSpec(ndim=4)]
        super(Upsampling, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        width = int(self.scale * input_shape[1] if input_shape[1] is not None else None)
        height = int(self.scale * input_shape[2] if input_shape[2] is not None else None)
        return (input_shape[0], width, height, input_shape[3])

    def call(self, X, mask=None):
        original_shape = K.int_shape(X)
        new_shape = tf.shape(X)[1:3]
        new_shape *= tf.constant(self.scale)
        X = tf.image.resize_bilinear(X, new_shape)
        X.set_shape((None, original_shape[1] * self.scale, original_shape[2] * self.scale, None))
        return X

    def get_config(self):
        config = {'scale': self.scale}
        base_config = super(Upsampling, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def ConvBlock(inputs, n_filters, kernel_size=(3, 3), padding="same"):
    """
    Basic conv block for Encoder-Decoder
    Apply successivly Convolution, BatchNormalization, ReLU nonlinearity
    """

    net = Conv2D(n_filters, kernel_size, padding=padding)(inputs)
    net = BatchNormalization()(net)
    net = Activation(activation="relu")(net)
    return net


def ConvUpscaleBlock(inputs, n_filters, kernel_size=(3, 3), scale=2,padding="same"):
    """
    Basic conv transpose block for Encoder-Decoder upsampling
    Apply successivly Transposed Convolution, BatchNormalization, ReLU nonlinearity
    """
    net = Conv2DTranspose(n_filters, kernel_size, strides=(scale, scale),padding=padding)(inputs)
    net = BatchNormalization()(net)
    net = Activation(activation="relu")(net)
    return net


def ResidualConvUnit(inputs, n_filters=256, kernel_size=3):
    """
    A local residual unit designed to fine-tune the pretrained ResNet weights
    Arguments:
      inputs: The input tensor
      n_filters: Number of output feature maps for each conv
      kernel_size: Size of convolution kernel
    Returns:
      Output of local residual block
    """

    net = Activation(activation="relu")(inputs)
    net = Conv2D(n_filters, kernel_size, padding='same')(net)
    net = Activation(activation="relu")(net)
    net = Conv2D(n_filters, kernel_size, padding='same')(net)
    net = Add()([net, inputs])

    return net


def ChainedResidualPooling(inputs, n_filters=256):
    """
    Chained residual pooling aims to capture background
    context from a large image region. This component is
    built as a chain of 2 pooling blocks, each consisting
    of one max-pooling layer and one convolution layer. One pooling
    block takes the output of the previous pooling block as
    input. The output feature maps of all pooling blocks are
    fused together with the input feature map through summation
    of residual connections.
    Arguments:
      inputs: The input tensor
      n_filters: Number of output feature maps for each conv
    Returns:
      Double-pooled feature maps
    """

    net_relu = Activation(activation="relu")(inputs)
    net = MaxPooling2D(pool_size=(5, 5), strides=1, padding='same')(net_relu)
    net = Conv2D(n_filters, 3, padding='same')(net)
    net_sum_1 = Add()([net, net_relu])

    net = MaxPooling2D(pool_size=(5, 5), strides=1, padding='same')(net)
    net = Conv2D(n_filters, 3, padding='same')(net)
    net_sum_2 = Add()([net, net_sum_1])

    return net_sum_2


def MultiResolutionFusion(high_inputs=None, low_inputs=None, n_filters=256):
    """
    Fuse together all path inputs. This block first applies convolutions
    for input adaptation, which generate feature maps of the same feature dimension
    (the smallest one among the inputs), and then up-samples all (smaller) feature maps to
    the largest resolution of the inputs. Finally, all features maps are fused by summation.
    Arguments:
      high_inputs: The input tensors that have the higher resolution
      low_inputs: The input tensors that have the lower resolution
      n_filters: Number of output feature maps for each conv
    Returns:
      Fused feature maps at higher resolution

    """
    if high_inputs is None:  # RefineNet block 4
        fuse = Conv2D(n_filters, 3, padding='same')(low_inputs)
        return fuse

    else:
        conv_low = Conv2D(n_filters, 3, padding='same')(low_inputs)
        conv_high = Conv2D(n_filters, 3, padding='same')(high_inputs)
        conv_low_up = Upsampling(scale=2)(conv_low)

        return Add()([conv_low_up, conv_high])


def RefineBlock(high_inputs=None, low_inputs=None):
    """
    A RefineNet Block which combines together the ResidualConvUnits,
    fuses the feature maps using MultiResolutionFusion, and then gets
    large-scale context with the ResidualConvUnit.
    Arguments:
      high_inputs: The input tensors that have the higher resolution
      low_inputs: The input tensors that have the lower resolution
    Returns:
      RefineNet block for a single path i.e one resolution

    """

    if low_inputs is None:  # block 4
        rcu_new_low = ResidualConvUnit(high_inputs, n_filters=512)
        rcu_new_low = ResidualConvUnit(rcu_new_low, n_filters=512)

        fuse = MultiResolutionFusion(high_inputs=None, low_inputs=rcu_new_low, n_filters=512)
        fuse_pooling = ChainedResidualPooling(fuse, n_filters=512)
        output = ResidualConvUnit(fuse_pooling, n_filters=512)
        return output
    else:
        rcu_high = ResidualConvUnit(high_inputs, n_filters=256)
        rcu_high = ResidualConvUnit(rcu_high, n_filters=256)

        fuse = MultiResolutionFusion(rcu_high, low_inputs, n_filters=256)
        fuse_pooling = ChainedResidualPooling(fuse, n_filters=256)
        output = ResidualConvUnit(fuse_pooling, n_filters=256)
        return output