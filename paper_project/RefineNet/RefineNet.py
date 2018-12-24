"""
Based on https://github.com/GeorgeSeif/Semantic-Segmentation-Suite
"""

from keras.models import Model
from RefineNet.resnet101 import resnet101_model
from keras.layers import Conv2D, Conv2DTranspose, BatchNormalization, Add, MaxPooling2D,Activation

from keras.layers import Layer, InputSpec
import keras.backend as K
import tensorflow as tf
from RefineNet.utils.utils import *

def build_refinenet(input_shape, num_class, resnet_weights, upscaling_method="bilinear"):
    """
    Builds the RefineNet model.
    Arguments:
      input_shape: Size of input image, including number of channels
      num_classes: Number of classes
      resnet_weights: Path to pre-trained weights for ResNet-101
      freeze_frontend: Whether or not to freeze ResNet layers during training
      upscaling_method: Either 'bilinear' or 'conv'
    Returns:
      RefineNet model
    """

    # Build ResNet-101
    model_base = resnet101_model(input_shape, resnet_weights)
    model_base.trainable = False

    # Get ResNet block output layers
    high = [model_base.get_layer('res5c_relu').output,
            model_base.get_layer('res4b22_relu').output,
            model_base.get_layer('res3b3_relu').output,
            model_base.get_layer('res2c_relu').output]

    low = [None, None, None]

    # Get the feature maps to the proper size with bottleneck
    high[0] = Conv2D(512, (1,1), padding='same')(high[0])
    high[1] = Conv2D(256, (1,1), padding='same')(high[1])
    high[2] = Conv2D(256, (1,1), padding='same')(high[2])
    high[3] = Conv2D(256, (1,1), padding='same')(high[3])

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

    net = Conv2D(num_class, 1, activation='sigmoid')(net)  #可以使用softmax

    model = Model(model_base.input, net)

    return model


if __name__ == '__main__':
    model = build_refinenet((224, 224, 3), 1, None, "conv")
    model.summary()
    print('load successfully')