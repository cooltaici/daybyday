from keras.applications.densenet import *
from keras.models import *
from keras.layers import *
from keras import layers
from keras_layer_L2Normalization import *
from keras.backend import tf as ktf
import os

from PSPNet.utils.utils import *
r"https://github.com/fchollet/deep-learning-models/releases/download/v0.8/densenet121_weights_tf_dim_ordering_tf_kernels_notop.h5"
def PSPNet_DenseNet(input_shape = (224,224,3)):
    base_model = DenseNet121(input_shape=input_shape,weights="imagenet",include_top=False)
    return base_model
if __name__ == "__main__":
    model = PSPNet_DenseNet()
    model.summary()
    print("over")