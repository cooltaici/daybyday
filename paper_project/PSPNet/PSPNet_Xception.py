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
    return base_model
if __name__ == "__main__":
    model = PSPNet_Xception()
    model.summary()
    print("over")