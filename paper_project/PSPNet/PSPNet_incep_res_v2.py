from keras.applications.inception_resnet_v2 import *
from keras.models import *
from keras.layers import *
from keras import layers
from keras_layer_L2Normalization import *
from keras.backend import tf as ktf
import os
from PSPNet.utils.utils import *
r"https://github.com/fchollet/deep-learning-models/releases/download/v0.7/inception_resnet_v2_weights_tf_dim_ordering_tf_kernels_notop.h5"
def PSPNet_Incep_Res_v2(input_shape = (224,224,3)):
    base_model = InceptionResNetV2(input_shape=input_shape,weights="imagenet",include_top=False)
    return base_model
if __name__ == "__main__":
    model = PSPNet_Incep_Res_v2()
    model.summary()
    print("over")