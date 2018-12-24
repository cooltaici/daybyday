from keras.applications.mobilenet import *
from keras.models import *
from keras.layers import *
from keras import layers
from keras_layer_L2Normalization import *
from keras.backend import tf as ktf
import os
from PSPNet.utils.utils import *
urlpath = r"https://github.com/fchollet/deep-learning-models/releases/download/v0.6/mobilenet_1_0_224_tf_no_top.h5"

def PSPNet_MobileNet(input_shape = (224,224,3)):
    base_model = MobileNet(input_shape=input_shape,weights="imagenet",include_top=False)
    return base_model
if __name__ == "__main__":
    model = PSPNet_MobileNet()
    model.summary()
    print("over")