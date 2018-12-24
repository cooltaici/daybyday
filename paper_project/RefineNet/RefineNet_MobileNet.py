from __future__ import print_function
from keras.applications.mobilenet import *
from math import ceil
from keras.models import Model
from keras.layers import *
from keras import layers
from keras_layer_L2Normalization import *
from keras.backend import tf as ktf
import os
from RefineNet.utils.utils import *

def RefineNet_MobileNet(input_shape = (224,224,3), upscaling_method="bilinear"):
    print("over")