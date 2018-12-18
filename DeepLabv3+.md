# Rethinking Atrous Convolution for Semantic Image Segmentation
**paper**：https://arxiv.org/pdf/1802.02611v1.pdf </br>
**Keras 源代码** :https://github.com/mjDelta/deeplabv3plus-keras </br>
**Keras 源代码1** https://github.com/Shmuelnaaman/deeplab_v3 </br>
**DeeplabV1-V3+**：https://blog.csdn.net/Dlyldxwl/article/details/81148810 </br>
**DeepLabV3+比较好的翻译**：https://blog.csdn.net/zziahgf/article/details/79557105 </br>
**实验代码**：https://github.com/cooltaici/daybyday/blob/master/paper_project/Deeplav_v3plus.py </br>
**2018语义分割类导读**：https://blog.csdn.net/SHAOYEZUIZUISHAUI/article/details/82666764  </br>
### 本文的主要贡献：
第一：对DeepLabv3添加了简单有效的解码模块，这可以大大提高对边界的分割，并且可以通过控制atrous convolution 来控制编码特征的分辨率，来平衡精度和运行时间（已有编码-解码结构不具有该能力）</br>
第二：主打的Xception模块，深度可分卷积结构(depthwise separable convolution) 用到带孔空间金字塔池化(Atrous Spatial Pyramid Pooling, ASPP)模块和解码模块中，得到更快速有效的 编码-解码网络。</br>
&emsp;如Figure 1所示，这是常见的用于语义分割的策略。（1）金字塔结构。（2）编码-解码机构。（3）本文提出的结合两种策略的方案。</br>
![Figure 1](https://github.com/cooltaici/daybyday/blob/master/picture_paper/DeepLabV3plus/DeepLabV3plus_Figure1.PNG)</br>
### 2. 近来相关工作
&emsp;基于FCN[64,49]的模型已经证明在多个分割数据库[17,52,13,83,5]取得了不错的效果。近年来提出的几种变体来利用语义信息的模型[18,65,36,39,22,79,51,14]，还有一些采用多尺度输入（图像金字塔）[18,16,58,44,11,9]，另外还有一些采用概率图模型的（比如SenseCRF[37]）[8,4,82,44,55,63,34,72,6,7,9]。在本文中，主要采用空间金字塔和编码-解码模型</br>
&emsp;**空间金字塔**。像PSPNet采用并行的不同尺度的空间池化层，以及DeepLab采用的空间孔洞金字塔池化（ASPP）。这样的方案证明取得了不错的效果。备注：想RefineNet其实采用的池化又有点不一样。</br>
&emsp;**编码-解码**：编码-解码网络已经被成功应用到很多计算机视觉任务当中，包括 人体姿态分析[53]，目标检测[45,66,19]，语义分割[49,54,61,3,43,59,57,33,76,20]。一般来说，编码-解码模型有两个模块，(1) 编码模块，逐渐减少特征大小以获取更高级的语义特征。(2)解码模块。逐渐恢复特征大小和细节信息。在本文中，我们使用DeepLabv3[10]作为编码模块，并且增加一个见得有效的解码模块一获得更准确的边缘信息。</br>
&emsp;**深度可分离卷积（Depthwise separable convolution）**：深度可分离卷积[67,71]或者组合卷积[38,78]，能有效减少计算参数的同时，还能保持效果的方法。这种方法最近已经被应用到很多深度卷积网络当中[35,74,12,31,80,60,84]。特别的在本文中，我们研究Xception模型[12]，和在COCO 2017目标检测竞赛提出的版本类似[60]，我们证明了速度和精度都有很大的提升。</br>
### 3. 方法
&emsp;在这一章节，我们简要介绍孔洞卷积，以及深度可分离卷积[67,71,74,12,310.然后我们惠顾DeepLabv3[10]，我们还提出了一个修改后的Xception模型[12,60]，他可以在提高速度的同时得到更好的表现。</br>
#### 3.1 使用孔洞卷积的编码-解码
&emsp;**孔洞卷积**：在之前版本就详细说明过，参考[9]获得更多信息</br>

&emsp;**深度可分离卷积**：深度可分卷积操作，将标准卷积分解为一个 depthwise conv，depthwise conv 后接 pointwise conv(如，1×1conv)，有效的降低计算复杂度.

depthwise conv 对每一个输入通道(channel) 分别进行 spatial conv；

pointwise conv 用于合并 depthwise conv 的输出
。TensorFlow 实现的 depthwise separable conv 已经在 depthwise conv 中支持孔洞卷积，我们发现孔洞卷积效果更好。</br>

&emsp;**DeepLabv3作为编码**：在Deeplabv3中，采用的output_stride=8或者16（为8时，最后两个block的孔洞卷积rate分别为2和4，并且去掉所有stride）。此外DeepLabv3中使用了纵式的多尺度ASPP，和全局池化的图像级信息[47]。在本文中把Deeplabv3中最后一层（在logits输出之前）作为编码模块的输出，它包含了丰富的语义信息。此外可以使用孔洞卷积的参数任意调整编码输出的分辨率，这取决与时间和性能的平衡。</br>

&emsp;**解码结构**：Deeplabv3中将最后的输出上采样16倍，这相当于一个简单的解码器，这种解码器并不能很好恢复图像的细节信息。本文提出了一个简单有效的解码模块，如Figure 2所示。解码的输出经过4倍上采样之后（双线性插值），然后与解码模块中间特征（相同分辨率）进行级联。因为编码输出channel=256，考虑到channel匹配的问题，在级联low-level特征之前，使用1*1的卷积调整channel。在级联操作之后，再采用一些3*3的卷积操作进行优化，最后再进行4倍上采样恢复到原图。在第四章节，我们发现output_stride=16的解码模块在速度以及精度上是最平衡的。当output_stride=8时，精度达到最优。</br>
![Figure 2](https://github.com/cooltaici/daybyday/blob/master/picture_paper/DeepLabV3plus/DeepLabV3plus_Figure2.PNG)</br>

#### 3.2 对Aligned Xception的修改
&emsp;Xception 模型[12]，在ImageNet上已经取得了非常不错的成绩。最近MSRA团队[62]修改了Xception结构并且把它命名为Aligned Xception，这种结构进一步推动了目标检测任务的Xception在目标检测任务中的发展。本文有这些工作中取得了灵感，我们将Xception应用到语义分割领域，特别的是我们在MSRA's的基础上，又增加了一些修改：
- 相同深度 Xception，除了没有修改 entry flow network 结构，以快速计算和内存效率.
- 采用 depthwise separable conv 来替换所有的 max-pooling 操作，以利用 atrous separable conv 来提取任意分辨率的 feature maps.
- 在每个 3×33×3 depthwise conv 后，添加 BN 和 ReLU，类似于 MobileNet.
![Figure 3](https://github.com/cooltaici/daybyday/blob/master/picture_paper/DeepLabV3plus/DeepLabV3plus_Figure3.PNG)</br>

### 4. 实验结果
#### 4.1 解码器设计
&emsp;在Deeplabv3[10]中，当我们使用output_stride=16时，使用线性插值上采样比不使用（对金标准下采样）结果要好1.2%。因为上采样也是简单的解码过程。解码模块的设计如第三章节所示（Figure 2）。这里我们用实验验证一下这样设计的理由。</br>
&emsp;**channel的影响**：改变low-level特征中1*1卷积的channle数目，效果变化如Tabel 1所示，本文采取[1*1,48]用于channel匹配。</br>
![Table 1](https://github.com/cooltaici/daybyday/blob/master/picture_paper/DeepLabV3plus/DeepLabV3plus_Table1.PNG)</br>
&emsp;**级联后的操作**：我们发现在级联之后，采用3*3的卷积操作，结果最好，如Table 2所示。我们采用了不同的channel数目以及不同的kernel size，发现两个[3*3,256]的效果最理想。</br>
![Table 2](https://github.com/cooltaici/daybyday/blob/master/picture_paper/DeepLabV3plus/DeepLabV3plus_Table2.PNG)</br>

#### 4.2 ResNet-101作为骨架网络
&emsp; 我们对ResNet-101作为的解码网络进行了速度与性能的详细测试，如Table 3所示。主要控制的变量有（1）**Baseline**。训练和评估采用不同outout_stride的结果，训练阶段用16，评估阶段用8结果更好。（2）**增加解码器**。增加了解码结构后，网络的表现有1%左右的提升，这里看来，其实解码结果不是本文主打的方向，提升有限。（3）**更加粗糙的编码特征**。如第三列所示，我们使用output_stride=32时，效果并不是很好。</br>
![Table 3](https://github.com/cooltaici/daybyday/blob/master/picture_paper/DeepLabV3plus/DeepLabV3plus_Table3.PNG)</br>

#### 4.2 Xception作为骨架网络
&emsp; 我们进一步探究了Xeption作为解码网络的表现。相比[60]，我们做了一些修改，如第三章节描述的那样。</br>
&emsp; “IamgeNet预训练**：我们使用ImageNet数据集对本文中的Xception进行预训练。动量因子0.9，初始学习率0.05，blabla...。如Table 4所示，我们修改后的Xception在误差上要比ResNet-101要小一些。</br>
![Table 4](https://github.com/cooltaici/daybyday/blob/master/picture_paper/DeepLabV3plus/DeepLabV3plus_Table4.PNG)</br>

**使用Xception作为编码结果的总体结果如Table 5所示**：可以看出应用所有技术，最高准确率能达到84.56%。随后我们使用output_stride=8进行训练，并且使用batch normalization参数的方法，最后达到了87.8%的准确率，使用JFT数据集后，达到了89.0%。其它流行的方法比较如 Table 6所示。
![Table 5](https://github.com/cooltaici/daybyday/blob/master/picture_paper/DeepLabV3plus/DeepLabV3plus_Table5.PNG)</br>
![Table 6](https://github.com/cooltaici/daybyday/blob/master/picture_paper/DeepLabV3plus/DeepLabV3plus_Table6.PNG)</br>

``` python
#code
# -*- coding: utf-8 -*-

""" Deeplabv3+ model for Keras.
This model is based on TF repo:
https://github.com/tensorflow/models/tree/master/research/deeplab
On Pascal VOC, original model gets to 84.56% mIOU

Now this model is only available for the TensorFlow backend,
due to its reliance on `SeparableConvolution` layers, but Theano will add
this layer soon.

MobileNetv2 backbone is based on this repo:
https://github.com/JonathanCMitchell/mobilenet_v2_keras

# Reference
- [Encoder-Decoder with Atrous Separable Convolution
    for Semantic Image Segmentation](https://arxiv.org/pdf/1802.02611.pdf)
- [Xception: Deep Learning with Depthwise Separable Convolutions]
    (https://arxiv.org/abs/1610.02357)
- [Inverted Residuals and Linear Bottlenecks: Mobile Networks for
    Classification, Detection and Segmentation](https://arxiv.org/abs/1801.04381)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from keras.models import Model
from keras import layers
from keras.layers import Input
from keras.layers import Activation
from keras.layers import Concatenate
from keras.layers import Add
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.layers import Conv2D
from keras.layers import DepthwiseConv2D
from keras.layers import ZeroPadding2D
from keras.layers import AveragePooling2D
from keras.engine import Layer
from keras.engine import InputSpec
from keras.engine.topology import get_source_inputs
from keras import backend as K
from keras.applications import imagenet_utils
from keras.utils import conv_utils
from keras.utils.data_utils import get_file

WEIGHTS_PATH_X = "https://github.com/bonlime/keras-deeplab-v3-plus/releases/download/1.1/deeplabv3_xception_tf_dim_ordering_tf_kernels.h5"
WEIGHTS_PATH_MOBILE = "https://github.com/bonlime/keras-deeplab-v3-plus/releases/download/1.1/deeplabv3_mobilenetv2_tf_dim_ordering_tf_kernels.h5"


class BilinearUpsampling(Layer):
    """Just a simple bilinear upsampling layer. Works only with TF.
       Args:
           upsampling: tuple of 2 numbers > 0. The upsampling ratio for h and w
           output_size: used instead of upsampling arg if passed!
    """

    def __init__(self, upsampling=(2, 2), output_size=None, data_format=None, **kwargs):

        super(BilinearUpsampling, self).__init__(**kwargs)

        self.data_format = conv_utils.normalize_data_format(data_format)
        self.input_spec = InputSpec(ndim=4)
        if output_size:
            self.output_size = conv_utils.normalize_tuple(
                output_size, 2, 'output_size')
            self.upsampling = None
        else:
            self.output_size = None
            self.upsampling = conv_utils.normalize_tuple(
                upsampling, 2, 'upsampling')

    def compute_output_shape(self, input_shape):
        if self.upsampling:
            height = self.upsampling[0] * \
                input_shape[1] if input_shape[1] is not None else None
            width = self.upsampling[1] * \
                input_shape[2] if input_shape[2] is not None else None
        else:
            height = self.output_size[0]
            width = self.output_size[1]
        return (input_shape[0],
                height,
                width,
                input_shape[3])

    def call(self, inputs):
        if self.upsampling:
            return K.tf.image.resize_bilinear(inputs, (inputs.shape[1] * self.upsampling[0],
                                                       inputs.shape[2] * self.upsampling[1]),
                                              align_corners=True)
        else:
            return K.tf.image.resize_bilinear(inputs, (self.output_size[0],
                                                       self.output_size[1]),
                                              align_corners=True)

    def get_config(self):
        config = {'upsampling': self.upsampling,
                  'output_size': self.output_size,
                  'data_format': self.data_format}
        base_config = super(BilinearUpsampling, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def SepConv_BN(x, filters, prefix, stride=1, kernel_size=3, rate=1, depth_activation=False, epsilon=1e-3):
    """ SepConv with BN between depthwise & pointwise. Optionally add activation after BN
        Implements right "same" padding for even kernel sizes
        Args:
            x: input tensor
            filters: num of filters in pointwise convolution
            prefix: prefix before name
            stride: stride at depthwise conv
            kernel_size: kernel size for depthwise convolution
            rate: atrous rate for depthwise convolution
            depth_activation: flag to use activation between depthwise & poinwise convs
            epsilon: epsilon to use in BN layer
    """

    if stride == 1:
        depth_padding = 'same'
    else:
        kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
        pad_total = kernel_size_effective - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        x = ZeroPadding2D((pad_beg, pad_end))(x)
        depth_padding = 'valid'

    if not depth_activation:
        x = Activation('relu')(x)
    x = DepthwiseConv2D((kernel_size, kernel_size), strides=(stride, stride), dilation_rate=(rate, rate),
                        padding=depth_padding, use_bias=False, name=prefix + '_depthwise')(x)
    x = BatchNormalization(name=prefix + '_depthwise_BN', epsilon=epsilon)(x)
    if depth_activation:
        x = Activation('relu')(x)
    x = Conv2D(filters, (1, 1), padding='same',
               use_bias=False, name=prefix + '_pointwise')(x)
    x = BatchNormalization(name=prefix + '_pointwise_BN', epsilon=epsilon)(x)
    if depth_activation:
        x = Activation('relu')(x)

    return x


def _conv2d_same(x, filters, prefix, stride=1, kernel_size=3, rate=1):
    """Implements right 'same' padding for even kernel sizes
        Without this there is a 1 pixel drift when stride = 2
        Args:
            x: input tensor
            filters: num of filters in pointwise convolution
            prefix: prefix before name
            stride: stride at depthwise conv
            kernel_size: kernel size for depthwise convolution
            rate: atrous rate for depthwise convolution
    """
    if stride == 1:
        return Conv2D(filters,
                      (kernel_size, kernel_size),
                      strides=(stride, stride),
                      padding='same', use_bias=False,
                      dilation_rate=(rate, rate),
                      name=prefix)(x)
    else:
        kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
        pad_total = kernel_size_effective - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        x = ZeroPadding2D((pad_beg, pad_end))(x)
        return Conv2D(filters,
                      (kernel_size, kernel_size),
                      strides=(stride, stride),
                      padding='valid', use_bias=False,
                      dilation_rate=(rate, rate),
                      name=prefix)(x)


def _xception_block(inputs, depth_list, prefix, skip_connection_type, stride,
                    rate=1, depth_activation=False, return_skip=False):
    """ Basic building block of modified Xception network
        Args:
            inputs: input tensor
            depth_list: number of filters in each SepConv layer. len(depth_list) == 3
            prefix: prefix before name
            skip_connection_type: one of {'conv','sum','none'}
            stride: stride at last depthwise conv
            rate: atrous rate for depthwise convolution
            depth_activation: flag to use activation between depthwise & pointwise convs
            return_skip: flag to return additional tensor after 2 SepConvs for decoder
            """
    residual = inputs
    for i in range(3):
        residual = SepConv_BN(residual,
                              depth_list[i],
                              prefix + '_separable_conv{}'.format(i + 1),
                              stride=stride if i == 2 else 1,
                              rate=rate,
                              depth_activation=depth_activation)
        if i == 1:
            skip = residual
    if skip_connection_type == 'conv':
        shortcut = _conv2d_same(inputs, depth_list[-1], prefix + '_shortcut',
                                kernel_size=1,
                                stride=stride)
        shortcut = BatchNormalization(name=prefix + '_shortcut_BN')(shortcut)
        outputs = layers.add([residual, shortcut])
    elif skip_connection_type == 'sum':
        outputs = layers.add([residual, inputs])
    elif skip_connection_type == 'none':
        outputs = residual
    if return_skip:
        return outputs, skip
    else:
        return outputs


def relu6(x):
    return K.relu(x, max_value=6)


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def _inverted_res_block(inputs, expansion, stride, alpha, filters, block_id, skip_connection, rate=1):
    in_channels = inputs._keras_shape[-1]
    pointwise_conv_filters = int(filters * alpha)
    pointwise_filters = _make_divisible(pointwise_conv_filters, 8)
    x = inputs
    prefix = 'expanded_conv_{}_'.format(block_id)
    if block_id:
        # Expand

        x = Conv2D(expansion * in_channels, kernel_size=1, padding='same',
                   use_bias=False, activation=None,
                   name=prefix + 'expand')(x)
        x = BatchNormalization(epsilon=1e-3, momentum=0.999,
                               name=prefix + 'expand_BN')(x)
        x = Activation(relu6, name=prefix + 'expand_relu')(x)
    else:
        prefix = 'expanded_conv_'
    # Depthwise
    x = DepthwiseConv2D(kernel_size=3, strides=stride, activation=None,
                        use_bias=False, padding='same', dilation_rate=(rate, rate),
                        name=prefix + 'depthwise')(x)
    x = BatchNormalization(epsilon=1e-3, momentum=0.999,
                           name=prefix + 'depthwise_BN')(x)

    x = Activation(relu6, name=prefix + 'depthwise_relu')(x)

    # Project
    x = Conv2D(pointwise_filters,
               kernel_size=1, padding='same', use_bias=False, activation=None,
               name=prefix + 'project')(x)
    x = BatchNormalization(epsilon=1e-3, momentum=0.999,
                           name=prefix + 'project_BN')(x)

    if skip_connection:
        return Add(name=prefix + 'add')([inputs, x])

    # if in_channels == pointwise_filters and stride == 1:
    #    return Add(name='res_connect_' + str(block_id))([inputs, x])

    return x


def Deeplabv3(weights='pascal_voc', input_tensor=None, input_shape=(512, 512, 3), classes=21, backbone='mobilenetv2', OS=16, alpha=1.):
    """ Instantiates the Deeplabv3+ architecture

    Optionally loads weights pre-trained
    on PASCAL VOC. This model is available for TensorFlow only,
    and can only be used with inputs following the TensorFlow
    data format `(width, height, channels)`.
    # Arguments
        weights: one of 'pascal_voc' (pre-trained on pascal voc)
            or None (random initialization)
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: shape of input image. format HxWxC
            PASCAL VOC model was trained on (512,512,3) images
        classes: number of desired classes. If classes != 21,
            last layer is initialized randomly
        backbone: backbone to use. one of {'xception','mobilenetv2'}
        OS: determines input_shape/feature_extractor_output ratio. One of {8,16}.
            Used only for xception backbone.
        alpha: controls the width of the MobileNetV2 network. This is known as the
            width multiplier in the MobileNetV2 paper.
                - If `alpha` < 1.0, proportionally decreases the number
                    of filters in each layer.
                - If `alpha` > 1.0, proportionally increases the number
                    of filters in each layer.
                - If `alpha` = 1, default number of filters from the paper
                    are used at each layer.
            Used only for mobilenetv2 backbone

    # Returns
        A Keras model instance.

    # Raises
        RuntimeError: If attempting to run this model with a
            backend that does not support separable convolutions.
        ValueError: in case of invalid argument for `weights` or `backbone`

    """

    if not (weights in {'pascal_voc', None}):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `pascal_voc` '
                         '(pre-trained on PASCAL VOC)')

    if K.backend() != 'tensorflow':
        raise RuntimeError('The Deeplabv3+ model is only available with '
                           'the TensorFlow backend.')

    if not (backbone in {'xception', 'mobilenetv2'}):
        raise ValueError('The `backbone` argument should be either '
                         '`xception`  or `mobilenetv2` ')

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    if backbone == 'xception':
        if OS == 8:
            entry_block3_stride = 1
            middle_block_rate = 2  # ! Not mentioned in paper, but required
            exit_block_rates = (2, 4)
            atrous_rates = (12, 24, 36)
        else:
            entry_block3_stride = 2
            middle_block_rate = 1
            exit_block_rates = (1, 2)
            atrous_rates = (6, 12, 18)

        x = Conv2D(32, (3, 3), strides=(2, 2),
                   name='entry_flow_conv1_1', use_bias=False, padding='same')(img_input)
        x = BatchNormalization(name='entry_flow_conv1_1_BN')(x)
        x = Activation('relu')(x)

        x = _conv2d_same(x, 64, 'entry_flow_conv1_2', kernel_size=3, stride=1)
        x = BatchNormalization(name='entry_flow_conv1_2_BN')(x)
        x = Activation('relu')(x)

        x = _xception_block(x, [128, 128, 128], 'entry_flow_block1',
                            skip_connection_type='conv', stride=2,
                            depth_activation=False)
        x, skip1 = _xception_block(x, [256, 256, 256], 'entry_flow_block2',
                                   skip_connection_type='conv', stride=2,
                                   depth_activation=False, return_skip=True)

        x = _xception_block(x, [728, 728, 728], 'entry_flow_block3',
                            skip_connection_type='conv', stride=entry_block3_stride,
                            depth_activation=False)
        for i in range(16):
            x = _xception_block(x, [728, 728, 728], 'middle_flow_unit_{}'.format(i + 1),
                                skip_connection_type='sum', stride=1, rate=middle_block_rate,
                                depth_activation=False)

        x = _xception_block(x, [728, 1024, 1024], 'exit_flow_block1',
                            skip_connection_type='conv', stride=1, rate=exit_block_rates[0],
                            depth_activation=False)
        x = _xception_block(x, [1536, 1536, 2048], 'exit_flow_block2',
                            skip_connection_type='none', stride=1, rate=exit_block_rates[1],
                            depth_activation=True)

    else:
        OS = 8
        first_block_filters = _make_divisible(32 * alpha, 8)
        x = Conv2D(first_block_filters,
                   kernel_size=3,
                   strides=(2, 2), padding='same',
                   use_bias=False, name='Conv')(img_input)
        x = BatchNormalization(
            epsilon=1e-3, momentum=0.999, name='Conv_BN')(x)
        x = Activation(relu6, name='Conv_Relu6')(x)

        x = _inverted_res_block(x, filters=16, alpha=alpha, stride=1,
                                expansion=1, block_id=0, skip_connection=False)

        x = _inverted_res_block(x, filters=24, alpha=alpha, stride=2,
                                expansion=6, block_id=1, skip_connection=False)
        x = _inverted_res_block(x, filters=24, alpha=alpha, stride=1,
                                expansion=6, block_id=2, skip_connection=True)

        x = _inverted_res_block(x, filters=32, alpha=alpha, stride=2,
                                expansion=6, block_id=3, skip_connection=False)
        x = _inverted_res_block(x, filters=32, alpha=alpha, stride=1,
                                expansion=6, block_id=4, skip_connection=True)
        x = _inverted_res_block(x, filters=32, alpha=alpha, stride=1,
                                expansion=6, block_id=5, skip_connection=True)

        # stride in block 6 changed from 2 -> 1, so we need to use rate = 2
        x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1,  # 1!
                                expansion=6, block_id=6, skip_connection=False)
        x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1, rate=2,
                                expansion=6, block_id=7, skip_connection=True)
        x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1, rate=2,
                                expansion=6, block_id=8, skip_connection=True)
        x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1, rate=2,
                                expansion=6, block_id=9, skip_connection=True)

        x = _inverted_res_block(x, filters=96, alpha=alpha, stride=1, rate=2,
                                expansion=6, block_id=10, skip_connection=False)
        x = _inverted_res_block(x, filters=96, alpha=alpha, stride=1, rate=2,
                                expansion=6, block_id=11, skip_connection=True)
        x = _inverted_res_block(x, filters=96, alpha=alpha, stride=1, rate=2,
                                expansion=6, block_id=12, skip_connection=True)

        x = _inverted_res_block(x, filters=160, alpha=alpha, stride=1, rate=2,  # 1!
                                expansion=6, block_id=13, skip_connection=False)
        x = _inverted_res_block(x, filters=160, alpha=alpha, stride=1, rate=4,
                                expansion=6, block_id=14, skip_connection=True)
        x = _inverted_res_block(x, filters=160, alpha=alpha, stride=1, rate=4,
                                expansion=6, block_id=15, skip_connection=True)

        x = _inverted_res_block(x, filters=320, alpha=alpha, stride=1, rate=4,
                                expansion=6, block_id=16, skip_connection=False)

    # end of feature extractor

    # branching for Atrous Spatial Pyramid Pooling

    # Image Feature branch
    #out_shape = int(np.ceil(input_shape[0] / OS))
    b4 = AveragePooling2D(pool_size=(int(np.ceil(input_shape[0] / OS)), int(np.ceil(input_shape[1] / OS))))(x)
    b4 = Conv2D(256, (1, 1), padding='same',
                use_bias=False, name='image_pooling')(b4)
    b4 = BatchNormalization(name='image_pooling_BN', epsilon=1e-5)(b4)
    b4 = Activation('relu')(b4)
    b4 = BilinearUpsampling((int(np.ceil(input_shape[0] / OS)), int(np.ceil(input_shape[1] / OS))))(b4)

    # simple 1x1
    b0 = Conv2D(256, (1, 1), padding='same', use_bias=False, name='aspp0')(x)
    b0 = BatchNormalization(name='aspp0_BN', epsilon=1e-5)(b0)
    b0 = Activation('relu', name='aspp0_activation')(b0)

    # there are only 2 branches in mobilenetV2. not sure why
    if backbone == 'xception':
        # rate = 6 (12)
        b1 = SepConv_BN(x, 256, 'aspp1',
                        rate=atrous_rates[0], depth_activation=True, epsilon=1e-5)
        # rate = 12 (24)
        b2 = SepConv_BN(x, 256, 'aspp2',
                        rate=atrous_rates[1], depth_activation=True, epsilon=1e-5)
        # rate = 18 (36)
        b3 = SepConv_BN(x, 256, 'aspp3',
                        rate=atrous_rates[2], depth_activation=True, epsilon=1e-5)

        # concatenate ASPP branches & project
        x = Concatenate()([b4, b0, b1, b2, b3])
    else:
        x = Concatenate()([b4, b0])

    x = Conv2D(256, (1, 1), padding='same',
               use_bias=False, name='concat_projection')(x)
    x = BatchNormalization(name='concat_projection_BN', epsilon=1e-5)(x)
    x = Activation('relu')(x)
    x = Dropout(0.1)(x)

    # DeepLab v.3+ decoder

    if backbone == 'xception':
        # Feature projection
        # x4 (x2) block
        x = BilinearUpsampling(output_size=(int(np.ceil(input_shape[0] / 4)),
                                            int(np.ceil(input_shape[1] / 4))))(x)
        dec_skip1 = Conv2D(48, (1, 1), padding='same',
                           use_bias=False, name='feature_projection0')(skip1)
        dec_skip1 = BatchNormalization(
            name='feature_projection0_BN', epsilon=1e-5)(dec_skip1)
        dec_skip1 = Activation('relu')(dec_skip1)
        x = Concatenate()([x, dec_skip1])
        x = SepConv_BN(x, 256, 'decoder_conv0',
                       depth_activation=True, epsilon=1e-5)
        x = SepConv_BN(x, 256, 'decoder_conv1',
                       depth_activation=True, epsilon=1e-5)

    # you can use it with arbitary number of classes
    if classes == 21:
        last_layer_name = 'logits_semantic'
    else:
        last_layer_name = 'custom_logits_semantic'

    x = Conv2D(classes, (1, 1), padding='same', name=last_layer_name)(x)
    x = BilinearUpsampling(output_size=(input_shape[0], input_shape[1]))(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input

    model = Model(inputs, x, name='deeplabv3+')

    # load weights

    if weights == 'pascal_voc':
        if backbone == 'xception':
            weights_path = get_file('deeplabv3_xception_tf_dim_ordering_tf_kernels.h5',
                                    WEIGHTS_PATH_X,
                                    cache_subdir='models')
        else:
            weights_path = get_file('deeplabv3_mobilenetv2_tf_dim_ordering_tf_kernels.h5',
                                    WEIGHTS_PATH_MOBILE,
                                    cache_subdir='models')
        model.load_weights(weights_path, by_name=True)
    return model


if __name__ == '__main__':
    print('load successfully')
 ```
