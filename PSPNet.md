#Semantic Segmentation--Pyramid Scene Parsing Network(PSPNet)
**paper**:https://arxiv.org/pdf/1612.01105.pdf </br>
**官方caffe源码**：https://github.com/hszhao/PSPNet </br>
**keras代码例子**：https://github.com/Vladkryvoruchko/PSPNet-Keras-tensorflow </br>
**keras代码例子**：https://github.com/ykamikawa/PSPNet </br>
**tensorflow代码例子**：https://github.com/hellochick/PSPNet-tensorflow </br>
### 1 介绍
&emsp;场景分割是机器视觉中最重要的任务之一。常见的数据集有LMQ[22]，PASCAL VOC[8,29] 以及ADE20K[43](最具有挑战的一个）。当前最热门的语义分割模型是FCN[26]，但是FCN仍有不足的地方，对于高层语义信息之间的关联还不是很强，如Figure 2所示，一个小船被识别为汽车。从语义之间的关联来看，小船的周围是水，那么从语义的关联性来看，是不应该把小船识别成汽车的。</br>
![Figure 2](https://github.com/cooltaici/daybyday/blob/master/picture_paper/PSPNet/PSPNet_Figure2.PNG)</br>
&emsp;我们知道了FCN的问题是缺少了全图场景的类别暗示。空间金字塔池化[18]可以很好的这种Image-level信息。不同于这些方法，为了更好的融合全局特征，我们提出了金字塔语义分割图像(PSPNet）。和FCN不同，我们把像素级特征和全局特征很好的融合在一起。此外，我们提出了一个优化方法以及损失函数。我们将公开所有实现细节，并且代码也是公开的。
- 提出了一个金字塔场景解析网络，能够将难解析的场景信息特征嵌入基于FCN预测框架中
- 在基于深度监督损失ResNet上制定有效的优化策略
- 构建了一个实用的系统，用于场景解析和语义分割，并包含了实施细节
### 2. 近年来的工作

### 3. Pyramin Scene Parsing Network
&emsp;我们的PSPNet网络结果如Figure 3所示。</br>
![Figure 3](https://github.com/cooltaici/daybyday/blob/master/picture_paper/PSPNet/PSPNet_Figure3.PNG)</br>
#### 3.1 重要的发现
&emsp;新的ADE20K数据集[43]包含了150个目标类别，以及1038个图像级场景描述（卧室，马路等）。根据FCN算法在ADE20K数据集的表现，我们总结了几个关于复杂场景分割的几个常见问题。</br>
**不匹配的类别**：语义之间的联系是重要的，也就是说目标之间是有联系的（**注：这在肿瘤分割中可能不是很适用**）。如果不能收集到丰富的语义特征将会增加错误识别的风险。</br>
**模糊的目标**：很多在ADE20K中的目标比较模糊。比如野外和土地，山和丘陵，建筑和摩天大楼，它们都拥有类似的外表。专业的标注人员也会有17.60%的误差[43]。比如Firgure 2中第二行，FCN把方框里的区域，一部分些判断成摩天大楼，一部分判断成建筑。这种样本应该提出在外，也就是说整个图片要么是摩天大楼，要么是建筑。但这是不必要的，因为这种错误是可以挽救的，只要能够利用类别之间的关系。</br>
**不明显的目标**：现实场景总是包含了不同大小个各种目标，有些比较小的目标，比如街灯和广告牌，是很难检测到的，然而这些目标又是非常重要的。相反，大的目标可能会超出FCN的感受野。像Figure 2中第三行所显示的那样，枕头和被子的外表看起来很类似，如果忽略了场景信息，那么将无法检测出枕头。所以综上所述，要提高检测大小不同的目标，需要更加注意包含不明显目标的不同的子区域。</br>
总结来说。很多的错误都与语义之间的关系以及不同感受野的全局信息有关。所以，拥有全局场景相关的先验知识，将会提高语义分割的能力。</br>

#### 3.2 金字塔池化模块
&emsp;尽管理论上来说，对于ResNet[13]的感受野已经大于输入图像了，但是Zhou等[42]中表明CNN网络的感受野一般要比理论上要小得多。这使得网络没有充分包含全局的先验信息。我们通过提出一个好的全局表示模块来解决这个问题。</br>

&emsp;全局池化是提取全局语义的一个好的方法，并且成功应用到图像识别领域[34,13]。In[24]，它也成功应用到语义分割上。但是对于复杂的场景图像ADE20K[43]，这种策略还不能包含必要的信息。直接将信息串联起来会失去空间信息，更有效的做法是想办法融合不同感受野的特征。这种结论文献[18,12]中也有提到。</br>

&emsp;在文献[12]中，全局池化后，将特征flatten，并连接全连接层，这种作为解除了对输入的大小的限制。为了更好地减少不同感受野特征的损失，本文提出了一个融合不同尺度不同子区域特征的结构。我们把它叫做**pyramid pooling module**,如Figure 3c所示，它位于DCNN特征map的后面。</br>

&emsp;金字塔池化结构融合了四个不同尺度的信息。最上面一层红色的是全局池化，每个特征图只负责一个输出。下面三个就是不同尺度的池化，将特征图分成一个个小格子，每个格子代表的子区域，负责输出一个信息。每一个池化操作后，进行1*1的卷积操作，这个卷积操作可以扮演融合权重的作用（在Deeplabv3，RefineNet里面也有提到），也可以调整channel进行维度匹配（把其它尺度语义channel缩小N倍）。最后，在级联之前，把所有路的特征上采样到相同size。</br>

&emsp;要注意的是，金字塔池化的路数以及各路的size是可以修改的（当然）。我们的金字塔的输出格式为1*1，2*2，3*3以及6*6。关于使用平均池化或者最大池化的差别，我们在5.2章节给出差异比较。</br>


#### 3.3 网络结构
&emsp; 本文的总体网络如Figure 3所示。在经过预训练的ResNet上使用dilated方案[3,40]，获取1/8大小的特征图。如Figure 3c所示，这个特征图要经过金字塔池化这个模块，得到全局的先验知识。然后将全局特征上采样并且和之前的特征图进行级联，最后通过另一个卷积得出最终输出。

### 4. 对Resnet-based FCN 进行深度监督
&emsp; 深度预训练的网络已经取得了不错的结果[17,33,13]，但是对于图像识别人物来说，越深的网络训练起来越复杂[32,19]，ResNet通过残差连接解决了这个问题。</br>
![Figure 4](https://github.com/cooltaici/daybyday/blob/master/picture_paper/PSPNet/PSPNet_Figure4.PNG)</br>
&emspc; 除了最后的softmax损失函数外，我们另外针对ResNet设计了辅助损失，如Figure 4所示。在测试阶段，我们将会丢掉辅助损失分支。</br>

### 5. 实验
&emsp; 我们提出的方法常语义分割挑战任务中取得很大的成功。本章节对三个数据集进行测试。(1) ImageNet parsing challenge 2016[43]。(2) PASCAL VOC 2012 [8]。 (3) urban scene understanding dataset Cityscapes[6]。</br>
#### 5.1 实现细节
&emsp;**训练参数设置**： 由DeepLab获得的灵感，我们采用了poly学习率更新策略[4]。初始学习率为0.01，power为0.9。不同数据集迭代次数不同：ImageNet 150K，PASCAL VOC 30K，Cityscape 90K。动量项和权重衰减分别为0.9和0.0001。其它的数据扩增如：随机缩放（0.5, 2)，随机旋转（-10,10），随机高斯模糊。这些数据扩增方案可以避免过拟合。我们的孔洞卷积方案和DeepLab相似。</br>

&emsp; 在实验过程中，我们发现大的”cropsize“能获得比较好的效果，另外" batchsize" 在Batch Normalization也很重要。由于GPU的内存限制，我们将Batchsize设置为16.</br>

#### 5.2 ImageNet Scene Parsing Challenge 2016

&emsp; 为了寻找PSPNet最优的效果。我们测试了均值池化和最大池化，只要一个全局池化以及不同尺度池化，在池化操作后以及级联前有没有惊醒维度衰减。所有的结果如Table 1所示。(1)均值池化要比最大池化效果好。(2)金字塔池化模块要比单一使用全局池化要好。(3)维度衰减效果更好。</br>
![Table 1](https://github.com/cooltaici/daybyday/blob/master/picture_paper/PSPNet/PSPNet_Table1.PNG)</br>

&emsp;**辅助损失测试**：辅助损失权重设置从0到1，测试结果如Table 2所示，当权重为0.4的时候效果最好。
![Table 2](https://github.com/cooltaici/daybyday/blob/master/picture_paper/PSPNet/PSPNet_Table2.PNG)</br>

&emsp; **网络深度测试**：在图像分类任务中，越深的网络一般获得更好的效果。我们对ResNet的网络深度进行了测试，分别有{ 50, 101, 152, 269}，如 Figure 5所示，网络越深，效果越好。结果统计如Table 3所示。</br>
![Table 3](https://github.com/cooltaici/daybyday/blob/master/picture_paper/PSPNet/PSPNet_Table3.PNG)</br>
![Figure 5](https://github.com/cooltaici/daybyday/blob/master/picture_paper/PSPNet/PSPNet_Figure5.PNG)</br>

&emsp; **其它测试**：其它相关测试如Table 4所示，辅助损失、均值池化、数据扩增以及多尺度输入等方案都有助于性能的提升。</br>
![Table 4](https://github.com/cooltaici/daybyday/blob/master/picture_paper/PSPNet/PSPNet_Table4.PNG)</br>

&emsp; **比赛结果**：我们提出的方法 ImageNet scene parsing 2016中获得了第一名。Table 5展示了部分比赛结果。
![Table 5](https://github.com/cooltaici/daybyday/blob/master/picture_paper/PSPNet/PSPNet_Table5.PNG)</br>

``` python
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
    interp_block2 = interp_block(res, 2, feature_map_size, input_shape)
    interp_block3 = interp_block(res, 3, feature_map_size, input_shape)
    interp_block6 = interp_block(res, 6, feature_map_size, input_shape)

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

    x = Conv2D(nb_classes, (1, 1), strides=(1, 1), name="conv6")(x)
    # x = Lambda(Interp, arguments={'shape': (
    #    input_shape[0], input_shape[1])})(x)
    x = Interp([input_shape[0], input_shape[1]])(x)
    x = Activation('softmax')(x)

    model = Model(inputs=inp, outputs=x)

    # Solver
    sgd = SGD(lr=learning_rate, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

if __name__ == '__main__':
    model = build_pspnet(1, 101, (473,473), out_activation='softmax')   #(473,473)（713,713）（640,480）
    model.load_weights("weights/keras/pspnet101_voc2012.h5")
    model.summary()
    print('load successfully')

 ```
