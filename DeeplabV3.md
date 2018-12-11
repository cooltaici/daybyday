# Rethinking Atrous Convolution for Semantic Image Segmentation
**地址**：https://arxiv.org/abs/1706.05587
**Tesorflow 源代码**:https://github.com/NanqingD/DeepLabV3-Tensorflow
**Keras 源代码** :https://github.com/Shmuelnaaman/deeplab_v3
**其它资料**：https://blog.csdn.net/u011974639/article/details/79144773
**摘要**：
&emsp;DeepLabv3进一步探讨空洞卷积，这是一个在语义分割任务中：可以调整滤波器视野、控制卷积神经网络计算的特征响应分辨率的强大工具。为了解决多尺度下的目标分割问题，我们设计了空洞卷积级联或不同采样率空洞卷积并行架构。此外，我们强调了ASPP(Atrous Spatial Pyramid Pooling)模块，该模块可以在获取多个尺度上卷积特征，进一步提升性能。同时，我们分享了实施细节和训练方法，此次提出的DeepLabv3相比先前的版本有显著的效果提升，在PASCAL VOC 2012上获得了先进的性能。
### Introduction
DeepLabv3的主要贡献在于：
- 本文重新讨论了空洞卷积的使用，这让我们在级联模块和空间金字塔池化的框架下，能够获取更大的感受野从而获取多尺度信息。
- 改进了ASPP模块：由不同采样率的空洞卷积和BN层组成，我们尝试以级联或并行的方式布局模块。
- 讨论了一个重要问题：使用大采样率的3×3的空洞卷积，因为图像边界响应无法捕捉远距离信息，会退化为1×1的卷积, 我们建议将图像级特征融合到ASPP模块中。
- 阐述了训练细节并分享了训练经验，论文提出的”DeepLabv3”改进了以前的工作，获得了很好的结果。
### Related Work
&emsp;现有多个工作表明全局特征或上下文之间的互相作用有助于做语义分割，我们讨论四种不同类型利用上下文信息做语义分割的全卷积网络。
&emsp;**图像金字塔(Image pyramid)**： 通常使用共享权重的模型，适用于多尺度的输入。小尺度的输入响应控制语义，大尺寸的输入响应控制细节。通过拉布拉斯金字塔对输入变换成多尺度，传入DCNN，融合输出。这类的缺点是：因为GPU存储器的限制，对于更大/更深的模型不方便扩展。通常应用于推断阶段。
&emsp;**编码器-解码器(Encoder-decoder)**： 编码器的高层次的特征容易捕获更长的距离信息，在解码器阶段使用编码器阶段的信息帮助恢复目标的细节和空间维度。例如SegNet利用下采样的池化索引作为上采样的指导；U-Net增加了编码器部分的特征跳跃连接到解码器；RefineNet等证明了Encoder-Decoder结构的有效性。
&emsp;**上下文模块(Context module)**：包含了额外的模块用于级联编码长距离的上下文。一种有效的方法是DenseCRF并入DCNN中，共同训练DCNN和CRF。
&emsp;**空间金字塔池化(Spatial pyramid pooling)**：采用空间金字塔池化可以捕捉多个层次的上下文。在ParseNet中从不同图像等级的特征中获取上下文信息；DeepLabv2提出ASPP，以不同采样率的并行空洞卷积捕捉多尺度信息。最近PSPNet在不同网格尺度上执行空间池化，并在多个数据集上获得优异的表现。还有其他基于LSTM方法聚合全局信息。我们的工作主要探讨空洞卷积作为上下文模块和一个空间金字塔池化的工具，这适用于任何网络。具体来说，我们取ResNet最后一个block，复制多个级联起来，送入到ASPP模块后。我们通过实验发现使用BN层有利于训练过程，为了进一步捕获全局上下文，我们建议在ASPP上融入图像级特征.
### 3 Methods
#### 3.1 孔洞卷积用于密集特征提取
&emsp;首先，孔洞卷积可以控制dilation_rate来随意控制感受也的大小。其次，孔洞卷积还可以随意控制在全连接网络中的特征密集程度，在本文中，将**output_stride**定义为从输入图像到输出特征的采样比率。比如说在分类网络中，最后的特征响应是输入的1/32，那么output_stride就是32。如果想要对特征响应大小进行翻倍，那么最后一层的池化层或者卷积层的stride要设置为1，然后所有后面的卷积层采用dilation_rate为2的孔洞卷积</br>
#### 3.2 孔洞卷积用于更深层次模型
&emsp;首先，我们探索吧孔洞卷积模块进行级联的情况。为了实现这种，我们从block4开始复制，如Fig 3所示。每个block有三个3*3卷积层，最后一个卷积的stride为2，但block7除外。这个模型的灵感来源于：stride可以在deep网络更容易获取长范围的信息。如Fig 3a所示，我们发现连续的stride过程不利于语义分割，因为细节信息丢失了很多，参考Table 1结果。所以如Fig 3b所示，我们使用孔洞卷积，使得output_stride达到16。</br>
![Figure 3]()</br>
##### 3.2.1 Multi_grid 策略
&emsp;由论文[4,81,5,67]以及[84,18]中得到的multi_grid策略。在bock4到block7之间采取不同dilation_rate的孔洞卷积。特别的，把每个block (block4-block7)中的三个卷积层的unit rate定义为Multi_Grid=(r1,r2,r3)。最终每个卷积层的dilation rate是整个block rate与Multi_Grid相乘。比如在output_stride=16的block4中，如果Multi_Grid = (1,2,4)， 那么，三个卷积的的dilation rate = 2*(1,2,4) = (2,4,8)。</br>
#### 空间孔洞金子塔池化（Atrous Spatial Pyramif Pooling）
&emsp;我们回顾了在文献[11]中提到的ASPP结构，他的灵感来自于SSP_Net[28,49,31]的成功。但是不同于[11]的是，我们在ASPP结构中添加了batch normalization层。</br>
&emsp;采取不同dilation rate的ASPP结构可以提取multi-scale信息。但是我们发现，当采样率变大的时候，有效的卷积核权重在减少。就像Figure 4所示的一样，当用3*3的卷积核去处理65*65的特征时，如果dilation rate达到了65这种级别，3*3的卷积核就相当于1*1的卷积核。
![Figure 4]()</br>
&emsp;为了解决这个问题并且融合全局信息。像[58,59]那样，融合图像级的特征。也就是说我们在每个block里面增加了一个256*1*1的卷积操作（带BN），然后双线性插值上采样到想要的维度。最后，当output_stride为16的时候，我们的ASPP结构包含了一个1*1的卷积，3个dilation rate为（6,12,18）的并行卷积结构，所有的filter数目为256。如Figure 5所示，所有的branch分支结构，最后串联一起，并通过一个256*1*1的卷积操作。</br>
![Figure 5]()</br>