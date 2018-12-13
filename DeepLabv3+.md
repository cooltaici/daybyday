# Rethinking Atrous Convolution for Semantic Image Segmentation
**paper**：https://arxiv.org/pdf/1802.02611v1.pdf </br>
**Keras 源代码** :https://github.com/mjDelta/deeplabv3plus-keras </br>
**DeeplabV1-V3+**：https://blog.csdn.net/Dlyldxwl/article/details/81148810 </br>
**DeepLabV3+比较好的翻译**：https://blog.csdn.net/zziahgf/article/details/79557105 </br>
**实验代码**：https://github.com/cooltaici/daybyday/blob/master/paper_project/Deeplav_v3plus.py </br>
**2018语义分割类导读**：https://blog.csdn.net/SHAOYEZUIZUISHAUI/article/details/82666764  </br>
### 本文的主要贡献：
第一：对DeepLabv3添加了简单有效的解码模块，这可以大大提高对边界的分割，并且可以通过控制atrous convolution 来控制编码特征的分辨率，来平衡精度和运行时间（已有编码-解码结构不具有该能力）</br>
第二：主打的Xception模块，深度可分卷积结构(depthwise separable convolution) 用到带孔空间金字塔池化(Atrous Spatial Pyramid Pooling, ASPP)模块和解码模块中，得到更快速有效的 编码-解码网络。</br>
&emsp;如Figure 1所示，这是常见的用于语义分割的策略。（1）金字塔结构。（2）编码-解码机构。（3）本文提出的结合两种策略的方案。</br>
![Figure 1](https://paper-reading-1258239805.cos.ap-chengdu.myqcloud.com/DeepLab_v3plus/DeepLabV3plus_Figure1.PNG)</br>
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
![Figure 2](https://paper-reading-1258239805.cos.ap-chengdu.myqcloud.com/DeepLab_v3plus/DeepLabV3plus_Figure2.PNG)</br>

#### 3.2 对Aligned Xception的修改
&emsp;Xception 模型[12]，在ImageNet上已经取得了非常不错的成绩。最近MSRA团队[62]修改了Xception结构并且把它命名为Aligned Xception，这种结构进一步推动了目标检测任务的Xception在目标检测任务中的发展。本文有这些工作中取得了灵感，我们将Xception应用到语义分割领域，特别的是我们在MSRA's的基础上，又增加了一些修改：
- 相同深度 Xception，除了没有修改 entry flow network 结构，以快速计算和内存效率.
- 采用 depthwise separable conv 来替换所有的 max-pooling 操作，以利用 atrous separable conv 来提取任意分辨率的 feature maps.
- 在每个 3×33×3 depthwise conv 后，添加 BN 和 ReLU，类似于 MobileNet.
![Figure 3](https://paper-reading-1258239805.cos.ap-chengdu.myqcloud.com/DeepLab_v3plus/DeepLabV3plus_Figure3.PNG)</br>

### 4. 实验结果
#### 4.1 解码器设计
&emsp;在Deeplabv3[10]中，当我们使用output_stride=16时，使用线性插值上采样比不使用（对金标准下采样）结果要好1.2%。因为上采样也是简单的解码过程。解码模块的设计如第三章节所示（Figure 2）。这里我们用实验验证一下这样设计的理由。</br>
&emsp;**channel的影响**：改变low-level特征中1*1卷积的channle数目，效果变化如Tabel 1所示，本文采取[1*1,48]用于channel匹配。</br>
![Table 1](https://paper-reading-1258239805.cos.ap-chengdu.myqcloud.com/DeepLab_v3plus/DeepLabV3plus_Table1.PNG)</br>
&emsp;**级联后的操作**：我们发现在级联之后，采用3*3的卷积操作，结果最好，如Table 2所示。我们采用了不同的channel数目以及不同的kernel size，发现两个[3*3,256]的效果最理想。</br>
![Table 2](https://paper-reading-1258239805.cos.ap-chengdu.myqcloud.com/DeepLab_v3plus/DeepLabV3plus_Table2.PNG)</br>

#### 4.2 ResNet-101作为骨架网络
&emsp; 我们对ResNet-101作为的解码网络进行了速度与性能的详细测试，如Table 3所示。主要控制的变量有（1）**Baseline**。训练和评估采用不同outout_stride的结果，训练阶段用16，评估阶段用8结果更好。（2）**增加解码器**。增加了解码结构后，网络的表现有1%左右的提升，这里看来，其实解码结果不是本文主打的方向，提升有限。（3）**更加粗糙的编码特征**。如第三列所示，我们使用output_stride=32时，效果并不是很好。</br>
![Table 3](https://paper-reading-1258239805.cos.ap-chengdu.myqcloud.com/DeepLab_v3plus/DeepLabV3plus_Table3.PNG)</br>

#### 4.2 Xception作为骨架网络
&emsp; 我们进一步探究了Xeption作为解码网络的表现。相比[60]，我们做了一些修改，如第三章节描述的那样。</br>
&emsp; “IamgeNet预训练**：我们使用ImageNet数据集对本文中的Xception进行预训练。动量因子0.9，初始学习率0.05，blabla...。如Table 4所示，我们修改后的Xception在误差上要比ResNet-101要小一些。</br>
![Table 4](https://paper-reading-1258239805.cos.ap-chengdu.myqcloud.com/DeepLab_v3plus/DeepLabV3plus_Table4.PNG)</br>

**使用Xception作为编码结果的总体结果如Table 5所示**：可以看出应用所有技术，最高准确率能达到84.56%。随后我们使用output_stride=8进行训练，并且使用batch normalization参数的方法，最后达到了87.8%的准确率，使用JFT数据集后，达到了89.0%。其它流行的方法比较如 Table 6所示。
![Table 5](https://paper-reading-1258239805.cos.ap-chengdu.myqcloud.com/DeepLab_v3plus/DeepLabV3plus_Table5.PNG)</br>
![Table 6](https://paper-reading-1258239805.cos.ap-chengdu.myqcloud.com/DeepLab_v3plus/DeepLabV3plus_Table6.PNG)</br>
