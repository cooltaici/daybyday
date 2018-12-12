# RefineNet: Multi-Path Refinement Networks for High-Resolution Semantic Segmentation
[原文]（https://arxiv.org/pdf/1611.06612.pdf）
**摘要**：
目前深度卷积网络在目标识别和图像分割等问题上表现突出，但频繁的下采样丢失了原图像的信息。我们提出一种RefineNet网络，使用残差链接显式将各个下采样层和后面的网络层结合在一起。这样网络高层的语义特征可以直接从底层的卷积层中获得refine。一个RefineNet单元使用残差链接和identity映射，对于端对端的训练很有效。我们也介绍了一种链接残差池化，它可以捕获大量背景信息。
### 背景
深度学习在语义分割，最近比较火的有[Deeplab2],和[FCN]。对于Deeplab2使用[空洞卷积]实现了大的感受野，但是有两个缺点。第一，需要计算大量的卷积特征图在高纬上而引起计算代价太大，这限制了计算高层特征和输出尺度只能为输入的1/8。第二，dilate卷积引起粗糙下采样特征，这潜在导致重要细节的损失。另外的方法FCN融合中层特征和高层特征。这种方法基于中层特征保持了空间信息。这种方法尽管补充了特征如边界、角落等，但缺乏强大的空间信息.
**本文的主要贡献**
- 我们提出了RefineNet，很好的融合了多层特征
- 我们在网络中应用了残差连接（恒等映射）
- 我们提出了链式残差池化（chained residual pooling）的结构
- 新提出的RefineNet取得了包括PASCAL VOC 2012, PASCAL-Context,NYUDv2等7个公开数据库最好的效果
### 1.近来的研究
略
### 2.研究背景
略
### 3.本文提出的方法
![Figure 2](https://paper-reading-1258239805.cos.ap-chengdu.myqcloud.com/RefineNet_Fig2.PNG)
**翻译**：全卷积网络在dense classification上的比较。对于（a）来说，ResNet由于大量的下采样，损失很多细节信息。（b）在最后两个block使用空洞卷积代替了下采样，但是难以训练并且耗费太多内存。（c）我们提出的方法利用了不同层次的特征，并且把它们进行融合得到高分辨率的预测，并且不需要保持大量维度的中间层特征。具体细节如图3所示。
![Figure 3](https://paper-reading-1258239805.cos.ap-chengdu.myqcloud.com/RefineNet_Fig3.PNG)
**翻译**：在本文中使用短残差网络连接（RefineNet内部）和长残差网络连接（ResNet和RefineNet模块之间）
#### 3.1 Multi-Path Refinement
&emsp;我们把在ImageNet数据集上训练好的ResNet按照map大小分成4个blok。我们构造了串联的4个RefineNet，其中
每个RefineNet一个输入端连接一个ResNet block，另一个输入端连接上一个RefineNet。这种设计并不是唯一的，RefineNet可以接受多种ResNet的输入，我们在4.3章节将会分析2-cascaded， single-block以及2-scale 7-path版本。如Fig.2(c)所示，每个RefineNet一端接受更抽象的语义信息，一端接受更信息的细节信息RefineNet1通过上采样恢复到原图大小，进行预测。
#### 3.2 RefineNet
&emsp;RefineNet网络结果如Fig.3(a)所示。RefineNet网络结构是灵活的，每个Refine block可以设置任意多个ResNet block作为输入。
**Residual convolution unit**。每一个RefineNet的开头有两个由 残差卷积单元构成(RCU)的构成的线性结构。有点类似ResNet的结构，但是去掉了所有的batch-normalization层（如Fig.3(b)所示)。RefineNet-4中的RCU的filter数目设置为512，其它设置为256。图中确实可以看出RefineNet的输入可以是一个ResNet block也可以是多个。
**Multi-resolution fusion**。每一路2RCU的输出要先通过一个convolution进行维度匹配，然后上采样到最大分辨率也就是（所有输入特征map）,最后将所有路特征相加融合在一起。如Fig.3（c）所示，ResNetblock-4不需要进行处理。
**Chained resdual pooling**。Multi-resolution fusion的输出，最后通过**链式残差池化**模块，如Fig.3(d)所示。它的目的是为了为了在高分辨率的map中获取背景信息。这个模块由一系列串联在一起的池化模块，每个模块包含一个池化层和一个卷积层。卷积层是作为融合的权重层，它能够学习到每个池化层的重要性。
**Output convolutions**。每一个Refinenet的最后也是一个的残差卷积结构。
### 4.实验
#### 4.1 Object Parsing
#### 4.2 Segmantic Segmentation
**PASCAL VOC 2012**
&emsp;在PASCAL VOC 2012的结果如Table 5所示。
![Table 5](https://paper-reading-1258239805.cos.ap-chengdu.myqcloud.com/RefineNet_Table5.PNG)
**PASCAL-Context**
&emsp;PASCAL-Context的结果如Table 6所示。
![Table 6](https://paper-reading-1258239805.cos.ap-chengdu.myqcloud.com/RefineNet_Table6.PNG)
#### 4.3 RefineNet的串联变体
&emsp;前面已经介绍过，RefineNet结构是很灵活的，这里我们分析了 a single RefineNet，a 2-cascaded RefineNet 和 a 4-cascaded RefineNet with 2-scale ResNet。这三种变体，结构如**图7**所示。
![Figure 7](https://paper-reading-1258239805.cos.ap-chengdu.myqcloud.com/RefineNet_figure7.PNG)
&emsp;single RefineNet model 结构很简单，只有一个RefineNet block，把所有4个ResNet输入作为一个RefineNet Block结构的输入。2-cascaded RefineNet 结构和咱们在图2（C）中描述的很类似，但是只用了两个RefineNet block。2-scale的版本，有两个尺度的输入，也就是两个独立的ResNet来提取特征。这三种变体和本文中主要的结构对比结果是Table 9所示。
![Table 9](https://paper-reading-1258239805.cos.ap-chengdu.myqcloud.com/RefineNet_Table9.PNG)



