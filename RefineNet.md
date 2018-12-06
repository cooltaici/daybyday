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
### 近来的研究
略
### 研究背景
略
### 本文提出的方法
![Figure 2](https://paper-reading-1258239805.cos.ap-chengdu.myqcloud.com/RefineNet_Fig2.PNG)
**翻译**：全卷积网络在dense classification上的比较。对于（a）来说，ResNet由于大量的下采样，损失很多细节信息。（b）在最后两个block使用空洞卷积代替了下采样，但是难以训练并且耗费太多内存。（c）我们提出的方法利用了不同层次的特征，并且把它们进行融合得到高分辨率的预测，并且不需要保持大量维度的中间层特征。具体细节如图3所示。
![Figure 3](https://paper-reading-1258239805.cos.ap-chengdu.myqcloud.com/RefineNet_Fig3.PNG)
**翻译**：在本文中使用短残差网络连接和长残差网络连接
我们把在ImageNet数据集上训练好的网络分成4个blok,

