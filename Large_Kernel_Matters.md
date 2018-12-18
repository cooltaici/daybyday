# Large Kernel Matters —— Improve Semantic Segmentation by Global Convolutional Network
**paper**: https://arxiv.org/pdf/1703.02719.pdf </br>
**摘要**：在PASCAL VOC 2012中获得了82.2%的准确率。</br>

### 1. 前言
&emsp; 我们的贡献：（1）我们提出了“Global Convolution Network" （GCN）同时解决分类和定位的问题，如Figure 2所示。（2）提出了一个边界优化模块，进一步提高目标边界的定位准确性。（3）我们在PASCAL VOC 2012数据集上获得了最好的效果 82.2%。</br>
![Figure 1](https://github.com/cooltaici/daybyday/blob/master/picture_paper/LargeKernel/LargeKernel_Figure1.PNG)</br>

### 3. 方法
&emsp; 在这一章节，我们介绍了一个非常好用的 GCN网络，整个网络的结构如Figure 2所示。</br>
![Figure 2](https://github.com/cooltaici/daybyday/blob/master/picture_paper/LargeKernel/LargeKernel_Figure2.PNG)</br>
#### 3.1 全局卷积网络（GCN）
