# Rethinking Atrous Convolution for Semantic Image Segmentation
**paper**：https://arxiv.org/pdf/1802.02611v1.pdf
**Keras 源代码** :https://github.com/mjDelta/deeplabv3plus-keras
**DeeplabV1-V3+**：https://blog.csdn.net/Dlyldxwl/article/details/81148810
**DeepLabV3+比较好的翻译**：https://blog.csdn.net/zziahgf/article/details/79557105
#### 一些自己的看法
本文的主要贡献：
第一：对DeepLabv3添加了简单有效的解码模块，这可以大大提高对边界的分割，并且可以通过控制atrous convolution 来控制编码特征的分辨率，来平衡精度和运行时间（已有编码-解码结构不具有该能力）
第二：主打的Xception模块，深度可分卷积结构(depthwise separable convolution) 用到带孔空间金字塔池化(Atrous Spatial Pyramid Pooling, ASPP)模块和解码模块中，得到更快速有效的 编码-解码网络。