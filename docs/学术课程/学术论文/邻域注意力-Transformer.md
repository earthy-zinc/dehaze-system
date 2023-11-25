---
tag:
    - 'Computer Science - Artificial Intelligence'
    - 'Computer Science - Computer Vision and Pattern Recognition'
    - 'Computer Science - Machine Learning'
title: 'Neighborhood Attention Transformer'
category:
    - 图像增强
version: 6273
libraryID: 1
itemKey: 4GF2VNSK

---
# 邻域注意力 Transformer

Comment: To appear in CVPR 2023. NATTEN is open-sourced at: https\://github.com/SHI-Labs/NATTEN/

邻域注意力机制( Neighborhood Attention，NA )，这是第一个有效的、可扩展的视觉滑动窗口注意力机制。NA是一种逐像素操作，将自注意力( self attention，SA )定位到最近邻像素，因此与自注意力的二次复杂度相比，邻域注意力具有线性的时间和空间复杂度。

基于自注意力的Vision Transformer token的数量通常与图像分辨率呈线性相关。因此，较高的图像分辨率会导致复杂度和内存使用量的平方增加。

另一个问题是卷积操作受益于诸如局部性和二维空间结构等归纳偏差，而采用点积方法的自注意力是一个全局的一维操作。这意味着一些归纳偏差必须通过大量的数据或先进的训练技术和数据增强增强来学习。

因此，局部注意力模块被提出来缓解这些问题。独立式自注意力机制( Stand-Alone Self-Attention，SASA )是基于局部窗口的视觉注意力机制的最早应用之一，其中每个像素会关注其周围的窗口。它的滑动窗口模式与卷积相同，因此保持平移等变性。

邻域注意力机制( Neighborhood Attention，NA )将自注意力定位到每个像素的最近邻居，这并不一定是像素周围的固定窗口。这种定义上的变化允许所有像素保持相同的注意力范围，否则将减少零填充方案独立式自注意力机制中的角点像素。随着邻域的增大，邻域注意力也逐渐逼近自注意力，并且在最大邻域处与自注意力等价。此外，NA还具有保持平移等变性的额外优势。
