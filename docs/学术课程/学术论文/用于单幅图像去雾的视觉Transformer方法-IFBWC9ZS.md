---
tags: []
parent: 'Vision Transformers for Single Image Dehazing'
collections:
    - 图像去雾
version: 4493
libraryID: 1
itemKey: IFBWC9ZS

---
# 用于单幅图像去雾的视觉Transformer方法

## 相关工作

作者发现，Vision Transformer中的层归一化和GELU激活函数会降低图像去雾性能，层归一化会分别对图像块对应的token进行归一化处理，导致图像块之间的相关性丢失，因此作者去掉了多层感知器之前的归一化层，用Rescale归一化来代替层归一化。Rescale归一化对整个特征图进行归一化处理，重新引入归一化后丢失的特征图的均值和方差。GELU在高级视觉任务中表现较好，但是ReLu激活函数在图像去雾方向表现好于GELU，作者认为GELU引入的非线性特性在解码时不易逆转。图像去雾不仅要求网络编码有高表达能力特征，还要求这些特征要容易恢复为图像域信号。

提出了DehazeFormer，修改了归一化层、激活函数、空间信息聚合方案。

Referred in <a href="./学术论文笔记汇总-RYZ5DF37.md" class="internal-link" zhref="zotero://note/u/RYZ5DF37/?ignore=1&#x26;line=-1" ztype="znotelink" class="internal-link">Workspace Note</a>
