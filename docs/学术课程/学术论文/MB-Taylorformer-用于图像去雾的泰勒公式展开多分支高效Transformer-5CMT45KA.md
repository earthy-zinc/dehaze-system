---
tags: []
parent: 'MB-TaylorFormer: Multi-branch Efficient Transformer Expanded by Taylor Formula for Image Dehazing'
collections:
    - 图像去雾
version: 4007
libraryID: 1
itemKey: 5CMT45KA

---
# MB-Taylorformer 用于图像去雾的泰勒公式展开多分支高效Transformer

ICCV 2023

## 摘要

近些年，Transformer由于其全局感受野和对输入的自适应的特点，在计算机视觉研究中逐渐取代纯卷积神经网络。但是Transformer当中的Softmax-Attention二次计算复杂度限制了它在图像去雾领域的应用，尤其是针对高分辨率图像上。为了解决这个问题，作者提出了Transformer的变体TaylorFormer，具体是用泰勒展开式近似Softmax-Attention操作，使用一种多尺度注意力细化模块修正泰勒展开式产生的误差，总体能够达到线性复杂度。此外，作者将具有多尺度块嵌入的多分支结构引入到TaylorFormer中，该结构通过重叠不同尺度的可变形卷积来嵌入特征。设计采用了如下操作：

1.  各种大小的感受野
2.  多层次的语义信息
3.  灵活的感受野形状

## 介绍

具体来讲，Transformer应用到图像去雾领域有以下几点局限性。

1.  Transformer的计算复杂度和特征图的分辨率呈现平方关系，因此不适合逐像素的去雾任务。虽然有些工作在小空间窗口上应用自注意力来缓解这个问题，但是会让Transformer的感受野受到限制。
2.  视觉领域的Transformer基本单元应该具有更灵活的尺度，但是当前的Transformer一般啊是通过固定卷积核来生成固定尺度的token。所以还有很多改进空间。

为了解决第一个挑战，作者提出了TaylorFomer，他在跨空间维度上对整个特征应用自注意力，也就是通过对Softmax进行泰勒展开计算自注意力权重，然后应用矩阵乘法结合律将复杂度由O(n<sup>2</sup>)降低到O(n)。具有以下优点：

1.  保留了Transformer对前后数据之间长距离依赖关系建模的能力，避免拆分窗口造成的感受野减小。
2.  相比于使用卷积核自注意力方法，该方法提供了更精确的近似值，更接近于原始的Transfomer
3.  该方法使得Transformer更加关注于像素级别的交互，而不仅仅是通道级别的，这将允许对特征进行更细粒度的处理

考虑到泰勒公式中忽略皮阿诺余项带来的误差，作者紧接着引入了多尺度注意力细化模块

Referred in <a href="./学术论文笔记汇总-RYZ5DF37.md" rel="noopener noreferrer nofollow" zhref="zotero://note/u/RYZ5DF37/?ignore=1&#x26;line=-1" ztype="znotelink" class="internal-link">图像去雾</a>

Referred in <a href="./学术论文笔记汇总-RYZ5DF37.md" rel="noopener noreferrer nofollow" zhref="zotero://note/u/RYZ5DF37/?ignore=1&#x26;line=-1" ztype="znotelink" class="internal-link">图像去雾</a>

Referred in <a href="./学术论文笔记汇总-RYZ5DF37.md" rel="noopener noreferrer nofollow" zhref="zotero://note/u/RYZ5DF37/?ignore=1&#x26;line=-1" ztype="znotelink" class="internal-link">Workspace Note</a>

<a href="./学术论文笔记汇总-RYZ5DF37.md" rel="noopener noreferrer nofollow" zhref="zotero://note/u/RYZ5DF37/" ztype="znotelink" class="internal-link">MB-Taylorformer 用于图像去雾的泰勒公式展开多分支高效Transformer</a>

Referred in <a href="./学术论文笔记汇总-RYZ5DF37.md" rel="noopener noreferrer nofollow" zhref="zotero://note/u/RYZ5DF37/?ignore=1&#x26;line=-1" ztype="znotelink" class="internal-link">Workspace Note</a>
