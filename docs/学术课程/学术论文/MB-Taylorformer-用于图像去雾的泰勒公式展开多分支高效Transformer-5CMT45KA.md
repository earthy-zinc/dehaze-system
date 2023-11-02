---
tags: []
parent: 'MB-TaylorFormer: Multi-branch Efficient Transformer Expanded by Taylor Formula for Image Dehazing'
collections:
    - 图像去雾
version: 4494
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

考虑到泰勒公式中忽略皮阿诺余项带来的误差，作者紧接着引入了多尺度注意力细化模块对TaylorFormer进行细化。首先通过卷积transformer中局部的query和key，来利用图像内局部相关性，从而输出带有缩放因子的特征图。特征图通道数等于多头自注意力的头数，因此每个头都有相应的缩放因子。

受到某些去雾网络中的的Inception模块和可变形卷积（Deformable convolution）的启发，作者提出了一种基于嵌入多尺度块的多分支编码器-解码器。命名为MB-TaylorFormer。嵌入到网络骨架中的的多尺度块由不同大小的感受野、灵活的感受野形状和多层次的语义信息。因为生成的每个Token应该遵循局部相关性先验，所以作者对可变形卷积的偏移量进行了截断。通过深度可分离卷积的方法降低了计算的复杂度和参数数量，然后将来自不同尺度的token独立的输入到TaylorFormer中，最后进行融合，多尺度块嵌入这个模块能够生成不同尺度和维度的token。这些多尺度的结构能够同时处理这些token来捕获更强大的特征。

简单来说，本文的工作主要在于以下三点。

1.  提出了一种新的基于泰勒展开的线性Transformer来建模像素之间长距离相互作用，为了修正错误引入了MSAR模块
2.  设计了多尺度块嵌入的多分支结构，多个字段大小，灵活的感受野形状和多层次语义信息能够生成多尺度token，捕获更强大的特征。
3.  性能好

## MB-TaylorFormer

### 多尺度骨干网络

1.  使用具有四个阶段的编码器-解码器网络进行深度特征提取。每个阶段都包含由多尺度块嵌入和多分支Transformer残差块。多尺度块嵌入能够生成多尺度token，分别输入到多个Transformer分支，每个分支包含多个transformer编码器
2.  多分支Transformer块的末尾使用SKFF块来融合不同分支产生的特征。
3.  在每个阶段分别对长采样和下采样应用像素逆shuffle和像素shuffle操作。利用跳跃连接聚合编码器和解码器信息。使用1 \* 1 卷积降维
4.  在编码器-解码器结构后面使用一个残差块恢复精细结构和纹理细节。
5.  最后，使用3 \* 3卷积层减少通道，输出残差图像，最后得到恢复图像。
6.  进一步压缩计算量和参数，使用了深度可分离卷积

### 多尺度Patch嵌入

图片上的物体在大小尺度上具有很大的变化，现有的工作在嵌入模块时应用的是固定大小的卷积核，这样获得的特征尺度非常单一的，不利于提取不同尺度的特征。为了解决这个问题，作者设计了多尺度块嵌入方法。有三个特性

1.  多尺寸的感受野
2.  多层次的语义信息
3.  灵活的感受野形状

具体方法就是，通过并行设计多个具有不同尺度卷积核的可变形卷积(DCN)这样就能够同时生成粗略的和精细的特征，具有灵活的建模能力。

由于堆叠卷积层能够扩大感受野，作者堆叠了几个具有小核的可变形卷积层，而不是单个具有大核的卷积层。这样不仅增加了网络深度，从而提供多层次的语义信息，而且有助于减小参数和计算负担。作者在可变形卷积层后面使用Hardswish激活函数

类似于深度可分离卷积，提出了深度可分离可变形卷积，这种卷积对DCN部分进行深度卷积和逐点卷积。降低了计算复杂度和参数数量。考虑到图像具有局部相关性，而patch嵌入捕捉了特征图的基本要素，因此应该更侧重于局部区域。作者通过阶段偏移量来控制patch嵌入层的感受野范围。根据物体的大小来通过学习选择感受野的大小上限是9 \* 9.相当于BF=4的扩张卷积。

### 泰勒拓展的多头自注意力

### 多尺度注意力细化

<span class="highlight" data-annotation="%7B%22attachmentURI%22%3A%22http%3A%2F%2Fzotero.org%2Fusers%2F10046823%2Fitems%2FN3ZRF8J7%22%2C%22pageLabel%22%3A%223%22%2C%22position%22%3A%7B%22pageIndex%22%3A2%2C%22rects%22%3A%5B%5B328.04%2C157.398%2C440.38%2C167.25%5D%5D%7D%2C%22citationItem%22%3A%7B%22uris%22%3A%5B%22http%3A%2F%2Fzotero.org%2Fusers%2F10046823%2Fitems%2FTPWAHTCZ%22%5D%2C%22locator%22%3A%223%22%7D%7D" ztype="zhighlight"><a href="zotero://open-pdf/library/items/N3ZRF8J7?page=3">“Multi-branch Backbone”</a></span> <span class="citation" data-citation="%7B%22citationItems%22%3A%5B%7B%22uris%22%3A%5B%22http%3A%2F%2Fzotero.org%2Fusers%2F10046823%2Fitems%2FTPWAHTCZ%22%5D%2C%22itemData%22%3A%7B%22id%22%3A%22http%3A%2F%2Fzotero.org%2Fusers%2F10046823%2Fitems%2FTPWAHTCZ%22%2C%22type%22%3A%22article%22%2C%22abstract%22%3A%22In%20recent%20years%2C%20Transformer%20networks%20are%20beginning%20to%20replace%20pure%20convolutional%20neural%20networks%20(CNNs)%20in%20the%20field%20of%20computer%20vision%20due%20to%20their%20global%20receptive%20field%20and%20adaptability%20to%20input.%20However%2C%20the%20quadratic%20computational%20complexity%20of%20softmax-attention%20limits%20the%20wide%20application%20in%20image%20dehazing%20task%2C%20especially%20for%20high-resolution%20images.%20To%20address%20this%20issue%2C%20we%20propose%20a%20new%20Transformer%20variant%2C%20which%20applies%20the%20Taylor%20expansion%20to%20approximate%20the%20softmax-attention%20and%20achieves%20linear%20computational%20complexity.%20A%20multi-scale%20attention%20refinement%20module%20is%20proposed%20as%20a%20complement%20to%20correct%20the%20error%20of%20the%20Taylor%20expansion.%20Furthermore%2C%20we%20introduce%20a%20multi-branch%20architecture%20with%20multi-scale%20patch%20embedding%20to%20the%20proposed%20Transformer%2C%20which%20embeds%20features%20by%20overlapping%20deformable%20convolution%20of%20different%20scales.%20The%20design%20of%20multi-scale%20patch%20embedding%20is%20based%20on%20three%20key%20ideas%3A%201)%20various%20sizes%20of%20the%20receptive%20field%3B%202)%20multi-level%20semantic%20information%3B%203)%20flexible%20shapes%20of%20the%20receptive%20field.%20Our%20model%2C%20named%20Multi-branch%20Transformer%20expanded%20by%20Taylor%20formula%20(MB-TaylorFormer)%2C%20can%20embed%20coarse%20to%20fine%20features%20more%20flexibly%20at%20the%20patch%20embedding%20stage%20and%20capture%20long-distance%20pixel%20interactions%20with%20limited%20computational%20cost.%20Experimental%20results%20on%20several%20dehazing%20benchmarks%20show%20that%20MB-TaylorFormer%20achieves%20state-of-the-art%20(SOTA)%20performance%20with%20a%20light%20computational%20burden.%20The%20source%20code%20and%20pre-trained%20models%20are%20available%20at%20https%3A%2F%2Fgithub.com%2FFVL2020%2FICCV-2023-MB-TaylorFormer.%22%2C%22note%22%3A%22arXiv%3A2308.14036%20%5Bcs%5D%5Cnversion%3A%202%22%2C%22number%22%3A%22arXiv%3A2308.14036%22%2C%22publisher%22%3A%22arXiv%22%2C%22source%22%3A%22arXiv.org%22%2C%22title%22%3A%22MB-TaylorFormer%3A%20Multi-branch%20Efficient%20Transformer%20Expanded%20by%20Taylor%20Formula%20for%20Image%20Dehazing%22%2C%22title-short%22%3A%22MB-TaylorFormer%22%2C%22URL%22%3A%22http%3A%2F%2Farxiv.org%2Fabs%2F2308.14036%22%2C%22author%22%3A%5B%7B%22family%22%3A%22Qiu%22%2C%22given%22%3A%22Yuwei%22%7D%2C%7B%22family%22%3A%22Zhang%22%2C%22given%22%3A%22Kaihao%22%7D%2C%7B%22family%22%3A%22Wang%22%2C%22given%22%3A%22Chenxi%22%7D%2C%7B%22family%22%3A%22Luo%22%2C%22given%22%3A%22Wenhan%22%7D%2C%7B%22family%22%3A%22Li%22%2C%22given%22%3A%22Hongdong%22%7D%2C%7B%22family%22%3A%22Jin%22%2C%22given%22%3A%22Zhi%22%7D%5D%2C%22accessed%22%3A%7B%22date-parts%22%3A%5B%5B%222023%22%2C10%2C21%5D%5D%7D%2C%22issued%22%3A%7B%22date-parts%22%3A%5B%5B%222023%22%2C8%2C30%5D%5D%7D%7D%7D%5D%2C%22properties%22%3A%7B%7D%7D" ztype="zcitation">(<span class="citation-item"><a href="zotero://select/library/items/TPWAHTCZ">Qiu 等, 2023</a></span>)</span>

Referred in <a href="./学术论文笔记汇总-RYZ5DF37.md" class="internal-link" zhref="zotero://note/u/RYZ5DF37/?ignore=1&#x26;line=-1" ztype="znotelink" class="internal-link">图像去雾</a>

Referred in <a href="./学术论文笔记汇总-RYZ5DF37.md" class="internal-link" zhref="zotero://note/u/RYZ5DF37/?ignore=1&#x26;line=-1" ztype="znotelink" class="internal-link">图像去雾</a>

Referred in <a href="./学术论文笔记汇总-RYZ5DF37.md" class="internal-link" zhref="zotero://note/u/RYZ5DF37/?ignore=1&#x26;line=-1" ztype="znotelink" class="internal-link">Workspace Note</a>

<a href="./学术论文笔记汇总-RYZ5DF37.md" class="internal-link" zhref="zotero://note/u/RYZ5DF37/" ztype="znotelink" class="internal-link">MB-Taylorformer 用于图像去雾的泰勒公式展开多分支高效Transformer</a>

Referred in <a href="./学术论文笔记汇总-RYZ5DF37.md" class="internal-link" zhref="zotero://note/u/RYZ5DF37/?ignore=1&#x26;line=-1" ztype="znotelink" class="internal-link">Workspace Note</a>
