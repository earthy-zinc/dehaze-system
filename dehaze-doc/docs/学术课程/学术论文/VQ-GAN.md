---
tag:
    - 'Computer Science - Computer Vision and Pattern Recognition'
    - VQGAN
title: 'Taming Transformers for High-Resolution Image Synthesis'
category:
    - 图像增强
version: 6177
libraryID: 1
itemKey: NYSYEZU2

---
# VQ-GAN

Comment: Changelog can be found in the supplementary

## 介绍

为了使用Transformer进行图像合成，

## 通过模型学习来建立一个码本

使用Transformer将图像表示为潜在空间的分布问题，这就需要我们将图像语义压缩并通过学习来建立一个语义丰富的码本。

VQGAN在提高压缩率的同时仍然能够保持很好的感知质量。

具体来说是通过一个判别器和感知损失来实现的。

将图像重建损失替换为感知损失，引入基于图块（Transformer）的判别器D来区分真实图像和重建图像。

![\<img alt="" data-attachment-key="SUH7XIL9" width="758" height="60" src="attachments/SUH7XIL9.png" ztype="zimage">](attachments/SUH7XIL9.png)

寻找最优的语义压缩模型的方法就是

![\<img alt="" data-attachment-key="CFCHCAKV" width="747" height="176" src="attachments/CFCHCAKV.png" ztype="zimage">](attachments/CFCHCAKV.png)其中的权重为

![\<img alt="" data-attachment-key="W56WTHYJ" width="364" height="122" src="attachments/W56WTHYJ.png" ztype="zimage">](attachments/W56WTHYJ.png)

L<sub>rec</sub>为感知重建损失

∇GL \[·]为解码器最后一层输入的梯度。

<span class="highlight" data-annotation="%7B%22attachmentURI%22%3A%22http%3A%2F%2Fzotero.org%2Fusers%2F10046823%2Fitems%2FZ57D2RNH%22%2C%22pageLabel%22%3A%224%22%2C%22position%22%3A%7B%22pageIndex%22%3A3%2C%22rects%22%3A%5B%5B378.443%2C707.546%2C418.886%2C718.883%5D%5D%7D%2C%22citationItem%22%3A%7B%22uris%22%3A%5B%22http%3A%2F%2Fzotero.org%2Fusers%2F10046823%2Fitems%2FIL93Q527%22%5D%2C%22locator%22%3A%224%22%7D%7D" ztype="zhighlight"><a href="zotero://open-pdf/library/items/Z57D2RNH?page=4">“δ = 10−6”</a></span> 为了数值稳定。

为了聚合上下文信息。单个注意力层应用在最低分辨率上，减少了code编码展开时的序列长度。

## 使用Transformer来学习图像的组成

在有了编码器和解码器之后，我们可以用码本中编码的下标来来表示图像。一个图像离散后的编码是通过

![\<img alt="" data-attachment-key="N5GJGU97" width="425" height="51" src="attachments/N5GJGU97.png" ztype="zimage">](attachments/N5GJGU97.png)

和码本中一连串的下标

![\<img alt="" data-attachment-key="Q8Y9AHE7" width="354" height="40" src="attachments/Q8Y9AHE7.png" ztype="zimage">](attachments/Q8Y9AHE7.png)

这种下表是通过将码本中的每一个编码code替换为其在码本中的索引（下标）得到的。

![\<img alt="" data-attachment-key="DYN6QMBA" width="495" height="59" src="attachments/DYN6QMBA.png" ztype="zimage">](attachments/DYN6QMBA.png)

我们通过这些下标很容易就能找到码本中对应的编码code。然后将其恢复为编码code并解码为图像。

这些下标其实也是用连续的内存空间存储的，他们在计算机上使用数组去存储的。

Transformer用于下标的分布情况？什么鬼在已知一些下标的时候，并且这些指标的位置都\<i，那么我们可以据此来预测下标si的分布情况。

也就是transformer会学习去根据已知的下标，去预测下一个下标可能的分布情况。

那么他预测的效果如何，通过这样方式来评判他预测效果。也就是直接最大化对数的估计可能性来确定Transformer的估计损失

![\<img alt="" data-attachment-key="J3WVJ7PE" width="552" height="58" src="attachments/J3WVJ7PE.png" ztype="zimage">](attachments/J3WVJ7PE.png)

在许多图像生成任务中，用户需要给出一些额外的信息来控制如何生成图像。这个信息可以是描述这副图像的一个标签，也可以是另一幅图像。这些任务的目标是在给定了信息c后学习下标序列s存在的可能性。

![\<img alt="" data-attachment-key="NAJI3RND" width="412" height="74" src="attachments/NAJI3RND.png" ztype="zimage">](attachments/NAJI3RND.png)

如果这个信息c具有空间范围。我们可以学习另一个VQGAN，用之前获得的码本来重新获得一个索引表示![\<img alt="" data-attachment-key="JN5QIYRB" width="393" height="41" src="attachments/JN5QIYRB.png" ztype="zimage">](attachments/JN5QIYRB.png)。由于Transformer自回归的结构，我们可以将r就看作s，并且计算r存在的概率。

> 也就是说，我们在得到码本之后，之后进行图像重建获得的离散化特征就直接可以看作和码本内编码同等类型的东西。然后从码本中寻找匹配项。

## 生成高分辨率图像
