---
tag: []
title: ""
category:
    - 图像去雾
version: 4964
libraryID: 1
itemKey: RYZ5DF37

---
# 图像去雾文献综述

## 介绍

雾霾是一种常见的图像降质因素，是导致图像模糊的最主要原因之一，受雾霾天气影响，专业的监控和遥感成像系统所拍摄的图像也无法满足相应的工作需求。数字图像质量的恶化会影响各类视觉任务的执行与处理。因此需要对图像进行预处理，以降低雾霾对其成像质量的影响。此外， 雾霾天气下获取的图像，直接影响计算机视觉中的语义分割、目标检测等任务的效果。综上所述，对包含雾或霾影响的图像进行相应的处理十分必要。

目前图像去雾领域中更多的是针对单幅图像去雾的研究，研究方法主要分为三种。基于图像增强的去雾算法、基于物理模型的去雾算法、基于深度学习的去雾算法。

![\<img alt="" data-attachment-key="Z6CAZCPH" width="1082" height="557" src="attachments/Z6CAZCPH.png" ztype="zimage">](attachments/Z6CAZCPH.png)

## 基于图像增强的去雾算法

![\<img alt="" data-attachment-key="X3YCVQ7V" width="1056" height="437" src="attachments/X3YCVQ7V.png" ztype="zimage">](attachments/X3YCVQ7V.png)

基于图像增强的去雾算法主要是通过增强对比度改善图像的视觉效果，但是对于图像突出部分的信息可能会造成一定损失。图像增强分为全局化增强和局部化增强两大类。在全局化增强的方法中，有基于直方图均衡化、基于同态滤波以及基于 Retinex 理论等算法。在局部增强的方法中，主要有对比增强和局部直方图均衡化等算法。

直方图均衡化即对图像的直方图进行均衡化处理，使原图的灰度级分布更加均匀，同时提升图像的对比度和亮度，使图像的感官效果更佳。 同态滤波旨在消除不均匀照度的影响而又 不损失图像细节，在频域中将图像动态范围进行压缩，可以同时增加对比度和亮度，借此达到图像增强的目的。 Retinex 理论以色感一致性（颜色恒常性）为基础，通过增强对比度改善图像视觉效果。不同于传统方法，Retinex 可以兼顾动态范围压缩、 边缘增强和颜色恒常三个方面，因此可以对各种不同类型的图像进行自适应的增强。随着研究的深入，单尺度 Retinex 算法改进成多尺度加权平均的 Retinex 算法，再发展成为带色彩恢复多尺度 Retinex 算法。以上三种算法可以达到一定的除雾效果，但是处理后图像的细节特征仍然不够突出，其根本原因在于图像的最大动态范围未能被充分利用， 对比度没有得到进一步的增强。但这些方法仍然可以作为图像预处理的手段，对图像进行初步的处理。

## 基于物理模型的去雾算法

![\<img alt="" data-attachment-key="7CE996AQ" width="1077" height="398" src="attachments/7CE996AQ.png" ztype="zimage">](attachments/7CE996AQ.png)

早期人们根据经验总结提出了大气光散射模型（Atmospheric Scattering Model，ASM），该模型能够估计图像中雾霾的生成状况。大气光散射现象如图所示。

![\<img alt="" data-attachment-key="HE7L9EGR" width="1545" height="801" src="attachments/HE7L9EGR.png" ztype="zimage">](attachments/HE7L9EGR.png)光线经过一个散射媒介之后，其原方向的光线强度会受到衰减，并且其能量会散发到其他方向。因此，在一个有雾的环境中，相机或者人眼接收到的某个物体（或场景）的光来源于两个部分：

1.  来自于该物体（或场景）本身，这个部分的光强受到散射媒介的影响会有衰弱；
2.  来自大气中其他光源散射至相机或人眼的光强

大气光散射模型可以归纳为以下数学公式：

![\<img alt="" data-attachment-key="YBFY2ZXP" width="882" height="108" src="attachments/YBFY2ZXP.png" ztype="zimage">](attachments/YBFY2ZXP.png)

<span class="highlight" data-annotation="%7B%22attachmentURI%22%3A%22http%3A%2F%2Fzotero.org%2Fusers%2F10046823%2Fitems%2F9SIX2C8F%22%2C%22pageLabel%22%3A%222%22%2C%22position%22%3A%7B%22pageIndex%22%3A1%2C%22rects%22%3A%5B%5B316.163%2C121.908%2C549.817%2C140.986%5D%2C%5B316.163%2C107.004%2C549.817%2C126.082%5D%2C%5B316.163%2C91.798%2C368.045%2C110.876%5D%2C%5B362.881%2C94.355%2C553.292%2C106.452%5D%2C%5B316.163%2C79.442%2C549.817%2C90.95%5D%2C%5B316.163%2C64.528%2C549.817%2C76.036%5D%5D%7D%2C%22citationItem%22%3A%7B%22uris%22%3A%5B%22http%3A%2F%2Fzotero.org%2Fusers%2F10046823%2Fitems%2FSW4N67ZU%22%5D%2C%22locator%22%3A%222%22%7D%7D" ztype="zhighlight"><a href="zotero://open-pdf/library/items/9SIX2C8F?page=2">“其中，I ( x) 是有雾图像，J ( x) 是物体（或场景）的原始辐射 ，A 是全局大气光值，t ( x) 被称作介质透射率且 t ( x ) = e<sup>-βd (x)</sup>，β 为全散射系数，d 为场景深度。同时求解 J (x)，t(x) 和 A 是一个欠适定的问题，往往需要利用各种先验知识来先估计透射图 t (x)，并以此求出其他未知量”</a></span>

<span class="highlight" data-annotation="%7B%22attachmentURI%22%3A%22http%3A%2F%2Fzotero.org%2Fusers%2F10046823%2Fitems%2FJBUG7CXA%22%2C%22pageLabel%22%3A%222%22%2C%22position%22%3A%7B%22pageIndex%22%3A1%2C%22rects%22%3A%5B%5B86.302%2C127.419%2C283.198%2C139.237%5D%2C%5B65.302%2C111.669%2C283.197%2C123.511%5D%2C%5B65.302%2C95.919%2C283.197%2C107.737%5D%2C%5B65.302%2C80.169%2C283.197%2C91.987%5D%2C%5B65.302%2C64.419%2C288.467%2C76.237%5D%5D%7D%2C%22citationItem%22%3A%7B%22uris%22%3A%5B%22http%3A%2F%2Fzotero.org%2Fusers%2F10046823%2Fitems%2FR7FBEADT%22%5D%2C%22locator%22%3A%222%22%7D%7D" ztype="zhighlight"><a href="zotero://open-pdf/library/items/JBUG7CXA?page=2">“基于物理模型的去雾算法常依赖于大气散射模型，通过得到其中的映射关系，根据有雾图像的形成过程来进行逆运算，恢复出清晰图像。此外，大气散射模型是后续众多去雾算法研究的基石，这些算法主要关注于模型中参数的求解，随后推算出无雾图像。”</a></span>

“之后人们对雾霾进行了大量的观测，提出了不少先验知识辅助图像去雾。其中，由何凯明在2010年提出的基于暗通道先验（Dark channel prior，DCP）的去雾算法广为人知。暗通道先验基于这样的前提：在没有雾霾的室外图像中，大多数局部区域包含的一些像素至少在一个颜色通道中具有非常低的强度，基于这个先验知识，可以直接评估出雾霾的厚度，并恢复出高质量的图像。但是该算法的理论前提存在一定的局限性，现实生活中并不是所有场景都能够满足这个条件。当算法在面对类似大面积雪地或天空的图像时，每个颜色通道的强度都相对较高，算法会认为此处雾霾厚度很高，从而过度对该区域进行处理造成失真的效果。 2015 年，Zhu 提出了基于颜色衰减先验的去雾算法。该算法建立在对大量有雾图像的统计分析之上。该算法认为，雾的浓度越高，景深越大，图像的亮度和饱和度相差就越大。利用这一先验并建立模型，我们就得到与有雾图像雾霾浓度及其对应的场景深度信息之间的关系，并利用场景深度信息恢复出无雾的清晰图像。但是在某些场景中，雾霾的浓度和场景深度没有太大的关联，属于非均匀雾霾。在这种情况下，该算法的去雾效果就不够理想。总之，雾霾现象较为复杂，在现实场景中往往表现出不同的特点，人们在某个场景下得出的结论可能不适合另一个场景。因此我们需要一种更加通用的方法。”

## 基于深度学习的去雾方法

![\<img alt="" data-attachment-key="JREHMBJL" width="799" height="412" src="attachments/JREHMBJL.png" ztype="zimage">](attachments/JREHMBJL.png)

<span class="highlight" data-annotation="%7B%22attachmentURI%22%3A%22http%3A%2F%2Fzotero.org%2Fusers%2F10046823%2Fitems%2F9SIX2C8F%22%2C%22pageLabel%22%3A%222%22%2C%22position%22%3A%7B%22pageIndex%22%3A1%2C%22rects%22%3A%5B%5B82.462%2C641.942%2C296.302%2C653.197%5D%2C%5B62.462%2C627.489%2C296.117%2C638.743%5D%2C%5B62.462%2C613.035%2C296.117%2C624.29%5D%2C%5B62.462%2C598.582%2C296.117%2C609.836%5D%2C%5B62.462%2C584.128%2C296.116%2C595.383%5D%2C%5B62.462%2C569.675%2C296.116%2C580.929%5D%2C%5B62.462%2C555.221%2C296.116%2C566.476%5D%2C%5B62.462%2C540.768%2C296.117%2C552.295%5D%2C%5B62.462%2C526.315%2C296.117%2C537.569%5D%2C%5B62.462%2C511.861%2C296.115%2C523.116%5D%2C%5B62.462%2C497.369%2C184.752%2C508.623%5D%5D%7D%2C%22citationItem%22%3A%7B%22uris%22%3A%5B%22http%3A%2F%2Fzotero.org%2Fusers%2F10046823%2Fitems%2FSW4N67ZU%22%5D%2C%22locator%22%3A%222%22%7D%7D" ztype="zhighlight"><a href="zotero://open-pdf/library/items/9SIX2C8F?page=2">“2012 年以来，以卷积神经网络（Convolutional Neural Network，CNN）为代表的深度学习在机器视觉、自然语言处理和语音处理等领域取得了突破性的进展，并逐步被应用在图像去雾领域。大量的研究结果表明，与传统的图像去雾算法相比，基于深度学习的图像去雾算法以大数据为依托，充分利用深度学习强大的特征学习和上下文信息提取能力，从中自动学习蕴含的丰富知识，可以从大数据中自动学习到雾霾图像到清晰图像之间的复杂映射关系，获得了性能上的大幅提升 . 基于深度学习的图像去雾也因此成为图像去雾领域的主流研究方向，取得了重要的进展。”</a></span>

在深度学习去雾方法中，人们也对传统图像去雾算法有或多或少的借鉴。主要是基于物理模型和先验知识，同时利用深度学习的手段，有些会辅以图像增强的预处理操作。许多研究开始采用卷积神经网络来估计大气散射模型中的参数，但是先估计参数，再进行去雾容易产生累积误差。为了避免参数估计过程中的累积误差，人们就提出了端到端的网络，不再分”两步走“，由有雾图像直接估计生成无雾图像。但是，由于在训练过程中往往需要大量无雾图像和有雾图像的样本对进行监督训练，而在实际应用中成对样本往往难以获得，有雾图像往往通过物理模型，主要是大气散射模型对清晰无雾图像处理降质后得到，大气散射模型本身也并不能完美的描述所有雾霾的形成过程，因此这种人为处理得到的图像无法很好代替真实有雾图像，因此训练出来的模型泛化能力差，用于处理真实图像时，往往会失效。因此，这些方法最后在合成数据集上能取得优异的性能，但是在真实数据集上的表现却有待提高。

为此人们开始尝试从两个角度展开研究，一是避开数据本身的不足，引入一部分非成对样本进行半监督学习，或者直接使用非成对样本开展无监督、自监督学习的图像去雾研究，降低对成对样本的依赖。二是将知识蒸馏、元学习、域自适应等机器学习领域的最新研究成果应用于图像去雾中，提升网络的泛化能力，提高实际图像的去雾效果。

因此我们可以大致的将基于深度学习的去雾方法再次划分。可以分为四类。

### 基于物理模型和先验知识的图像去雾

DehazeNet 是基于深度学习的图像去雾方法前驱，网络架构较浅，全局大气光仍然通过传统方法估计，网络包含特征提取层、多尺度映射层、局部极值层以及非线性回归层，通过学习大气光散射模型中的介质透射率 t(x) 进行去雾。计算时，假设所得到的大气光值 A 为一个固定值，与实际大气光值之间会有差异，因而模型求解得到的去雾图像也相应地会产生偏差。为了解决这一问题，Li等人提出了AODNet，将介质透射率 t(x) 和大气光值 A 统一到一个变量 K(x) 中，网络本身需要求解一个 K(x)来实现图像增强。该网络仅包含 5 个卷积层，计算复杂度低，去雾效果有了进一步的提升。

### 基于像素域端到端映射的图像去雾

这一类去雾算法一是借鉴图像分割领域常用的编码器到解码器结构对图像中的信息进行挖掘，结合注意力机制、特征融合等策略，提升特征的表达能力。从而帮助图像去雾。二是从自然语言领域的Transformer获得灵感，进而构建了适用于图像领域的Vision Transformer用于图像去雾，也取得了不错的性能。

#### 编码器-解码器架构

编码器用于对输入图像进行特征提取，而解码器则利用编码器得到的特征重构目标图像。图像不同层级之间提取的特征种类、特征图的感受野，特征的细腻程度往往不同，为了充分利用图像各层级之间的特征，该类网络常常再编码器和解码器之间添加跳跃连接。在网络内部，通常会结合注意力机制、特征融合等策略来提升特征的表达能力。与普通的CNN网络不同，编码器解码器架构能够更好的进行特征提取和表达，有效提升网络利用率，在图像去雾领域得到了广泛应用。

生成对抗网络（Generative Adversarial Network， GAN）是 2014 年 Goodfellow 等人提出的一种网络结构，包含生成器和判别器，生成器用于获得真实数据样本的特征分布，并且据此生成新的数据样本。判别器是一个二分类器，用于判别输入的是真实数据还是生成的样本。生成对抗网络主要是用于解决图像分类和识别任务中的数据集样本扩充、图像风格迁移和图像增强等问题。在图像去雾领域中，生成器主要用来获取有雾图像的雾霾特征，除去雾霾并生成干净无雾的图像。判别器则将生成的无雾图像和真实无雾图像进行比较，判断生成图像的质量并指导生成器迭代优化。两者通过循环交替达到纳什均衡，从而训练出最优的网络模型。很多去雾方法都是根据生成对抗网络的原理来指导模型训练。

近些年来有很多学者在这方面做了不少工作。Qu提出将图像去雾问题简化为图像到图像的转换问题，在不依赖于大气散射模型的情况下生成无雾图像。<span class="highlight" data-annotation="%7B%22attachmentURI%22%3A%22http%3A%2F%2Fzotero.org%2Fusers%2F10046823%2Fitems%2F9SIX2C8F%22%2C%22pageLabel%22%3A%225%22%2C%22position%22%3A%7B%22pageIndex%22%3A4%2C%22rects%22%3A%5B%5B189.548%2C483.161%2C296.116%2C494.416%5D%2C%5B62.462%2C468.704%2C296.117%2C479.958%5D%2C%5B62.462%2C454.246%2C296.117%2C465.5%5D%2C%5B62.462%2C439.788%2C296.116%2C451.043%5D%2C%5B62.462%2C425.331%2C296.116%2C436.585%5D%2C%5B62.462%2C410.873%2C296.116%2C422.128%5D%2C%5B62.462%2C396.416%2C191.16%2C407.67%5D%5D%7D%2C%22citationItem%22%3A%7B%22uris%22%3A%5B%22http%3A%2F%2Fzotero.org%2Fusers%2F10046823%2Fitems%2FSW4N67ZU%22%5D%2C%22locator%22%3A%225%22%7D%7D" ztype="zhighlight"><a href="zotero://open-pdf/library/items/9SIX2C8F?page=5">“EPDN 由多分辨率生成器模块、增强器模块和多尺度判别器模块组成。多分辨率生成器对雾霾图像在两个尺度上进行特征提取；增强模块用于恢复去雾图像的颜色和细节信息；多尺度判别器用于对生成的去雾结果进行鉴别。虽然算法在主客观结果上都有了一定提升，但是对真实雾霾图像进行处理时，会存在过增强现象。”</a></span> <span class="citation" data-citation="%7B%22citationItems%22%3A%5B%7B%22uris%22%3A%5B%22http%3A%2F%2Fzotero.org%2Fusers%2F10046823%2Fitems%2FSW4N67ZU%22%5D%2C%22locator%22%3A%225%22%7D%5D%2C%22properties%22%3A%7B%7D%7D" ztype="zcitation">(<span class="citation-item"><a href="zotero://select/library/items/SW4N67ZU">贾童瑶 等, 2023, p. 5</a></span>)</span><span class="highlight" data-annotation="%7B%22attachmentURI%22%3A%22http%3A%2F%2Fzotero.org%2Fusers%2F10046823%2Fitems%2F9SIX2C8F%22%2C%22pageLabel%22%3A%225%22%2C%22position%22%3A%7B%22pageIndex%22%3A4%2C%22rects%22%3A%5B%5B136.938%2C324.128%2C296.285%2C336.082%5D%2C%5B62.462%2C309.67%2C296.116%2C320.925%5D%2C%5B62.462%2C295.213%2C296.116%2C306.467%5D%2C%5B62.462%2C280.755%2C211.16%2C292.01%5D%5D%7D%2C%22citationItem%22%3A%7B%22uris%22%3A%5B%22http%3A%2F%2Fzotero.org%2Fusers%2F10046823%2Fitems%2FSW4N67ZU%22%5D%2C%22locator%22%3A%225%22%7D%7D" ztype="zhighlight"><a href="zotero://open-pdf/library/items/9SIX2C8F?page=5">“Liu 等人提出了 GridDehazeNet 网络结构，通过独特的网格式结构，并利用网络注意力机制进行多尺度特征融合，充分融合底层和高层特征，网络取得了较好的映射能力。”</a></span> <span class="citation" data-citation="%7B%22citationItems%22%3A%5B%7B%22uris%22%3A%5B%22http%3A%2F%2Fzotero.org%2Fusers%2F10046823%2Fitems%2FSW4N67ZU%22%5D%2C%22locator%22%3A%225%22%7D%5D%2C%22properties%22%3A%7B%7D%7D" ztype="zcitation">(<span class="citation-item"><a href="zotero://select/library/items/SW4N67ZU">贾童瑶 等, 2023, p. 5</a></span>)</span><span class="highlight" data-annotation="%7B%22attachmentURI%22%3A%22http%3A%2F%2Fzotero.org%2Fusers%2F10046823%2Fitems%2F9SIX2C8F%22%2C%22pageLabel%22%3A%225%22%2C%22position%22%3A%7B%22pageIndex%22%3A4%2C%22rects%22%3A%5B%5B82.462%2C266.298%2C296.117%2C278.252%5D%2C%5B62.462%2C251.84%2C296.195%2C263.368%5D%2C%5B62.462%2C237.383%2C296.116%2C248.637%5D%2C%5B62.462%2C222.925%2C296.117%2C234.452%5D%2C%5B62.462%2C208.468%2C296.116%2C219.722%5D%2C%5B62.462%2C194.01%2C296.116%2C205.265%5D%2C%5B62.462%2C179.553%2C301.136%2C190.807%5D%2C%5B62.462%2C165.095%2C301.136%2C176.35%5D%2C%5B62.462%2C150.638%2C236.159%2C161.892%5D%5D%7D%2C%22citationItem%22%3A%7B%22uris%22%3A%5B%22http%3A%2F%2Fzotero.org%2Fusers%2F10046823%2Fitems%2FSW4N67ZU%22%5D%2C%22locator%22%3A%225%22%7D%7D" ztype="zhighlight"><a href="zotero://open-pdf/library/items/9SIX2C8F?page=5">“Dong 等人提出了一种基于 U-Net 架构的具有密集特征融合的多尺度特征增强（Multi-Scale Boosted Dehazing Network，MSBDN），通过一个增强解码器来逐步恢复无雾霾图像 . 为了解决在 U-Net 架构中保留空间信息的问题，他们设计了一个使用反投影反馈方案的密集特征融合模块。结果表明，密集特征融合模块可以同时弥补高分辨率特征中缺失的空间信息， 并利用非相邻特征。但是算法的模型复杂、参数量大，而且在下采样过程中容易丢失细节信息。”</a></span> <span class="citation" data-citation="%7B%22citationItems%22%3A%5B%7B%22uris%22%3A%5B%22http%3A%2F%2Fzotero.org%2Fusers%2F10046823%2Fitems%2FSW4N67ZU%22%5D%2C%22locator%22%3A%225%22%7D%5D%2C%22properties%22%3A%7B%7D%7D" ztype="zcitation">(<span class="citation-item"><a href="zotero://select/library/items/SW4N67ZU">贾童瑶 等, 2023, p. 5</a></span>)</span><span class="highlight" data-annotation="%7B%22attachmentURI%22%3A%22http%3A%2F%2Fzotero.org%2Fusers%2F10046823%2Fitems%2F9SIX2C8F%22%2C%22pageLabel%22%3A%225%22%2C%22position%22%3A%7B%22pageIndex%22%3A4%2C%22rects%22%3A%5B%5B82.462%2C136.18%2C296.116%2C148.135%5D%2C%5B62.462%2C121.723%2C301.136%2C132.977%5D%2C%5B62.462%2C107.265%2C296.116%2C118.792%5D%2C%5B62.462%2C92.808%2C296.116%2C104.062%5D%2C%5B62.462%2C78.35%2C296.116%2C89.605%5D%2C%5B62.462%2C63.893%2C282.458%2C75.147%5D%5D%7D%2C%22citationItem%22%3A%7B%22uris%22%3A%5B%22http%3A%2F%2Fzotero.org%2Fusers%2F10046823%2Fitems%2FSW4N67ZU%22%5D%2C%22locator%22%3A%225%22%7D%7D" ztype="zhighlight"><a href="zotero://open-pdf/library/items/9SIX2C8F?page=5">“Qin 等人去除了上下采样操作，提出了一种端到端特征融合注意网络（Feature Fusion Attention Network， FFA-Net）来直接恢复无雾霾图像 . 该方法的主要思想是自适应地学习特征权重，给重要特征赋予更多的权重 . 在每一个残差块后加入特征注意力，并且对各个组的特征进行加权自适应选择，提升网络的映射能力 .”</a></span> <span class="citation" data-citation="%7B%22citationItems%22%3A%5B%7B%22uris%22%3A%5B%22http%3A%2F%2Fzotero.org%2Fusers%2F10046823%2Fitems%2FSW4N67ZU%22%5D%2C%22locator%22%3A%225%22%7D%5D%2C%22properties%22%3A%7B%7D%7D" ztype="zcitation">(<span class="citation-item"><a href="zotero://select/library/items/SW4N67ZU">贾童瑶 等, 2023, p. 5</a></span>)</span>

在图像复原领域，有学者将生成对抗网络进行扩展，提出了扩散模型，在图像去雾、去噪、去雨等任务表现出色。其工作原理主要是通过前向扩散过程和反向采样过程实现的。具体来说，扩散模型在前向扩散过程中对图像逐步施加噪声，直至图像被破坏变成完全的高斯噪声，这种噪声通常是可逆的，同时图像中还保留有图像原本的特征。然后，在反向采样过程中，模型学习如何从高斯噪声还原出真实图像。但是这类模型往往有这样几个缺点。一是<span style="color: rgb(44, 62, 80)"><span style="background-color: rgb(255, 255, 255)">依赖对数据集的规划，二是要求图像退化参数已知。</span></span>

#### Transformer架构

Transformer最初是针对自然语言处理任务提出的，通过多头自注意力机制和前馈层的堆叠，捕获单词之间的非局部交互。“Doso⁃vitski 等人提出了用于图像领域的Vision Transformer 模型（Vision Transformer，ViT），展示了其在图像处理领域应用的潜力。

### 基于非成对低质量到高质量样本对的图像去雾

### 域知识迁移学习

## 数据集

## 评价指标

## 思考和展望

目前图像去雾领域存在的困难主要有以下几点。

1.  网络的训练需要大量的无雾-有雾图像对作为支撑，但是实际中这样的数据集获取困难。目前的做法是通过一些物理模型如大气散射模型，将高质量无雾图像处理得到低质量的有雾图像，形成合成数据集。但是这一类模型往往无法很好地模拟真实图像降质的过程。
2.  网络的泛化能力差，主要表现在一个数据集上训练的模型应用到另一个数据集上效果往往不佳。这是因为样本分布不一致，根本原因在于雾霾变化多样，在某些场景中得到的雾霾特征往往不适用于另一个场景。

为了解决以上两大难题，我们可以从这些角度出发。

## <span style="background-color: rgb(255, 255, 255)">创新方向</span>

### <span style="background-color: rgb(255, 255, 255)">数据预处理</span>

针对网络的训练需要大量数据作为支撑，但是目前数据集数据量有限，我们就需要考虑数据预处理。

1.  **组合多个数据集并降低差异**，

    <span style="color: rgb(44, 62, 80)"><span style="background-color: rgb(255, 255, 255)">由于每个数据集之间有一些细微的差异，例如颜色差异、物体复杂性、拍摄所用的相机差别等。直接组合会降低去雾结果指标，因此我们设计一种数据预处理技术，来减少数据集之间的分布差距。</span></span>

    <span class="citation" data-citation="%7B%22citationItems%22%3A%5B%7B%22uris%22%3A%5B%22http%3A%2F%2Fzotero.org%2Fusers%2F10046823%2Fitems%2FKA6CRQKY%22%5D%7D%5D%2C%22properties%22%3A%7B%7D%7D" ztype="zcitation">(<span class="citation-item"><a href="zotero://select/library/items/KA6CRQKY">Liu 等, 2023</a></span>)</span>

    提出了一种新的预处理技术，对数据集之间明显的颜色差异进行校正，

    <span style="color: rgb(44, 62, 80)"><span style="background-color: rgb(255, 255, 255)">并且将增强后的数据分布转移到目标数据分布。显著的降低了数据集之间的差异，增加了数据量，从而提高了去雾效果。</span></span>

2.  **缓解合成数据集和真实数据集之间的差距。**

    <span class="citation" data-citation="%7B%22citationItems%22%3A%5B%7B%22uris%22%3A%5B%22http%3A%2F%2Fzotero.org%2Fusers%2F10046823%2Fitems%2FBJNHXL93%22%5D%7D%5D%2C%22properties%22%3A%7B%7D%7D" ztype="zcitation">(<span class="citation-item"><a href="zotero://select/library/items/BJNHXL93">Wu 等, 2023</a></span>)</span>

    重新设计了合成数据集的生成过程，

    <span style="color: rgb(44, 62, 80)"><span style="background-color: rgb(255, 255, 255)">考虑到了图像退化的各种因素。由此得到的数据集缓解合成数据和真实数据之间的差距。</span></span>

3.  应用图像增强的方法，如直方图均衡化、对比度提升初步处理数据。

### <span style="background-color: rgb(255, 255, 255)">模型结构</span>

**引入多分支及分类。**针对雾霾变化多样，在某些场景中得到的雾霾特征往往不适用于另一个场景这一特点。我们可以针对不同的雾霾具体使用不同的网络进行去雾。对当前去雾数据集分析可以得知，均匀雾霾、非均匀雾霾有一定的区别。<span style="color: rgb(44, 62, 80)"><span style="background-color: rgb(255, 255, 255)">非均匀雾霾并不完全由图像场景深度决定，不同区域的雾霾浓度往往不一。因此通常的去雾方法在去除非均匀雾霾上效果不佳。我们可以根据不同种类的雾霾，引入不同的编码器，提取不同类别雾霾之间特异的特征，形成一种多分支结构。随后再接上普通编码器，提取雾霾大类之间的相同特征。随后送去解码器输出去雾图像。</span></span>

**引入高质量先验。**从去雾网络发展历程可以得知，高质量先验知识对去雾网络的设计有很大的帮助，以往的网络模型都是人工通过经验总结得到的先验知识，然后据此设计网络，如暗通道先验和颜色衰减先验，由于先验知识本身具有一定的局限性，从而导致设计出来的网络泛化能力不佳。因此<span class="citation" data-citation="%7B%22citationItems%22%3A%5B%7B%22uris%22%3A%5B%22http%3A%2F%2Fzotero.org%2Fusers%2F10046823%2Fitems%2FBJNHXL93%22%5D%7D%5D%2C%22properties%22%3A%7B%7D%7D" ztype="zcitation">(<span class="citation-item"><a href="zotero://select/library/items/BJNHXL93">Wu 等, 2023</a></span>)</span>通过预训练一个网络来得到高质量的先验知识，然后将高质量先验与网络进行融合训练，再通过解码器输出无雾图像。

**引入选择机制。**图像中并不是所有区域都是同等重要的，比如天空、雪地等区域去雾重要性不高，而其他雾霾浓度高，距离近的区域则相比于雾霾少距离远的物体去雾重要性更强。<span class="citation" data-citation="%7B%22citationItems%22%3A%5B%7B%22uris%22%3A%5B%22http%3A%2F%2Fzotero.org%2Fusers%2F10046823%2Fitems%2F4SHMI7H5%22%5D%7D%5D%2C%22properties%22%3A%7B%7D%7D" ztype="zcitation">(<span class="citation-item"><a href="zotero://select/library/items/4SHMI7H5">Cui 等, 2023</a></span>)</span>提出了双域选择机制，双域主要表现在<span style="color: rgb(44, 62, 80)"><span style="background-color: rgb(255, 255, 255)">空间选择模块和频率选择模块。空间选择模块通过深度卷积层来确定每个通道中图像退化区域的大致位置。然后利用频率选择模块对高频信号或硬区域进行放大，去除特征中的低频成分。通过这种机制，模型会更专注于雾霾更重的关键区域。</span></span>

学习雾霾和图像背景之间交互的特征。<span class="citation" data-citation="%7B%22citationItems%22%3A%5B%7B%22uris%22%3A%5B%22http%3A%2F%2Fzotero.org%2Fusers%2F10046823%2Fitems%2F8GJPSKZ7%22%5D%7D%5D%2C%22properties%22%3A%7B%7D%7D" ztype="zcitation">(<span class="citation-item"><a href="zotero://select/library/items/8GJPSKZ7">Guo 等, 2023</a></span>)</span>通过注意力生成和场景重建网络<span style="color: rgb(44, 62, 80)"><span style="background-color: rgb(255, 255, 255)">专注于学习非均匀雾霾和场景之间的复杂交互特征。</span></span>

提取多尺度特征。使用不同大小的卷积核、多尺度、并行地提取特征。

### <span style="background-color: rgb(255, 255, 255)">损失函数</span>

1.  采用多个损失函数结合来辅助模型训练，如感知损失、对抗损失、结构相似性损失

### <span style="background-color: rgb(255, 255, 255)">正则化方法</span>

1.  对比正则化方法

### 训练策略

1.  **对比学习。**2023年的CVPR中

    <span class="citation" data-citation="%7B%22citationItems%22%3A%5B%7B%22uris%22%3A%5B%22http%3A%2F%2Fzotero.org%2Fusers%2F10046823%2Fitems%2FYMAP3M6X%22%5D%7D%5D%2C%22properties%22%3A%7B%7D%7D" ztype="zcitation">(<span class="citation-item"><a href="zotero://select/library/items/YMAP3M6X">Zheng 等, 2023</a></span>)</span>

    提出了这一观点，在训练的过程中不断和其他模型训练得到的去雾图像做对比，一开始和效果较差的比较并进行自我矫正，逐步和效果好的比较直到接近真实无雾图像。在学习的过程中会逐步结合各种去雾方法的优点、不断地调整来达到更好的结果。因此我们可以精挑细选近些年去雾效果较好的、去雾方法差别较大的作为比较对象进行对比学习。

2.  采用

    <span style="color: rgb(44, 62, 80)"><span style="background-color: rgb(255, 255, 255)">YCbCr颜色空间，</span></span>

    <span class="citation" data-citation="%7B%22citationItems%22%3A%5B%7B%22uris%22%3A%5B%22http%3A%2F%2Fzotero.org%2Fusers%2F10046823%2Fitems%2F8GJPSKZ7%22%5D%7D%5D%2C%22properties%22%3A%7B%7D%7D" ztype="zcitation">(<span class="citation-item"><a href="zotero://select/library/items/8GJPSKZ7">Guo 等, 2023</a></span>)</span>

    、

    ()

3.  课程学习。

    <span class="citation" data-citation="%7B%22citationItems%22%3A%5B%7B%22uris%22%3A%5B%22http%3A%2F%2Fzotero.org%2Fusers%2F10046823%2Fitems%2F8GJPSKZ7%22%5D%7D%5D%2C%22properties%22%3A%7B%7D%7D" ztype="zcitation">(<span class="citation-item"><a href="zotero://select/library/items/8GJPSKZ7">Guo 等, 2023</a></span>)</span>

    在其论文中

    <span style="color: rgb(44, 62, 80)"><span style="background-color: rgb(255, 255, 255)">为了增强亮度差异大的区域去雾效果，引入了自组织半课程学习的注意力图生成策略，该方法加快了模型参数收敛。减少了训练早期多目标预测引起的学习歧义。</span></span>

### 轻量化

随着深度学习的发展，模型逐渐往复杂化、巨大化发展，所需的计算量更是水涨船高，这对去雾系统的实时应用不利，因此，有必要对模型进行效果和运行速度的权衡，节约计算开销。

1.  采用四元数网络。

    <span class="citation" data-citation="%7B%22citationItems%22%3A%5B%7B%22uris%22%3A%5B%22http%3A%2F%2Fzotero.org%2Fusers%2F10046823%2Fitems%2F6ELGGPS7%22%5D%7D%5D%2C%22properties%22%3A%7B%7D%7D" ztype="zcitation">(<span class="citation-item"><a href="zotero://select/library/items/6ELGGPS7">Frants 等, 2023</a></span>)</span>

    在其设计的网络QCNN-H中指出采用四元数的卷积神经网络

    <span style="color: rgb(44, 62, 80)"><span style="background-color: rgb(255, 255, 255)">更好的保留了颜色信息，同时减小了网络参数、节约了内存资源</span></span>

## 常用模块
