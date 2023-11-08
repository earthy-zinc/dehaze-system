---
tag: []
title: ""
category:
    - 图像去雾
version: 5065
libraryID: 1
itemKey: RYZ5DF37

---
# 图像去雾文献综述

## 介绍

雾霾是一种常见的图像降质因素，是导致图像模糊的最主要原因之一，受雾霾天气影响，专业的监控和遥感成像系统所拍摄的图像也无法满足相应的工作需求。数字图像质量的恶化会影响各类视觉任务的执行与处理。因此需要对图像进行预处理，以降低雾霾对其成像质量的影响。此外， 雾霾天气下获取的图像，直接影响计算机视觉中的语义分割、目标检测等任务的效果。综上所述，对包含雾或霾影响的图像进行相应的处理十分必要。

目前图像去雾领域中更多的是针对单幅图像去雾的研究，研究方法主要分为三种。基于图像增强的去雾算法、基于物理模型的去雾算法、基于深度学习的去雾算法。

![\<img alt="" data-attachment-key="Z6CAZCPH" src="attachments/Z6CAZCPH.png" ztype="zimage">](attachments/Z6CAZCPH.png)

## 基于图像增强的去雾算法

![\<img alt="" data-attachment-key="X3YCVQ7V" src="attachments/X3YCVQ7V.png" ztype="zimage">](attachments/X3YCVQ7V.png)

基于图像增强的去雾算法主要是通过增强对比度改善图像的视觉效果，但是对于图像突出部分的信息可能会造成一定损失。图像增强分为全局化增强和局部化增强两大类。在全局化增强的方法中，有基于直方图均衡化、基于同态滤波以及基于 Retinex 理论等算法。在局部增强的方法中，主要有对比增强和局部直方图均衡化等算法。

直方图均衡化即对图像的直方图进行均衡化处理，使原图的灰度级分布更加均匀，同时提升图像的对比度和亮度，使图像的感官效果更佳。 同态滤波旨在消除不均匀照度的影响而又 不损失图像细节，在频域中将图像动态范围进行压缩，可以同时增加对比度和亮度，借此达到图像增强的目的。 Retinex 理论以色感一致性（颜色恒常性）为基础，通过增强对比度改善图像视觉效果。不同于传统方法，Retinex 可以兼顾动态范围压缩、 边缘增强和颜色恒常三个方面，因此可以对各种不同类型的图像进行自适应的增强。随着研究的深入，单尺度 Retinex 算法改进成多尺度加权平均的 Retinex 算法，再发展成为带色彩恢复多尺度 Retinex 算法。以上三种算法可以达到一定的除雾效果，但是处理后图像的细节特征仍然不够突出，其根本原因在于图像的最大动态范围未能被充分利用， 对比度没有得到进一步的增强。但这些方法仍然可以作为图像预处理的手段，对图像进行初步的处理。

## 基于物理模型的去雾算法

![\<img alt="" data-attachment-key="7CE996AQ" src="attachments/7CE996AQ.png" ztype="zimage">](attachments/7CE996AQ.png)

早期人们根据经验总结提出了大气光散射模型（Atmospheric Scattering Model，ASM），该模型能够估计图像中雾霾的生成状况。大气光散射现象如图所示。

![\<img alt="" data-attachment-key="HE7L9EGR" src="attachments/HE7L9EGR.png" ztype="zimage">](attachments/HE7L9EGR.png)光线经过一个散射媒介之后，其原方向的光线强度会受到衰减，并且其能量会散发到其他方向。因此，在一个有雾的环境中，相机或者人眼接收到的某个物体（或场景）的光来源于两个部分：

1.  来自于该物体（或场景）本身，这个部分的光强受到散射媒介的影响会有衰弱；
2.  来自大气中其他光源散射至相机或人眼的光强

大气光散射模型可以归纳为以下数学公式：

![\<img alt="" data-attachment-key="YBFY2ZXP" src="attachments/YBFY2ZXP.png" ztype="zimage">](attachments/YBFY2ZXP.png)

<span class="highlight" data-annotation="%7B%22attachmentURI%22%3A%22http%3A%2F%2Fzotero.org%2Fusers%2F10046823%2Fitems%2F9SIX2C8F%22%2C%22pageLabel%22%3A%222%22%2C%22position%22%3A%7B%22pageIndex%22%3A1%2C%22rects%22%3A%5B%5B316.163%2C121.908%2C549.817%2C140.986%5D%2C%5B316.163%2C107.004%2C549.817%2C126.082%5D%2C%5B316.163%2C91.798%2C368.045%2C110.876%5D%2C%5B362.881%2C94.355%2C553.292%2C106.452%5D%2C%5B316.163%2C79.442%2C549.817%2C90.95%5D%2C%5B316.163%2C64.528%2C549.817%2C76.036%5D%5D%7D%2C%22citationItem%22%3A%7B%22uris%22%3A%5B%22http%3A%2F%2Fzotero.org%2Fusers%2F10046823%2Fitems%2FSW4N67ZU%22%5D%2C%22locator%22%3A%222%22%7D%7D" ztype="zhighlight"><a href="zotero://open-pdf/library/items/9SIX2C8F?page=2">“其中，I ( x) 是有雾图像，J ( x) 是物体（或场景）的原始辐射 ，A 是全局大气光值，t ( x) 被称作介质透射率且 t ( x ) = e<sup>-βd (x)</sup>，β 为全散射系数，d 为场景深度。同时求解 J (x)，t(x) 和 A 是一个欠适定的问题，往往需要利用各种先验知识来先估计透射图 t (x)，并以此求出其他未知量”</a></span>

<span class="highlight" data-annotation="%7B%22attachmentURI%22%3A%22http%3A%2F%2Fzotero.org%2Fusers%2F10046823%2Fitems%2FJBUG7CXA%22%2C%22pageLabel%22%3A%222%22%2C%22position%22%3A%7B%22pageIndex%22%3A1%2C%22rects%22%3A%5B%5B86.302%2C127.419%2C283.198%2C139.237%5D%2C%5B65.302%2C111.669%2C283.197%2C123.511%5D%2C%5B65.302%2C95.919%2C283.197%2C107.737%5D%2C%5B65.302%2C80.169%2C283.197%2C91.987%5D%2C%5B65.302%2C64.419%2C288.467%2C76.237%5D%5D%7D%2C%22citationItem%22%3A%7B%22uris%22%3A%5B%22http%3A%2F%2Fzotero.org%2Fusers%2F10046823%2Fitems%2FR7FBEADT%22%5D%2C%22locator%22%3A%222%22%7D%7D" ztype="zhighlight"><a href="zotero://open-pdf/library/items/JBUG7CXA?page=2">“基于物理模型的去雾算法常依赖于大气散射模型，通过得到其中的映射关系，根据有雾图像的形成过程来进行逆运算，恢复出清晰图像。此外，大气散射模型是后续众多去雾算法研究的基石，这些算法主要关注于模型中参数的求解，随后推算出无雾图像。”</a></span>

“之后人们对雾霾进行了大量的观测，提出了不少先验知识辅助图像去雾。其中，由何凯明在2010年提出的基于暗通道先验（Dark channel prior，DCP）的去雾算法广为人知。暗通道先验基于这样的前提：在没有雾霾的室外图像中，大多数局部区域包含的一些像素至少在一个颜色通道中具有非常低的强度，基于这个先验知识，可以直接评估出雾霾的厚度，并恢复出高质量的图像。但是该算法的理论前提存在一定的局限性，现实生活中并不是所有场景都能够满足这个条件。当算法在面对类似大面积雪地或天空的图像时，每个颜色通道的强度都相对较高，算法会认为此处雾霾厚度很高，从而过度对该区域进行处理造成失真的效果。 2015 年，Zhu 提出了基于颜色衰减先验的去雾算法。该算法建立在对大量有雾图像的统计分析之上。该算法认为，雾的浓度越高，景深越大，图像的亮度和饱和度相差就越大。利用这一先验并建立模型，我们就得到与有雾图像雾霾浓度及其对应的场景深度信息之间的关系，并利用场景深度信息恢复出无雾的清晰图像。但是在某些场景中，雾霾的浓度和场景深度没有太大的关联，属于非均匀雾霾。在这种情况下，该算法的去雾效果就不够理想。总之，雾霾现象较为复杂，在现实场景中往往表现出不同的特点，人们在某个场景下得出的结论可能不适合另一个场景。因此我们需要一种更加通用的方法。”

## 基于深度学习的去雾方法

![\<img alt="" data-attachment-key="JREHMBJL" src="attachments/JREHMBJL.png" ztype="zimage">](attachments/JREHMBJL.png)

2012 年以来，以卷积神经网络（Convolutional Neural Network，CNN）为代表的深度学习在机器视觉、自然语言处理和语音处理等领域取得了突破性的进展，并逐步被应用在图像去雾领域。大量的研究结果表明，与传统的图像去雾算法相比，基于深度学习的图像去雾算法以大数据为依托，充分利用深度学习强大的特征学习和上下文信息提取能力，从中自动学习蕴含的丰富知识，可以从大数据中自动学习到雾霾图像到清晰图像之间的复杂映射关系，获得了性能上的大幅提升 . 基于深度学习的图像去雾也因此成为图像去雾领域的主流研究方向，取得了重要的进展。

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

近些年来有很多学者在这方面做了不少工作。Qu提出将图像去雾问题简化为图像到图像的转换问题，在不依赖于大气散射模型的情况下生成无雾图像。EPDN 由多分辨率生成器模块、增强器模块和多尺度判别器模块组成。多分辨率生成器对雾霾图像在两个尺度上进行特征提取；增强模块用于恢复去雾图像的颜色和细节信息；多尺度判别器用于对生成的去雾结果进行鉴别。虽然算法在主客观结果上都有了一定提升，但是对真实雾霾图像进行处理时，会存在过增强现象。Liu 等人提出了 GridDehazeNet 网络结构，通过独特的网格式结构，并利用网络注意力机制进行多尺度特征融合，充分融合底层和高层特征，网络取得了较好的映射能力。Dong 等人提出了一种基于 U-Net 架构的具有密集特征融合的多尺度特征增强（Multi-Scale Boosted Dehazing Network，MSBDN），通过一个增强解码器来逐步恢复无雾霾图像 . 为了解决在 U-Net 架构中保留空间信息的问题，他们设计了一个使用反投影反馈方案的密集特征融合模块。结果表明，密集特征融合模块可以同时弥补高分辨率特征中缺失的空间信息， 并利用非相邻特征。但是算法的模型复杂、参数量大，而且在下采样过程中容易丢失细节信息。Qin 等人去除了上下采样操作，提出了一种端到端特征融合注意网络（Feature Fusion Attention Network， FFA-Net）来直接恢复无雾霾图像 . 该方法的主要思想是自适应地学习特征权重，给重要特征赋予更多的权重 . 在每一个残差块后加入特征注意力，并且对各个组的特征进行加权自适应选择，提升网络的映射能力.

在图像复原领域，有学者将生成对抗网络进行扩展，提出了扩散模型，在图像去雾、去噪、去雨等任务表现出色。其工作原理主要是通过前向扩散过程和反向采样过程实现的。具体来说，扩散模型在前向扩散过程中对图像逐步施加噪声，直至图像被破坏变成完全的高斯噪声，这种噪声通常是可逆的，同时图像中还保留有图像原本的特征。然后，在反向采样过程中，模型学习如何从高斯噪声还原出真实图像。但是这类模型往往有这样几个缺点。一是依赖对数据集的规划，二是要求图像退化参数已知。

#### Transformer架构

Transformer最初是针对自然语言处理任务提出的，通过多头自注意力机制和前馈层的堆叠，捕获单词之间的非局部交互。“Doso⁃vitski 等人提出了用于图像领域的Vision Transformer 模型（Vision Transformer，ViT），展示了其在图像处理领域应用的潜力。

### 基于非成对低质量到高质量样本对的图像去雾

#### 无监督学习

Zhu提出的循环一致对抗网络（Cycle-consistent Adversarial Networks，CycleGAN）是一种比较具有代表性的基于非成对样本的网络结构，该网络是面向图像风格迁移任务设计的。整体架构包含了两个生成器和两个判别器。一个生成器负责将 X 域图像映射到 Y 域，另一个生成器负责将 Y 域图像映射到 X 域；判别器用于判断输入图像是否属于 X 域。假设存在一对非成对样本{xi ,yi} ,xi ∈ X,yi ∈ Y，以正向训练为例，xi 用于训练 Dx，标签为真，G(xi )用于训练 Dy，标签为假，此时判别器 Dy 可以监督生成器 G 的训练； 通过优化输入 xi 与 F(G(xi ))之间进行 L1 范式损失，可以 同时监督生成器 G 和 F 的训练 . 这个损失称为循环一致性损失；反向训练时同理 . 通过正反向交替训练可以达到训练生成器 G 和 F 的目的。

#### 自监督学习

### 基于域知识迁移学习的图像去雾

## 数据集

### RESIDE

发布时间：2019

网址：[RESIDE: V0 (google.com)](https://sites.google.com/view/reside-dehaze-datasets/reside-v0)

论文地址：<https://arxiv.org/abs/1712.04143>

该数据集使用由合成和真实世界有雾图像组成的新的大规模基准数据，称为真实单图像去雾 REalistic Single Image DEhazing (RESIDE)，用于训练、评估和比较单幅图像去雾算法。 RESIDE的一个显着特征在于其评估标准的多样性，从传统的完整参考指标到更实用的无参考指标，再到所需的人类主观评估和任务驱动评估。RESIDE 根据不同的数据源和图像内容，分为五个子集，每个子集有不同的目的（训练或评估）或来源（室内或室外）。

各子集图片示例：

*   ITS (Indoor Training Set) 室内训练集
*   OTS (Outdoor Training Set) 室外训练集
*   SOTS (Synthetic Objective Testing Set) 合成目标测试集
*   RTTS (Real-world Task-Driven Testing Set)
*   HSTS (Hybrid Subjective Testing Set)
*   Unannotated Real-world Hazy Images（不包含在上述子集中）

### RESIDE-Standard（RESIDE-IN）

网址：[RESIDE-Standard (google.com)](https://sites.google.com/view/reside-dehaze-datasets/reside-standard)

训练集包含13,990个合成有雾图像，使用来自现有室内深度数据集NYU2和米德尔伯里立体数据库的1,399个清晰图像生成。其中每个清晰无雾图像合成10个有雾图像，13,000个用于训练和990个用于验证。 图片每个通道大气光A在\[0.7，1.0]之间，均匀地随机选择β在\[0.6,1.8]。 因此，它包含成对的清晰和有雾的图像，其中清晰无雾图像对应多个有雾图像，这些有雾图像是在不同的参数A和β下生成的。

测试集由综合目标测试集（SOTS）和混合主观测试集（HSTS）组成，旨在表现出多种评估观点。 SOTS从NYU2中选择500个室内图像（与训练图像不重叠），并按照与训练数据相同的过程来合成模糊图像。HSTS采用与SOTS相同的方式生成10个合成的户外有雾图像，以及10个真实世界的有雾图像，收集现实世界的室外场景 ，结合进行真人主观评审。

### RESIDE-β（RESIDE-OUT）

网址：[RESIDE-β (google.com)](https://sites.google.com/view/reside-dehaze-datasets/reside-%CE%B2)

作者使用2061张来自北京实时天气的真实室外图，使用在Learning depth from single monocular images using deep convolutional neural fifields中提到的算法，对每幅图进行深度估计，最终用β在\[0.04、0.06、0.08、0.08、0.1、0.15、0.95、0.95、1]中合成了72,135张户外有雾图像。这套新的图像被称为户外训练集（OTS），由成对的干净无雾的户外图像和生成的有雾图像组成。

### RESIDE-4K

训练集包含3000张ITS图像对和3000张OTS图像对。

测试集将室内和室外的图像对混合在一起，形成一个由1000张图像对组成的测试集。

### D-HAZY

D-HAZY，建立在Middelbury 和NYU深度数据集上，这些数据集提供各种场景的图像及其相应的深度图。包含1400多对图像的数据集，其中包括同一场景的地面真实参考图像和模糊图像。

### DENSE-HAZE

发布时间：2019

以浓密均匀的朦胧场景为特征，包含33对真实的朦胧图像和各种室外场景的相应无霾图像。通过引入由专业雾霾机器生成的真实雾霾来记录雾霾场景。

### HAZE-4K

### I-HAZE

发布时间：2018

### O-HAZE

发布时间：2018

### RS-HAZE

### NH-HAZE

发布时间：2020

这是一个非均匀的真实数据集，具有成对的真实雾霾和相应的无雾图像。这是第一个非均匀图像去雾数据集，包含55个室外场景。在场景中引入了非均匀雾，使用专业雾霾制造器模拟有雾场景。

## 评价指标

![\<img alt="" data-attachment-key="VSJWAQPH" src="attachments/VSJWAQPH.png" ztype="zimage">](attachments/VSJWAQPH.png)

## 思考和展望

目前图像去雾领域存在的困难主要有以下几点。

1.  网络的训练需要大量的无雾-有雾图像对作为支撑，但是实际中这样的数据集获取困难。目前的做法是通过一些物理模型如大气散射模型，将高质量无雾图像处理得到低质量的有雾图像，形成合成数据集。但是这一类模型往往无法很好地模拟真实图像降质的过程。
2.  网络的泛化能力差，主要表现在一个数据集上训练的模型应用到另一个数据集上效果往往不佳。这是因为样本分布不一致，根本原因在于雾霾变化多样，在某些场景中得到的雾霾特征往往不适用于另一个场景。

为了解决以上两大难题，我们可以从这些角度出发。

## 创新方向

### 数据预处理

针对网络的训练需要大量数据作为支撑，但是目前数据集数据量有限，我们就需要考虑数据预处理。

1.  **组合多个数据集并降低差异**，

    <span style="color: rgb(44, 62, 80)"><span style="background-color: rgb(255, 255, 255)">由于每个数据集之间有一些细微的差异，例如颜色差异、物体复杂性、拍摄所用的相机差别等。直接组合会降低去雾结果指标，因此我们设计一种数据预处理技术，来减少数据集之间的分布差距。</span></span>

    <span class="citation" data-citation="%7B%22citationItems%22%3A%5B%7B%22uris%22%3A%5B%22http%3A%2F%2Fzotero.org%2Fusers%2F10046823%2Fitems%2FKA6CRQKY%22%5D%2C%22itemData%22%3A%7B%22id%22%3A%22http%3A%2F%2Fzotero.org%2Fusers%2F10046823%2Fitems%2FKA6CRQKY%22%2C%22type%22%3A%22article%22%2C%22abstract%22%3A%22Recent%20years%20have%20witnessed%20an%20increased%20interest%20in%20image%20dehazing.%20Many%20deep%20learning%20methods%20have%20been%20proposed%20to%20tackle%20this%20challenge%2C%20and%20have%20made%20significant%20accomplishments%20dealing%20with%20homogeneous%20haze.%20However%2C%20these%20solutions%20cannot%20maintain%20comparable%20performance%20when%20they%20are%20applied%20to%20images%20with%20non-homogeneous%20haze%2C%20e.g.%2C%20NH-HAZE23%20dataset%20introduced%20by%20NTIRE%20challenges.%20One%20of%20the%20reasons%20for%20such%20failures%20is%20that%20non-homogeneous%20haze%20does%20not%20obey%20one%20of%20the%20assumptions%20that%20is%20required%20for%20modeling%20homogeneous%20haze.%20In%20addition%2C%20a%20large%20number%20of%20pairs%20of%20non-homogeneous%20hazy%20image%20and%20the%20clean%20counterpart%20is%20required%20using%20traditional%20end-to-end%20training%20approaches%2C%20while%20NH-HAZE23%20dataset%20is%20of%20limited%20quantities.%20Although%20it%20is%20possible%20to%20augment%20the%20NH-HAZE23%20dataset%20by%20leveraging%20other%20non-homogeneous%20dehazing%20datasets%2C%20we%20observe%20that%20it%20is%20necessary%20to%20design%20a%20proper%20data-preprocessing%20approach%20that%20reduces%20the%20distribution%20gaps%20between%20the%20target%20dataset%20and%20the%20augmented%20one.%20This%20finding%20indeed%20aligns%20with%20the%20essence%20of%20data-centric%20AI.%20With%20a%20novel%20network%20architecture%20and%20a%20principled%20data-preprocessing%20approach%20that%20systematically%20enhances%20data%20quality%2C%20we%20present%20an%20innovative%20dehazing%20method.%20Specifically%2C%20we%20apply%20RGB-channel-wise%20transformations%20on%20the%20augmented%20datasets%2C%20and%20incorporate%20the%20state-of-the-art%20transformers%20as%20the%20backbone%20in%20the%20two-branch%20framework.%20We%20conduct%20extensive%20experiments%20and%20ablation%20study%20to%20demonstrate%20the%20effectiveness%20of%20our%20proposed%20method.%22%2C%22language%22%3A%22en%22%2C%22note%22%3A%22arXiv%3A2304.07874%20%5Bcs%5D%22%2C%22number%22%3A%22arXiv%3A2304.07874%22%2C%22publisher%22%3A%22arXiv%22%2C%22source%22%3A%22arXiv.org%22%2C%22title%22%3A%22A%20Data-Centric%20Solution%20to%20NonHomogeneous%20Dehazing%20via%20Vision%20Transformer%22%2C%22URL%22%3A%22http%3A%2F%2Farxiv.org%2Fabs%2F2304.07874%22%2C%22author%22%3A%5B%7B%22family%22%3A%22Liu%22%2C%22given%22%3A%22Yangyi%22%7D%2C%7B%22family%22%3A%22Liu%22%2C%22given%22%3A%22Huan%22%7D%2C%7B%22family%22%3A%22Li%22%2C%22given%22%3A%22Liangyan%22%7D%2C%7B%22family%22%3A%22Wu%22%2C%22given%22%3A%22Zijun%22%7D%2C%7B%22family%22%3A%22Chen%22%2C%22given%22%3A%22Jun%22%7D%5D%2C%22accessed%22%3A%7B%22date-parts%22%3A%5B%5B%222023%22%2C10%2C25%5D%5D%7D%2C%22issued%22%3A%7B%22date-parts%22%3A%5B%5B%222023%22%2C4%2C18%5D%5D%7D%7D%7D%5D%2C%22properties%22%3A%7B%7D%7D" ztype="zcitation">(<span class="citation-item"><a href="zotero://select/library/items/KA6CRQKY">Liu 等, 2023</a></span>)</span>

    提出了一种新的预处理技术，对数据集之间明显的颜色差异进行校正，

    <span style="color: rgb(44, 62, 80)"><span style="background-color: rgb(255, 255, 255)">并且将增强后的数据分布转移到目标数据分布。显著的降低了数据集之间的差异，增加了数据量，从而提高了去雾效果。</span></span>

2.  **缓解合成数据集和真实数据集之间的差距。**

    <span class="citation" data-citation="%7B%22citationItems%22%3A%5B%7B%22uris%22%3A%5B%22http%3A%2F%2Fzotero.org%2Fusers%2F10046823%2Fitems%2FBJNHXL93%22%5D%2C%22itemData%22%3A%7B%22id%22%3A%22http%3A%2F%2Fzotero.org%2Fusers%2F10046823%2Fitems%2FBJNHXL93%22%2C%22type%22%3A%22paper-conference%22%2C%22abstract%22%3A%22Existing%20dehazing%20approaches%20struggle%20to%20process%20realworld%20hazy%20images%20owing%20to%20the%20lack%20of%20paired%20real%20data%20and%20robust%20priors.%20In%20this%20work%2C%20we%20present%20a%20new%20paradigm%20for%20real%20image%20dehazing%20from%20the%20perspectives%20of%20synthesizing%20more%20realistic%20hazy%20data%20and%20introducing%20more%20robust%20priors%20into%20the%20network.%20Speci%EF%AC%81cally%2C%20(1)%20instead%20of%20adopting%20the%20de%20facto%20physical%20scattering%20model%2C%20we%20rethink%20the%20degradation%20of%20real%20hazy%20images%20and%20propose%20a%20phenomenological%20pipeline%20considering%20diverse%20degradation%20types.%20(2)%20We%20propose%20a%20Real%20Image%20Dehazing%20network%20via%20high-quality%20Codebook%20Priors%20(RIDCP).%20Firstly%2C%20a%20VQGAN%20is%20pre-trained%20on%20a%20large-scale%20high-quality%20dataset%20to%20obtain%20the%20discrete%20codebook%2C%20encapsulating%20high-quality%20priors%20(HQPs).%20After%20replacing%20the%20negative%20effects%20brought%20by%20haze%20with%20HQPs%2C%20the%20decoder%20equipped%20with%20a%20novel%20normalized%20feature%20alignment%20module%20can%20effectively%20utilize%20high-quality%20features%20and%20produce%20clean%20results.%20However%2C%20although%20our%20degradation%20pipeline%20drastically%20mitigates%20the%20domain%20gap%20between%20synthetic%20and%20real%20data%2C%20it%20is%20still%20intractable%20to%20avoid%20it%2C%20which%20challenges%20HQPs%20matching%20in%20the%20wild.%20Thus%2C%20we%20re-calculate%20the%20distance%20when%20matching%20the%20features%20to%20the%20HQPs%20by%20a%20controllable%20matching%20operation%2C%20which%20facilitates%20%EF%AC%81nding%20better%20counterparts.%20We%20provide%20a%20recommendation%20to%20control%20the%20matching%20based%20on%20an%20explainable%20solution.%20Users%20can%20also%20%EF%AC%82exibly%20adjust%20the%20enhancement%20degree%20as%20per%20their%20preference.%20Extensive%20experiments%20verify%20the%20effectiveness%20of%20our%20data%20synthesis%20pipeline%20and%20the%20superior%20performance%20of%20RIDCP%20in%20real%20image%20dehazing.%20Code%20and%20data%20are%20released%20at%20https%3A%2F%2Frqwu.github.io%2Fprojects%2FRIDCP.%22%2C%22container-title%22%3A%222023%20IEEE%2FCVF%20Conference%20on%20Computer%20Vision%20and%20Pattern%20Recognition%20(CVPR)%22%2C%22DOI%22%3A%2210.1109%2FCVPR52729.2023.02134%22%2C%22event-place%22%3A%22Vancouver%2C%20BC%2C%20Canada%22%2C%22event-title%22%3A%222023%20IEEE%2FCVF%20Conference%20on%20Computer%20Vision%20and%20Pattern%20Recognition%20(CVPR)%22%2C%22ISBN%22%3A%229798350301298%22%2C%22language%22%3A%22en%22%2C%22page%22%3A%2222282-22291%22%2C%22publisher%22%3A%22IEEE%22%2C%22publisher-place%22%3A%22Vancouver%2C%20BC%2C%20Canada%22%2C%22source%22%3A%22DOI.org%20(Crossref)%22%2C%22title%22%3A%22RIDCP%3A%20Revitalizing%20Real%20Image%20Dehazing%20via%20High-Quality%20Codebook%20Priors%22%2C%22title-short%22%3A%22RIDCP%22%2C%22URL%22%3A%22https%3A%2F%2Fieeexplore.ieee.org%2Fdocument%2F10203847%2F%22%2C%22author%22%3A%5B%7B%22family%22%3A%22Wu%22%2C%22given%22%3A%22Rui-Qi%22%7D%2C%7B%22family%22%3A%22Duan%22%2C%22given%22%3A%22Zheng-Peng%22%7D%2C%7B%22family%22%3A%22Guo%22%2C%22given%22%3A%22Chun-Le%22%7D%2C%7B%22family%22%3A%22Chai%22%2C%22given%22%3A%22Zhi%22%7D%2C%7B%22family%22%3A%22Li%22%2C%22given%22%3A%22Chongyi%22%7D%5D%2C%22accessed%22%3A%7B%22date-parts%22%3A%5B%5B%222023%22%2C9%2C19%5D%5D%7D%2C%22issued%22%3A%7B%22date-parts%22%3A%5B%5B%222023%22%2C6%5D%5D%7D%7D%7D%5D%2C%22properties%22%3A%7B%7D%7D" ztype="zcitation">(<span class="citation-item"><a href="zotero://select/library/items/BJNHXL93">Wu 等, 2023</a></span>)</span>

    重新设计了合成数据集的生成过程，

    <span style="color: rgb(44, 62, 80)"><span style="background-color: rgb(255, 255, 255)">考虑到了图像退化的各种因素。由此得到的数据集缓解合成数据和真实数据之间的差距。</span></span>

3.  应用图像增强的方法，如直方图均衡化、对比度提升初步处理数据。

### 模型结构

\*\*引入多分支及分类。 \*\*针对雾霾变化多样，在某些场景中得到的雾霾特征往往不适用于另一个场景这一特点。我们可以针对不同的雾霾具体使用不同的网络进行去雾。对当前去雾数据集分析可以得知，均匀雾霾、非均匀雾霾有一定的区别。<span style="color: rgb(44, 62, 80)"><span style="background-color: rgb(255, 255, 255)">非均匀雾霾并不完全由图像场景深度决定，不同区域的雾霾浓度往往不一。因此通常的去雾方法在去除非均匀雾霾上效果不佳。我们可以根据不同种类的雾霾，引入不同的编码器，提取不同类别雾霾之间特异的特征，形成一种多分支结构。随后再接上普通编码器，提取雾霾大类之间的相同特征。随后送去解码器输出去雾图像。</span></span>

\*\*引入高质量先验。 \*\*从去雾网络发展历程可以得知，高质量先验知识对去雾网络的设计有很大的帮助，以往的网络模型都是人工通过经验总结得到的先验知识，然后据此设计网络，如暗通道先验和颜色衰减先验，由于先验知识本身具有一定的局限性，从而导致设计出来的网络泛化能力不佳。因此<span class="citation" data-citation="%7B%22citationItems%22%3A%5B%7B%22uris%22%3A%5B%22http%3A%2F%2Fzotero.org%2Fusers%2F10046823%2Fitems%2FBJNHXL93%22%5D%2C%22itemData%22%3A%7B%22id%22%3A%22http%3A%2F%2Fzotero.org%2Fusers%2F10046823%2Fitems%2FBJNHXL93%22%2C%22type%22%3A%22paper-conference%22%2C%22abstract%22%3A%22Existing%20dehazing%20approaches%20struggle%20to%20process%20realworld%20hazy%20images%20owing%20to%20the%20lack%20of%20paired%20real%20data%20and%20robust%20priors.%20In%20this%20work%2C%20we%20present%20a%20new%20paradigm%20for%20real%20image%20dehazing%20from%20the%20perspectives%20of%20synthesizing%20more%20realistic%20hazy%20data%20and%20introducing%20more%20robust%20priors%20into%20the%20network.%20Speci%EF%AC%81cally%2C%20(1)%20instead%20of%20adopting%20the%20de%20facto%20physical%20scattering%20model%2C%20we%20rethink%20the%20degradation%20of%20real%20hazy%20images%20and%20propose%20a%20phenomenological%20pipeline%20considering%20diverse%20degradation%20types.%20(2)%20We%20propose%20a%20Real%20Image%20Dehazing%20network%20via%20high-quality%20Codebook%20Priors%20(RIDCP).%20Firstly%2C%20a%20VQGAN%20is%20pre-trained%20on%20a%20large-scale%20high-quality%20dataset%20to%20obtain%20the%20discrete%20codebook%2C%20encapsulating%20high-quality%20priors%20(HQPs).%20After%20replacing%20the%20negative%20effects%20brought%20by%20haze%20with%20HQPs%2C%20the%20decoder%20equipped%20with%20a%20novel%20normalized%20feature%20alignment%20module%20can%20effectively%20utilize%20high-quality%20features%20and%20produce%20clean%20results.%20However%2C%20although%20our%20degradation%20pipeline%20drastically%20mitigates%20the%20domain%20gap%20between%20synthetic%20and%20real%20data%2C%20it%20is%20still%20intractable%20to%20avoid%20it%2C%20which%20challenges%20HQPs%20matching%20in%20the%20wild.%20Thus%2C%20we%20re-calculate%20the%20distance%20when%20matching%20the%20features%20to%20the%20HQPs%20by%20a%20controllable%20matching%20operation%2C%20which%20facilitates%20%EF%AC%81nding%20better%20counterparts.%20We%20provide%20a%20recommendation%20to%20control%20the%20matching%20based%20on%20an%20explainable%20solution.%20Users%20can%20also%20%EF%AC%82exibly%20adjust%20the%20enhancement%20degree%20as%20per%20their%20preference.%20Extensive%20experiments%20verify%20the%20effectiveness%20of%20our%20data%20synthesis%20pipeline%20and%20the%20superior%20performance%20of%20RIDCP%20in%20real%20image%20dehazing.%20Code%20and%20data%20are%20released%20at%20https%3A%2F%2Frqwu.github.io%2Fprojects%2FRIDCP.%22%2C%22container-title%22%3A%222023%20IEEE%2FCVF%20Conference%20on%20Computer%20Vision%20and%20Pattern%20Recognition%20(CVPR)%22%2C%22DOI%22%3A%2210.1109%2FCVPR52729.2023.02134%22%2C%22event-place%22%3A%22Vancouver%2C%20BC%2C%20Canada%22%2C%22event-title%22%3A%222023%20IEEE%2FCVF%20Conference%20on%20Computer%20Vision%20and%20Pattern%20Recognition%20(CVPR)%22%2C%22ISBN%22%3A%229798350301298%22%2C%22language%22%3A%22en%22%2C%22page%22%3A%2222282-22291%22%2C%22publisher%22%3A%22IEEE%22%2C%22publisher-place%22%3A%22Vancouver%2C%20BC%2C%20Canada%22%2C%22source%22%3A%22DOI.org%20(Crossref)%22%2C%22title%22%3A%22RIDCP%3A%20Revitalizing%20Real%20Image%20Dehazing%20via%20High-Quality%20Codebook%20Priors%22%2C%22title-short%22%3A%22RIDCP%22%2C%22URL%22%3A%22https%3A%2F%2Fieeexplore.ieee.org%2Fdocument%2F10203847%2F%22%2C%22author%22%3A%5B%7B%22family%22%3A%22Wu%22%2C%22given%22%3A%22Rui-Qi%22%7D%2C%7B%22family%22%3A%22Duan%22%2C%22given%22%3A%22Zheng-Peng%22%7D%2C%7B%22family%22%3A%22Guo%22%2C%22given%22%3A%22Chun-Le%22%7D%2C%7B%22family%22%3A%22Chai%22%2C%22given%22%3A%22Zhi%22%7D%2C%7B%22family%22%3A%22Li%22%2C%22given%22%3A%22Chongyi%22%7D%5D%2C%22accessed%22%3A%7B%22date-parts%22%3A%5B%5B%222023%22%2C9%2C19%5D%5D%7D%2C%22issued%22%3A%7B%22date-parts%22%3A%5B%5B%222023%22%2C6%5D%5D%7D%7D%7D%5D%2C%22properties%22%3A%7B%7D%7D" ztype="zcitation">(<span class="citation-item"><a href="zotero://select/library/items/BJNHXL93">Wu 等, 2023</a></span>)</span>通过预训练一个网络来得到高质量的先验知识，然后将高质量先验与网络进行融合训练，再通过解码器输出无雾图像。

\*\*引入选择机制。 \*\*图像中并不是所有区域都是同等重要的，比如天空、雪地等区域去雾重要性不高，而其他雾霾浓度高，距离近的区域则相比于雾霾少距离远的物体去雾重要性更强。<span class="citation" data-citation="%7B%22citationItems%22%3A%5B%7B%22uris%22%3A%5B%22http%3A%2F%2Fzotero.org%2Fusers%2F10046823%2Fitems%2F4SHMI7H5%22%5D%2C%22itemData%22%3A%7B%22id%22%3A%22http%3A%2F%2Fzotero.org%2Fusers%2F10046823%2Fitems%2F4SHMI7H5%22%2C%22type%22%3A%22article-journal%22%2C%22abstract%22%3A%22Image%20restoration%20aims%20to%20reconstruct%20a%20sharp%20image%20from%20its%20degraded%20counterpart%2C%20which%20plays%20an%20important%20role%20in%20many%20fields.%20Recently%2C%20Transformer%20models%20have%20achieved%20promising%20performance%20on%20various%20image%20restoration%20tasks.%20However%2C%20their%20quadratic%20complexity%20remains%20an%20intractable%20issue%20for%20practical%20applications.%20The%20aim%20of%20this%20study%20is%20to%20develop%20an%20efficient%20and%20effective%20framework%20for%20image%20restoration.%20Inspired%20by%20the%20fact%20that%20different%20regions%20in%20a%20corrupted%20image%20always%20undergo%20degradations%20in%20various%20degrees%2C%20we%20propose%20to%20focus%20more%20on%20the%20important%20areas%20for%20reconstruction.%20To%20this%20end%2C%20we%20introduce%20a%20dualdomain%20selection%20mechanism%20to%20emphasize%20crucial%20information%20for%20restoration%2C%20such%20as%20edge%20signals%20and%20hard%20regions.%20In%20addition%2C%20we%20split%20high-resolution%20features%20to%20insert%20multiscale%20receptive%20fields%20into%20the%20network%2C%20which%20improves%20both%20efficiency%20and%20performance.%20Finally%2C%20the%20proposed%20network%2C%20dubbed%20FocalNet%2C%20is%20built%20by%20incorporating%20these%20designs%20into%20a%20U-shaped%20backbone.%20Extensive%20experiments%20demonstrate%20that%20our%20model%20achieves%20state-of-the-art%20performance%20on%20ten%20datasets%20for%20three%20tasks%2C%20including%20single-image%20defocus%20deblurring%2C%20image%20dehazing%2C%20and%20image%20desnowing.%20Our%20code%20is%20available%20at%20https%3A%2F%2Fgithub.com%2Fc-yn%2FFocalNet.%22%2C%22language%22%3A%22en%22%2C%22source%22%3A%22Zotero%22%2C%22title%22%3A%22Focal%20Network%20for%20Image%20Restoration%22%2C%22author%22%3A%5B%7B%22family%22%3A%22Cui%22%2C%22given%22%3A%22Yuning%22%7D%2C%7B%22family%22%3A%22Ren%22%2C%22given%22%3A%22Wenqi%22%7D%2C%7B%22family%22%3A%22Cao%22%2C%22given%22%3A%22Xiaochun%22%7D%2C%7B%22family%22%3A%22Knoll%22%2C%22given%22%3A%22Alois%22%7D%5D%2C%22issued%22%3A%7B%22date-parts%22%3A%5B%5B%222023%22%5D%5D%7D%7D%7D%5D%2C%22properties%22%3A%7B%7D%7D" ztype="zcitation">(<span class="citation-item"><a href="zotero://select/library/items/4SHMI7H5">Cui 等, 2023</a></span>)</span>提出了双域选择机制，双域主要表现在<span style="color: rgb(44, 62, 80)"><span style="background-color: rgb(255, 255, 255)">空间选择模块和频率选择模块。空间选择模块通过深度卷积层来确定每个通道中图像退化区域的大致位置。然后利用频率选择模块对高频信号或硬区域进行放大，去除特征中的低频成分。通过这种机制，模型会更专注于雾霾更重的关键区域。</span></span>

学习雾霾和图像背景之间交互的特征。<span class="citation" data-citation="%7B%22citationItems%22%3A%5B%7B%22uris%22%3A%5B%22http%3A%2F%2Fzotero.org%2Fusers%2F10046823%2Fitems%2F8GJPSKZ7%22%5D%2C%22itemData%22%3A%7B%22id%22%3A%22http%3A%2F%2Fzotero.org%2Fusers%2F10046823%2Fitems%2F8GJPSKZ7%22%2C%22type%22%3A%22article%22%2C%22abstract%22%3A%22The%20presence%20of%20non-homogeneous%20haze%20can%20cause%20scene%20blurring%2C%20color%20distortion%2C%20low%20contrast%2C%20and%20other%20degradations%20that%20obscure%20texture%20details.%20Existing%20homogeneous%20dehazing%20methods%20struggle%20to%20handle%20the%20non-uniform%20distribution%20of%20haze%20in%20a%20robust%20manner.%20The%20crucial%20challenge%20of%20non-homogeneous%20dehazing%20is%20to%20effectively%20extract%20the%20non-uniform%20distribution%20features%20and%20reconstruct%20the%20details%20of%20hazy%20areas%20with%20high%20quality.%20In%20this%20paper%2C%20we%20propose%20a%20novel%20self-paced%20semi-curricular%20attention%20network%2C%20called%20SCANet%2C%20for%20non-homogeneous%20image%20dehazing%20that%20focuses%20on%20enhancing%20haze-occluded%20regions.%20Our%20approach%20consists%20of%20an%20attention%20generator%20network%20and%20a%20scene%20reconstruction%20network.%20We%20use%20the%20luminance%20differences%20of%20images%20to%20restrict%20the%20attention%20map%20and%20introduce%20a%20self-paced%20semi-curricular%20learning%20strategy%20to%20reduce%20learning%20ambiguity%20in%20the%20early%20stages%20of%20training.%20Extensive%20quantitative%20and%20qualitative%20experiments%20demonstrate%20that%20our%20SCANet%20outperforms%20many%20state-of-the-art%20methods.%20The%20code%20is%20publicly%20available%20at%20https%3A%2F%2Fgithub.com%2Fgy65896%2FSCANet.%22%2C%22note%22%3A%22arXiv%3A2304.08444%20%5Bcs%5D%5Cnversion%3A%201%22%2C%22number%22%3A%22arXiv%3A2304.08444%22%2C%22publisher%22%3A%22arXiv%22%2C%22source%22%3A%22arXiv.org%22%2C%22title%22%3A%22SCANet%3A%20Self-Paced%20Semi-Curricular%20Attention%20Network%20for%20Non-Homogeneous%20Image%20Dehazing%22%2C%22title-short%22%3A%22SCANet%22%2C%22URL%22%3A%22http%3A%2F%2Farxiv.org%2Fabs%2F2304.08444%22%2C%22author%22%3A%5B%7B%22family%22%3A%22Guo%22%2C%22given%22%3A%22Yu%22%7D%2C%7B%22family%22%3A%22Gao%22%2C%22given%22%3A%22Yuan%22%7D%2C%7B%22family%22%3A%22Liu%22%2C%22given%22%3A%22Ryan%20Wen%22%7D%2C%7B%22family%22%3A%22Lu%22%2C%22given%22%3A%22Yuxu%22%7D%2C%7B%22family%22%3A%22Qu%22%2C%22given%22%3A%22Jingxiang%22%7D%2C%7B%22family%22%3A%22He%22%2C%22given%22%3A%22Shengfeng%22%7D%2C%7B%22family%22%3A%22Ren%22%2C%22given%22%3A%22Wenqi%22%7D%5D%2C%22accessed%22%3A%7B%22date-parts%22%3A%5B%5B%222023%22%2C10%2C21%5D%5D%7D%2C%22issued%22%3A%7B%22date-parts%22%3A%5B%5B%222023%22%2C4%2C17%5D%5D%7D%7D%7D%5D%2C%22properties%22%3A%7B%7D%7D" ztype="zcitation">(<span class="citation-item"><a href="zotero://select/library/items/8GJPSKZ7">Guo 等, 2023</a></span>)</span>通过注意力生成和场景重建网络<span style="color: rgb(44, 62, 80)"><span style="background-color: rgb(255, 255, 255)">专注于学习非均匀雾霾和场景之间的复杂交互特征。</span></span>

提取多尺度特征。使用不同大小的卷积核、多尺度、并行地提取特征。

### 损失函数

采用多个损失函数结合来辅助模型训练，如感知损失、对抗损失、结构相似性损失

### 正则化方法

对比正则化方法

### 训练策略

1.  \*\*对比学习。 \*\*2023年的CVPR中

    <span class="citation" data-citation="%7B%22citationItems%22%3A%5B%7B%22uris%22%3A%5B%22http%3A%2F%2Fzotero.org%2Fusers%2F10046823%2Fitems%2FYMAP3M6X%22%5D%2C%22itemData%22%3A%7B%22id%22%3A%22http%3A%2F%2Fzotero.org%2Fusers%2F10046823%2Fitems%2FYMAP3M6X%22%2C%22type%22%3A%22paper-conference%22%2C%22abstract%22%3A%22Considering%20the%20ill-posed%20nature%2C%20contrastive%20regularization%20has%20been%20developed%20for%20single%20image%20dehazing%2C%20introducing%20the%20information%20from%20negative%20images%20as%20a%20lower%20bound.%20However%2C%20the%20contrastive%20samples%20are%20nonconsensual%2C%20as%20the%20negatives%20are%20usually%20represented%20distantly%20from%20the%20clear%20(i.e.%2C%20positive)%20image%2C%20leaving%20the%20solution%20space%20still%20under-constricted.%20Moreover%2C%20the%20interpretability%20of%20deep%20dehazing%20models%20is%20underexplored%20towards%20the%20physics%20of%20the%20hazing%20process.%20In%20this%20paper%2C%20we%20propose%20a%20novel%20curricular%20contrastive%20regularization%20targeted%20at%20a%20consensual%20contrastive%20space%20as%20opposed%20to%20a%20non-consensual%20one.%20Our%20negatives%2C%20which%20provide%20better%20lower-bound%20constraints%2C%20can%20be%20assembled%20from%201)%20the%20hazy%20image%2C%20and%202)%20corresponding%20restorations%20by%20other%20existing%20methods.%20Further%2C%20due%20to%20the%20different%20similarities%20between%20the%20embeddings%20of%20the%20clear%20image%20and%20negatives%2C%20the%20learning%20dif%EF%AC%81culty%20of%20the%20multiple%20components%20is%20intrinsically%20imbalanced.%20To%20tackle%20this%20issue%2C%20we%20customize%20a%20curriculum%20learning%20strategy%20to%20reweight%20the%20importance%20of%20different%20negatives.%20In%20addition%2C%20to%20improve%20the%20interpretability%20in%20the%20feature%20space%2C%20we%20build%20a%20physics-aware%20dual-branch%20unit%20according%20to%20the%20atmospheric%20scattering%20model.%20With%20the%20unit%2C%20as%20well%20as%20curricular%20contrastive%20regularization%2C%20we%20establish%20our%20dehazing%20network%2C%20named%20C2PNet.%20Extensive%20experiments%20demonstrate%20that%20our%20C2PNet%20signi%EF%AC%81cantly%20outperforms%20state-of-the-art%20methods%2C%20with%20extreme%20PSNR%20boosts%20of%203.94dB%20and%201.50dB%2C%20respectively%2C%20on%20SOTSindoor%20and%20SOTS-outdoor%20datasets.%20Code%20is%20available%20at%20https%3A%2F%2Fgithub.com%2FYuZheng9%2FC2PNet.%22%2C%22container-title%22%3A%222023%20IEEE%2FCVF%20Conference%20on%20Computer%20Vision%20and%20Pattern%20Recognition%20(CVPR)%22%2C%22DOI%22%3A%2210.1109%2FCVPR52729.2023.00560%22%2C%22event-place%22%3A%22Vancouver%2C%20BC%2C%20Canada%22%2C%22event-title%22%3A%222023%20IEEE%2FCVF%20Conference%20on%20Computer%20Vision%20and%20Pattern%20Recognition%20(CVPR)%22%2C%22ISBN%22%3A%229798350301298%22%2C%22language%22%3A%22en%22%2C%22page%22%3A%225785-5794%22%2C%22publisher%22%3A%22IEEE%22%2C%22publisher-place%22%3A%22Vancouver%2C%20BC%2C%20Canada%22%2C%22source%22%3A%22DOI.org%20(Crossref)%22%2C%22title%22%3A%22Curricular%20Contrastive%20Regularization%20for%20Physics-Aware%20Single%20Image%20Dehazing%22%2C%22URL%22%3A%22https%3A%2F%2Fieeexplore.ieee.org%2Fdocument%2F10204264%2F%22%2C%22author%22%3A%5B%7B%22family%22%3A%22Zheng%22%2C%22given%22%3A%22Yu%22%7D%2C%7B%22family%22%3A%22Zhan%22%2C%22given%22%3A%22Jiahui%22%7D%2C%7B%22family%22%3A%22He%22%2C%22given%22%3A%22Shengfeng%22%7D%2C%7B%22family%22%3A%22Dong%22%2C%22given%22%3A%22Junyu%22%7D%2C%7B%22family%22%3A%22Du%22%2C%22given%22%3A%22Yong%22%7D%5D%2C%22accessed%22%3A%7B%22date-parts%22%3A%5B%5B%222023%22%2C9%2C19%5D%5D%7D%2C%22issued%22%3A%7B%22date-parts%22%3A%5B%5B%222023%22%2C6%5D%5D%7D%7D%7D%5D%2C%22properties%22%3A%7B%7D%7D" ztype="zcitation">(<span class="citation-item"><a href="zotero://select/library/items/YMAP3M6X">Zheng 等, 2023</a></span>)</span>

    提出了这一观点，在训练的过程中不断和其他模型训练得到的去雾图像做对比，一开始和效果较差的比较并进行自我矫正，逐步和效果好的比较直到接近真实无雾图像。在学习的过程中会逐步结合各种去雾方法的优点、不断地调整来达到更好的结果。因此我们可以精挑细选近些年去雾效果较好的、去雾方法差别较大的作为比较对象进行对比学习。

2.  \*\*课程学习。 \*\*

    <span class="citation" data-citation="%7B%22citationItems%22%3A%5B%7B%22uris%22%3A%5B%22http%3A%2F%2Fzotero.org%2Fusers%2F10046823%2Fitems%2F8GJPSKZ7%22%5D%2C%22itemData%22%3A%7B%22id%22%3A%22http%3A%2F%2Fzotero.org%2Fusers%2F10046823%2Fitems%2F8GJPSKZ7%22%2C%22type%22%3A%22article%22%2C%22abstract%22%3A%22The%20presence%20of%20non-homogeneous%20haze%20can%20cause%20scene%20blurring%2C%20color%20distortion%2C%20low%20contrast%2C%20and%20other%20degradations%20that%20obscure%20texture%20details.%20Existing%20homogeneous%20dehazing%20methods%20struggle%20to%20handle%20the%20non-uniform%20distribution%20of%20haze%20in%20a%20robust%20manner.%20The%20crucial%20challenge%20of%20non-homogeneous%20dehazing%20is%20to%20effectively%20extract%20the%20non-uniform%20distribution%20features%20and%20reconstruct%20the%20details%20of%20hazy%20areas%20with%20high%20quality.%20In%20this%20paper%2C%20we%20propose%20a%20novel%20self-paced%20semi-curricular%20attention%20network%2C%20called%20SCANet%2C%20for%20non-homogeneous%20image%20dehazing%20that%20focuses%20on%20enhancing%20haze-occluded%20regions.%20Our%20approach%20consists%20of%20an%20attention%20generator%20network%20and%20a%20scene%20reconstruction%20network.%20We%20use%20the%20luminance%20differences%20of%20images%20to%20restrict%20the%20attention%20map%20and%20introduce%20a%20self-paced%20semi-curricular%20learning%20strategy%20to%20reduce%20learning%20ambiguity%20in%20the%20early%20stages%20of%20training.%20Extensive%20quantitative%20and%20qualitative%20experiments%20demonstrate%20that%20our%20SCANet%20outperforms%20many%20state-of-the-art%20methods.%20The%20code%20is%20publicly%20available%20at%20https%3A%2F%2Fgithub.com%2Fgy65896%2FSCANet.%22%2C%22note%22%3A%22arXiv%3A2304.08444%20%5Bcs%5D%5Cnversion%3A%201%22%2C%22number%22%3A%22arXiv%3A2304.08444%22%2C%22publisher%22%3A%22arXiv%22%2C%22source%22%3A%22arXiv.org%22%2C%22title%22%3A%22SCANet%3A%20Self-Paced%20Semi-Curricular%20Attention%20Network%20for%20Non-Homogeneous%20Image%20Dehazing%22%2C%22title-short%22%3A%22SCANet%22%2C%22URL%22%3A%22http%3A%2F%2Farxiv.org%2Fabs%2F2304.08444%22%2C%22author%22%3A%5B%7B%22family%22%3A%22Guo%22%2C%22given%22%3A%22Yu%22%7D%2C%7B%22family%22%3A%22Gao%22%2C%22given%22%3A%22Yuan%22%7D%2C%7B%22family%22%3A%22Liu%22%2C%22given%22%3A%22Ryan%20Wen%22%7D%2C%7B%22family%22%3A%22Lu%22%2C%22given%22%3A%22Yuxu%22%7D%2C%7B%22family%22%3A%22Qu%22%2C%22given%22%3A%22Jingxiang%22%7D%2C%7B%22family%22%3A%22He%22%2C%22given%22%3A%22Shengfeng%22%7D%2C%7B%22family%22%3A%22Ren%22%2C%22given%22%3A%22Wenqi%22%7D%5D%2C%22accessed%22%3A%7B%22date-parts%22%3A%5B%5B%222023%22%2C10%2C21%5D%5D%7D%2C%22issued%22%3A%7B%22date-parts%22%3A%5B%5B%222023%22%2C4%2C17%5D%5D%7D%7D%7D%5D%2C%22properties%22%3A%7B%7D%7D" ztype="zcitation">(<span class="citation-item"><a href="zotero://select/library/items/8GJPSKZ7">Guo 等, 2023</a></span>)</span>

    在其论文中

    <span style="color: rgb(44, 62, 80)"><span style="background-color: rgb(255, 255, 255)">为了增强亮度差异大的区域去雾效果，引入了自组织半课程学习的注意力图生成策略，该方法加快了模型参数收敛。减少了训练早期多目标预测引起的学习歧义。</span></span>

    (Zheng 等, 2023)

    也在其论文中采用了课程学习策略，其主要思想是根据学习的难度不同动态的调整学习参数。

3.  采用

    <span style="color: rgb(44, 62, 80)"><span style="background-color: rgb(255, 255, 255)">YCbCr颜色空间。</span></span>

    <span class="citation" data-citation="%7B%22citationItems%22%3A%5B%7B%22uris%22%3A%5B%22http%3A%2F%2Fzotero.org%2Fusers%2F10046823%2Fitems%2F8GJPSKZ7%22%5D%2C%22itemData%22%3A%7B%22id%22%3A%22http%3A%2F%2Fzotero.org%2Fusers%2F10046823%2Fitems%2F8GJPSKZ7%22%2C%22type%22%3A%22article%22%2C%22abstract%22%3A%22The%20presence%20of%20non-homogeneous%20haze%20can%20cause%20scene%20blurring%2C%20color%20distortion%2C%20low%20contrast%2C%20and%20other%20degradations%20that%20obscure%20texture%20details.%20Existing%20homogeneous%20dehazing%20methods%20struggle%20to%20handle%20the%20non-uniform%20distribution%20of%20haze%20in%20a%20robust%20manner.%20The%20crucial%20challenge%20of%20non-homogeneous%20dehazing%20is%20to%20effectively%20extract%20the%20non-uniform%20distribution%20features%20and%20reconstruct%20the%20details%20of%20hazy%20areas%20with%20high%20quality.%20In%20this%20paper%2C%20we%20propose%20a%20novel%20self-paced%20semi-curricular%20attention%20network%2C%20called%20SCANet%2C%20for%20non-homogeneous%20image%20dehazing%20that%20focuses%20on%20enhancing%20haze-occluded%20regions.%20Our%20approach%20consists%20of%20an%20attention%20generator%20network%20and%20a%20scene%20reconstruction%20network.%20We%20use%20the%20luminance%20differences%20of%20images%20to%20restrict%20the%20attention%20map%20and%20introduce%20a%20self-paced%20semi-curricular%20learning%20strategy%20to%20reduce%20learning%20ambiguity%20in%20the%20early%20stages%20of%20training.%20Extensive%20quantitative%20and%20qualitative%20experiments%20demonstrate%20that%20our%20SCANet%20outperforms%20many%20state-of-the-art%20methods.%20The%20code%20is%20publicly%20available%20at%20https%3A%2F%2Fgithub.com%2Fgy65896%2FSCANet.%22%2C%22note%22%3A%22arXiv%3A2304.08444%20%5Bcs%5D%5Cnversion%3A%201%22%2C%22number%22%3A%22arXiv%3A2304.08444%22%2C%22publisher%22%3A%22arXiv%22%2C%22source%22%3A%22arXiv.org%22%2C%22title%22%3A%22SCANet%3A%20Self-Paced%20Semi-Curricular%20Attention%20Network%20for%20Non-Homogeneous%20Image%20Dehazing%22%2C%22title-short%22%3A%22SCANet%22%2C%22URL%22%3A%22http%3A%2F%2Farxiv.org%2Fabs%2F2304.08444%22%2C%22author%22%3A%5B%7B%22family%22%3A%22Guo%22%2C%22given%22%3A%22Yu%22%7D%2C%7B%22family%22%3A%22Gao%22%2C%22given%22%3A%22Yuan%22%7D%2C%7B%22family%22%3A%22Liu%22%2C%22given%22%3A%22Ryan%20Wen%22%7D%2C%7B%22family%22%3A%22Lu%22%2C%22given%22%3A%22Yuxu%22%7D%2C%7B%22family%22%3A%22Qu%22%2C%22given%22%3A%22Jingxiang%22%7D%2C%7B%22family%22%3A%22He%22%2C%22given%22%3A%22Shengfeng%22%7D%2C%7B%22family%22%3A%22Ren%22%2C%22given%22%3A%22Wenqi%22%7D%5D%2C%22accessed%22%3A%7B%22date-parts%22%3A%5B%5B%222023%22%2C10%2C21%5D%5D%7D%2C%22issued%22%3A%7B%22date-parts%22%3A%5B%5B%222023%22%2C4%2C17%5D%5D%7D%7D%7D%5D%2C%22properties%22%3A%7B%7D%7D" ztype="zcitation">(<span class="citation-item"><a href="zotero://select/library/items/8GJPSKZ7">Guo 等, 2023</a></span>)</span>

    、

    (Singh 等, 2020)

    在其论文中提出采用

    <span style="color: rgb(44, 62, 80)"><span style="background-color: rgb(255, 255, 255)">YCbCr 颜色空间，因为其相比于 RGB 多了一个能够表示图像的亮度的量，而雾霾浓度大的区域往往亮度更强。有助于模型学习到有关雾霾亮度的特征。（沛：大多数情况，也不是所有情况都行，有时候雾霾可能不太符合这种先验）</span></span>

### 轻量化

随着深度学习的发展，模型逐渐往复杂化、巨大化发展，所需的计算量更是水涨船高，这对去雾系统的实时应用不利，因此，有必要对模型进行效果和运行速度的权衡，节约计算开销。

1.  采用四元数网络。

    <span class="citation" data-citation="%7B%22citationItems%22%3A%5B%7B%22uris%22%3A%5B%22http%3A%2F%2Fzotero.org%2Fusers%2F10046823%2Fitems%2F6ELGGPS7%22%5D%2C%22itemData%22%3A%7B%22id%22%3A%22http%3A%2F%2Fzotero.org%2Fusers%2F10046823%2Fitems%2F6ELGGPS7%22%2C%22type%22%3A%22article-journal%22%2C%22abstract%22%3A%22Single-image%20haze%20removal%20is%20challenging%20due%20to%20its%20ill-posed%20nature.%20The%20breadth%20of%20real-world%20scenarios%20makes%20it%20dif%EF%AC%81cult%20to%20%EF%AC%81nd%20an%20optimal%20dehazing%20approach%20that%20works%20well%20for%20various%20applications.%20This%20article%20addresses%20this%20challenge%20by%20utilizing%20a%20novel%20robust%20quaternion%20neural%20network%20architecture%20for%20single-image%20dehazing%20applications.%20The%20architecture%E2%80%99s%20performance%20to%20dehaze%20images%20and%20its%20impact%20on%20real%20applications%2C%20such%20as%20object%20detection%2C%20is%20presented.%20The%20proposed%20single-image%20dehazing%20network%20is%20based%20on%20an%20encoder-decoder%20architecture%20capable%20of%20taking%20advantage%20of%20quaternion%20image%20representation%20without%20interrupting%20the%20quaternion%20data%EF%AC%82ow%20end-to-end.%20We%20achieve%20this%20by%20introducing%20a%20novel%20quaternion%20pixel-wise%20loss%20function%20and%20quaternion%20instance%20normalization%20layer.%20The%20performance%20of%20the%20proposed%20QCNN-H%20quaternion%20framework%20is%20evaluated%20on%20two%20synthetic%20datasets%2C%20two%20real-world%20datasets%2C%20and%20one%20real-world%20task-oriented%20benchmark.%20Extensive%20experiments%20con%EF%AC%81rm%20that%20the%20QCNN-H%20outperforms%20state-of-the-art%20haze%20removal%20procedures%20in%20visual%20quality%20and%20quantitative%20metrics.%20Furthermore%2C%20the%20evaluation%20shows%20increased%20accuracy%20and%20recall%20of%20state-of-the-art%20object%20detection%20in%20hazy%20scenes%20using%20the%20presented%20QCNN-H%20method.%20This%20is%20the%20%EF%AC%81rst%20time%20the%20quaternion%20convolutional%20network%20has%20been%20applied%20to%20the%20haze%20removal%20task.%22%2C%22container-title%22%3A%22IEEE%20Transactions%20on%20Cybernetics%22%2C%22DOI%22%3A%2210.1109%2FTCYB.2023.3238640%22%2C%22ISSN%22%3A%222168-2267%2C%202168-2275%22%2C%22journalAbbreviation%22%3A%22IEEE%20Trans.%20Cybern.%22%2C%22language%22%3A%22en%22%2C%22page%22%3A%221-11%22%2C%22source%22%3A%22DOI.org%20(Crossref)%22%2C%22title%22%3A%22QCNN-H%3A%20Single-Image%20Dehazing%20Using%20Quaternion%20Neural%20Networks%22%2C%22title-short%22%3A%22QCNN-H%22%2C%22URL%22%3A%22https%3A%2F%2Fieeexplore.ieee.org%2Fdocument%2F10040717%2F%22%2C%22author%22%3A%5B%7B%22family%22%3A%22Frants%22%2C%22given%22%3A%22Vladimir%22%7D%2C%7B%22family%22%3A%22Agaian%22%2C%22given%22%3A%22Sos%22%7D%2C%7B%22family%22%3A%22Panetta%22%2C%22given%22%3A%22Karen%22%7D%5D%2C%22accessed%22%3A%7B%22date-parts%22%3A%5B%5B%222023%22%2C3%2C13%5D%5D%7D%2C%22issued%22%3A%7B%22date-parts%22%3A%5B%5B%222023%22%5D%5D%7D%7D%7D%5D%2C%22properties%22%3A%7B%7D%7D" ztype="zcitation">(<span class="citation-item"><a href="zotero://select/library/items/6ELGGPS7">Frants 等, 2023</a></span>)</span>

    在其设计的网络QCNN-H中指出采用四元数的卷积神经网络

    <span style="color: rgb(44, 62, 80)"><span style="background-color: rgb(255, 255, 255)">更好的保留了颜色信息，同时减小了网络参数、节约了内存资源</span></span>

## 常用模块
