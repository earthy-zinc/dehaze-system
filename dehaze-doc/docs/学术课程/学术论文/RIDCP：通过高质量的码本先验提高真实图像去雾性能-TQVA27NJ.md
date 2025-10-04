---
tag:
    - 有代码
    - 待阅
    - ⭐⭐⭐
    - CVPR2023
title: 'RIDCP: Revitalizing Real Image Dehazing via High-Quality Codebook Priors'
category:
    - 图像去雾
version: 6062
libraryID: 1
itemKey: TQVA27NJ

---
# RIDCP：通过高质量的码本先验提高真实图像去雾性能

## 摘要

本文从合成更真实的有雾图像和在网络中引入更健壮的先验知识这两个角度出发，提出了一种真实图像去雾方法。

1.  重新思考真实雾天图像质量退化原理，提出了考虑不同退化类型的气象学通道。
2.  提出了基于高质量码本先验的真实图像去雾网络(RIDCP)。首先大规模高质量数据集上预训练一个VQGAN网络得到离散的码本，封装高质量的先验知识(HQPs)。使用先验知识替换雾霾带来的负面影响后，具有新型归一化特征对齐模块的解码器就能够有效利用高质量特征产生干净的结果。通过一个可控的匹配操作重新计算特征与高质量先验匹配时的距离。这样有利于找到更好的对应物。用户可以根据喜好调整去雾强度。

## 介绍

作者进一步根据VQGAN的特点和统计结果设计了独特的实数域自适应策略，提出了可控的高质量先验匹配操作，在推理阶段对输入特征和高质量先验之间的距离添加设计好的权重来取代最近邻匹配。权重是由一个可控参数和高质量先验活动的统计分布间隙决定。通过调节这个可控参数，我们就能够改变高质量先验活动的分布。通过最小化概率分布的库尔贝克-莱布勒散度来获得最优的参数值，这个值也可以直观的反应为图像增强的程度。

*   首次在真实图像去雾中使用高质量的码本先验，提出了可控的高质量先验匹配操作，来克服合成域和真实域之间的差距，并产生可调节的结果
*   重新制定了真实雾天图像退化模型，提出了气象学退化通道来模拟现实中拍摄的雾天图像

## 相关工作

早期人们根据经验总结提出了大气散射模型，该模型能够估计图像中雾霾的生成状况。这时候人们就尝试估计大气散射模型中的参数来进行单幅图像去雾，但是这些基于经验观测得到的模型很难在不同场景中表现的很好。最明显的例子就是DCP，它在天空区域就是无法工作的。

随着深度学习的发展，利用数据驱动训练模型来去除雾霾的方法得到了广泛关注，许多研究开始采用卷积神经网络来估计大气散射模型中的参数，为了避免参数估计过程中的累积误差，人们就提出了端到端的网络，不再分”两步走“，由有雾图像直接估计生成无雾图像。这些方法在合成数据集上取得了优异的性能，但是在真实数据集上的表现却有待提高。

最近一些工作开始关注真实图像的去雾。其中一种路线就是利用生成对抗网络生成一种符合真实雾霾规律的雾霾图像，D4 就能够估计有雾图像的场景深度，生成不同雾霾浓度的有雾图像用做去雾模型的训练。但是生成对抗网络生成的结果容易产生伪影，这对下一步训练模型是有害的。另一种路线是通过损失函数或者网络架构引入有关去雾的先验知识。然而直接使用手工塑造的先验知识无法避免一些缺陷。本方法就利用数据生成流水线和潜在的高质量先验来可恶真实图像去雾方法现有的缺陷。

## 真实图像去雾的数据准备

要解决现实世界中的低级视觉任务，我们就重新设计数据生成流程。作者在合成用于训练去雾网络数据集时考虑可图像退化的各种因素，这些因素可以缓解合成数据和真实数据之间的差距。

雾霾图像的形成可以转换为如下数学公式：

![\<img alt="" data-attachment-key="ZGGWPS5V" width="831" height="96" src="attachments/ZGGWPS5V.png" ztype="zimage">](attachments/ZGGWPS5V.png)

*   γ∈\[ 1.5、3.0]为亮度调整因子，N为高斯噪声分布。这两个分量可以模拟出雾霾天气中频繁出现的恶劣光照条件。
*   作为退化模型中的关键参数，我们采用深度估计算法来估计深度图d (x)，并使用β∈\[ 0.3、1.5]来控制雾霾密度。
*   为了获得多样化的雾天图像，作者考虑了大气光的颜色偏差，通过一个三通道矢量 ΔA∈\[-0.025, 0.025]来实现。A的范围在\[0.25，1.0]范围内。
*   我们观察到去雾算法放大了JPEG伪影，应在去雾的同时去除此类伪影。JPEG ( · )表示对结果进行JPEG压缩。

## 方法

核心思想是采用离散码本，将高质量先验引入到去雾网络中。整体训练分为两个阶段。

第一阶段，在一些高质量数据上预训练一个VQGAN，道德带有高质量先验的潜在离散码本和对一个的解码器。

第二个阶段，基于预训练的VQGAN的RIDCP在合成流水线中生成的雾天图像中进行训练。为了帮助网络找到更精确的代码，我们在高质量图像上提出了一种基于代码激活分布的可控调整特征匹配策略

### 高质量码本

VQGAN：给定一个高质量的图像块作为VQGAN的编码器的输入，对应输出潜在特征，然后将潜在特征中的每个像素匹配到码本中最近的高质量先验上。从而的到了一个离散的数据表示。

![\<img alt="" data-attachment-key="SJC7ZF3X" width="670" height="99" src="attachments/SJC7ZF3X.png" ztype="zimage">](attachments/SJC7ZF3X.png)

为了了解码本中的高质量先验的潜力。作者对预训练后的VQGAN重建的图像结果进行了观察。实验证明该模型能够去除薄雾并且恢复图像颜色。作者认为以匹配的方式使用高质量先验能够替换退化的特征，从而帮助其跳转到高质量的域中。但是该去雾能力难以匹配到正确的代码，由于矢量量化阶段信息缺失，会产生一些失真纹理。下一步工作是训练一个能够帮助先验匹配的编码器E和一个能够利用HQPs重建特征的解码器G。

### 基于特征匹配的图像去雾

随后我们将图像去雾分解为两个子任务。一是将编码器输出的离散特征匹配到码本中的正确编码；二是去除纹理失真。

#### 用于匹配高质量先验的编码器

借鉴SwinIR在图像复原领域的强大的特征提取能力来设计编码器。其中浅层特征是由残差层和四倍下采样特征组成。随后为用作特征提取的4个残差Swin Transformer块RSTB。

> ![\<img alt="" data-attachment-key="CRESDNPA" width="882" height="368" src="attachments/CRESDNPA.png" ztype="zimage">](attachments/CRESDNPA.png)
>
> ![\<img alt="" data-attachment-key="TUQ8M88V" width="808" height="335" src="attachments/TUQ8M88V.png" ztype="zimage">](attachments/TUQ8M88V.png)
>
> 其中残差Swin Transformer块（RSTB）如图所示，内部由Swin Transformer Layer（STL）组成。而Neighborhood Attention Transformer已经在图<span style="color: rgb(18, 18, 18)"><span style="background-color: rgb(255, 255, 255)">像分类和下游视觉任务(包括目标检测和语义分割)中被证明有效。注意到STL和NAT block之间只有一个区别，那就是NAT block将MSA</span></span> (Multi-head Self Attention) <span style="color: rgb(18, 18, 18)"><span style="background-color: rgb(255, 255, 255)">替换为了NA（Neighborhood Attention）那么我们直接将STL替换为NAT block，能否提高模型性能？</span></span>

#### 解码器

提出了归一化特征对齐(Normalized Feature Alignment，NFA)来帮助解码器解码离散特征。首先由于离散的向量带来的信息损失会降低结果的精准度。本文解决方案就是在高质量先验匹配之前通过特征拟合来消除信息损失。具体的做法就是在特征的第i层使用可变形卷积将来自解码器Gvq的特征和解码器G的特征对齐。

![\<img alt="" data-attachment-key="HKNYSEKE" width="812" height="81" src="attachments/HKNYSEKE.png" ztype="zimage">](attachments/HKNYSEKE.png)

### 可控的高质量先验匹配操作

因为合成数据和真实数据之间的领域差距，在个别真实图像中会存在颜色饱和度较低的现象，因为这样的差距，模型就难以找到离散变量对应的码本，也就是对应的高质量先验。从而难以生成生动的结果。为了验证遮挡合成数据和真实数据之间存在的领域差距，作者做出了如下实验：随机拍摄200张高质量图像作为预训练的VQGAN的输入，计算码本中每个编码的激活频率，随后将有雾图像送到去雾网络中计算激活频率，实验结果证明具有明显的分布偏移

这证明了缩小合成数据域和真实数据域差距的域自适应和域迁移仍有必要的用处，而高质量先验在去雾网络中还未被充分利用。

### 通过重新计算距离来实现可以控制的匹配

当遇到真实的有雾图像时，去雾的一个关键环节就是匹配到更适合的高质量先验。影响高质量先验匹配情况的因素有两个，一个是编码器输出的离散向量，另一个是匹配操作。

在没有参考图像的情况下很难重新训练编码器获得更加拟合先验的编码器，因此我们可以从匹配操作的精确度入手。通过在匹配阶段分配不同的权重来重新计算距离。

> 找到最接近的码本中的编码，首先有雾图像通过编码器的到的离散编码Z<sub>haze</sub>和无雾图像得到的离散编码Z<sub>clear </sub>或许就有本质的不同，使用Z<sub>haze</sub>来找Z<sub>clear</sub>对应的码本编码，大概率是匹配不到的，不过可以通过计算他们距离，来从码本中找最接近的编码。
>
> 1.  在编码器中添加一个有雾图像和无雾图像之间的像素损失、感知损失。我们通过编码器，在训练有雾图像时加入有雾图像和无雾图像之间的损失。迫使编码器通过学习将有雾图像的离散编码向无雾图像的离散编码靠近。
>
> 2.  在将有雾图像的离散编码和清晰无雾图像的码本进行匹配时，通过一种手段，让有雾图像的离散编码尽量能够匹配到对应无雾图像的应该匹配到的码本编码。方法是通过计算有雾图像离散编码和码本中各个编码之间的距离找到距离最小的码本编码。然后通过一个权重来调节最终计算出来的距离。
>
> 3.  如果码本中的编码在无雾图像中被激活了，但是在有雾图像中未被激活。则将该码本编码权重提高。使其在计算距离时得到的结果更小。将权重提高多少呢。我们计算有雾图像激活频率和无雾图像激活频率之差，差距越大，则说明越需要调整。也就意味着需要越少的激活，则该项权重需要降低。
>
> 4.  ![\<img alt="" data-attachment-key="KCF5MMDH" width="396" height="87" src="attachments/KCF5MMDH.png" ztype="zimage">](attachments/KCF5MMDH.png)最终我们选择该函数作为权重，接下来要解决的问题权重设置的大小问题。如果权重设置太小，会导致不必要的激活，而权重设置太大会无法激活该码本编码。因此我们需要一个超参数a来辅助控制权重的大小。
>
> 5.  超参数a的取值。将编码器所得到的编码和清晰无雾图像的编码之间的差距用两个概率分布之间的差异来表示。这样，两个不同的域，无雾图像所在的领域和有雾图像所在的领**域适应**问题就转化为秋姐一个最优参数a，使得P
>
>     <sub>clear</sub>
>
>     ( x = z
>
>     <sub>k</sub>
>
>     )和P
>
>     <sub>haze</sub>
>
>     ( x = z
>
>     <sub>k</sub>
>
>     \| α)前向库尔贝克-莱布勒散度最小
>
> 6.  ~~（存疑）利用VQGAN的Transformer模型自回归地利用编码器输出的部分有雾图像的离散编码预测下一个离散编码。首先将得到的有雾图像的编码展平，随机将一些已有的离散编码替换为随机生成的相同维度的变量。然后送入Transformer学习，重构出被替换的那些离散编码。~~

![\<img alt="" data-attachment-key="623YWD63" width="683" height="99" src="attachments/623YWD63.png" ztype="zimage">](attachments/623YWD63.png)

其中F函数是根据频率差生成的权重函数，通过参数α。

1.  较高的fk意味着较低的激活，因此F和fk单调，从而两者趋势一致
2.  F( 0 , α)≡1，使得在清晰和模糊数据上具有相同频率的HQPs不被调整。
3.  调节程度可由α单调控制

因此F这里我们选择使用指数函数：

![\<img alt="" data-attachment-key="BV62YIKS" width="380" height="91" src="attachments/BV62YIKS.png" ztype="zimage">](attachments/BV62YIKS.png)

### 推荐的值α

最终的目标是找到一个合适的α使得网络适应真实的域，我们可以使用

![\<img alt="" data-attachment-key="MU74T8FQ" width="770" height="473" src="attachments/MU74T8FQ.png" ztype="zimage">](attachments/MU74T8FQ.png)

计算得到<span class="highlight" data-annotation="%7B%22attachmentURI%22%3A%22http%3A%2F%2Fzotero.org%2Fusers%2F10046823%2Fitems%2FCZG79VXE%22%2C%22pageLabel%22%3A%2222286%22%2C%22position%22%3A%7B%22pageIndex%22%3A4%2C%22rects%22%3A%5B%5B323.84%2C86.723%2C368.924%2C96.586%5D%5D%7D%2C%22citationItem%22%3A%7B%22uris%22%3A%5B%22http%3A%2F%2Fzotero.org%2Fusers%2F10046823%2Fitems%2FBJNHXL93%22%5D%2C%22locator%22%3A%2222286%22%7D%7D" ztype="zhighlight"><a href="zotero://open-pdf/library/items/CZG79VXE?page=5">“α =21.25”</a></span> <span class="citation" data-citation="%7B%22citationItems%22%3A%5B%7B%22uris%22%3A%5B%22http%3A%2F%2Fzotero.org%2Fusers%2F10046823%2Fitems%2FBJNHXL93%22%5D%7D%5D%2C%22properties%22%3A%7B%7D%7D" ztype="zcitation">(<span class="citation-item"><a href="zotero://select/library/items/BJNHXL93">Wu 等, 2023</a></span>)</span>

## 扩展：VQ-VAE——首个提出码本机制的生成模型

[轻松理解 VQ-VAE：首个提出 codebook 机制的生成模型 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/633744455)

VQ-VAE利用码本机制将图像编码成离散向量。为什么要这样做？首先要了解自编码器（Autoencoder，AE），自编码器是一类能够把图片压缩成较短向量的生成模型，模型包含一个解码器和编码器。在训练时，输入图像会被编码为一个较短的向量，在被解码为另一幅长得差不多的图像，网络学习的目标就是让重建出来的图像和原图像尽可能相似。解码器可以把向量解码成图像，换一个角度来说，解码器就是一个图像生成模型，但是自编码器中，编码器编码出来的向量空间是不完整的，也就是说，解码器值认识经过编码器编码出来的向量，而不认识其他向量。如果把自己随机生成的向量输入给解码器，解码器不能生成有意义的图片，因此自编码器不能够随机生成图片，只能起到图像压缩的作用。

但是只要自编码器的编码空间比较规整，符合某个简单的数学分布。那我们就可以从这个分布中随机采样。再让解码器根据这个向量来完成随机图片生成，VAE就是改进版的AE（自编码器）。VAE采用一些巧妙的方法约束编码向量，是的编码向量满足标准的正太分布，这样训练出来的编码器-解码器对中的解码器不仅能够认识编码器编码的向量，还可以认识来自其他标准正态分布的向量。训练完成之后，我们就可以扔掉编码器，用来自标准正太分布的随机向量和解码器完成随机图像的生成了。但是VAE编码出来的是连续向量，生成的图片都不太好看，VQ-VAE认为，VAE之所以生成的图片质量不高，是因为图片被编码成了连续向量，但是把图片编码成离散向量会更加自然。

但是神经网络会默认输入满足一个连续的分布，而不善于处理离散的向量。为了解决这一问题，学者们借鉴了自然语言处理中对单词的处理方法。也就是增加一个词嵌入层，把每个输入的单词映射到一个独一无二的连续向量上。那么这个嵌入层就叫做码本（cookbook）

VQ-VAE的整个工作流程如下：

1.  训练VQ-VAE的编码器和解码器，使得VQ-VAE能够把图像编码成离散向量，也能够把离散的向量解码为图像
2.  训练PixelCNN，让他学习如何生成离散向量
3.  随机采样时，先用PixelCNN采样出离散向量，然后再用VQ-VAE把离散向量解码成最终图像。

### VQ-VAE 设计细节

#### 输出离散编码

编码器解码器架构中编码器会将编码输入图片为张量，我们如果想要获得编码器输出的离散编码，则需要通过一个概率分布计算来获得离散编码，但是紧接着在输入解码器前，又需要将离散编码转回张量。

而VQ-VAE使用了最近邻算法来关联编码器的输出和解码器的输入。首先计算向量和嵌入空间k个向量每个向量的距离，再对距离数组取argmin，求出最近的下标，最后用下标去嵌入空间里取向量。

#### 优化编码器和解码器

编码器和解码器整体优化目标就是原图像和目标图像的重建误差

#### 优化嵌入空间

略

VQ-VAE是一个把图像编码成离散向量的图像压缩模型，为了让神经网络理解离散编码，VQ-VAE借鉴了NLP思想，让每个离散编码值对应一个嵌入，（也就是离散变量对应的唯一连续变量值之间的对应关系层）所有嵌入都存放在一个嵌入空间，称为码本。VQ-VAE编码器输出是若干个假嵌入，也就是说，这个编码器输出的离散变量和连续变量值之间的对应关系，并不是真实图像应该具有的离散变量和连续变量值之间的对应关系，这个关系是编码器本次输入得到的。换言之，假嵌入是编码器本次输入得到的结果。而在嵌入空间内这些假嵌入会被替换为嵌入空间内的真嵌入。输入进编码器中。这些真嵌入也可以称作是编码器通过大量训练得到的高质量码本。

VQ-VAE的优化目标分为两部分，重建误差和嵌入空间误差，重建误差为输入图片和重建图片之间的均方误差。为了让梯度反向传播，也就是从解码器传回编码器，作者使用了一种巧妙的停止梯度算子，让正向传播和反向传播按照不同的方式计算，嵌入空间的误差为真嵌入和编码器输出的假嵌入之间的均方误差。为了让嵌入和编码器用不同的速度优化，作者使用了停止梯度算子，把嵌入的更新和编码器的更新分开计算。

训练完成后，为了实现随机图像的生成，需要对VQ-VAE的离散分布进行采样，再把采样出来的离散变量对应的嵌入输入进解码器。而离散变量的获取是一个关键。VQ-VAE使用了Pixel来采样离散分布。但是这个离散分布并不只能使用PixelCNN，当然也可以采用diffusion扩散模型，Transformer等。采用diffusion扩散模型的典型例子就是Stable Diffusion，采用Transformer的例子就是VQGAN。

## VQGAN

VQGAN特点是使用码本来离散编码模型的获取的特征，并且使用Transformer作为编码的生成工具。而VQ-GAN通过引入对抗监督进行离散向量的学习，也就是采用PatchGAN的判别器。进一步提高了最终生成结果的质量。

### 为什么要采用离散向量编码特征

在VQVAE中的VQ（vector quatization，向量离散化），编码出来的每一维的特征都是离散的数值。这样做是符合自然界的一些模态的。因为自然界中的事物直接的差别是很大的。不是连续的变化的。一张输入的RGB三通道图片，通过编码器后会得到中间特征，普通的编码器会将这些特征直接送到解码器中重建，而自编码器VQVAE会将特征进一步离散化编码。具体的做法就是预先生成一个离散数值的码本，在中间特征中每一个编码位置中寻找距离离散码本最近的值，生成具有相同维度的变量。进一步离散编码的过程表示为：

![\<img alt="" data-attachment-key="6TGPBDJC" width="535" height="96" src="attachments/6TGPBDJC.png" ztype="zimage">](attachments/6TGPBDJC.png)

这样一来就可以将离散化的特征使用解码器进行解码。

在训练过程中，模型会逐步训练将输入图像训练的和高质量图像尽可能接近。整个训练需要同时进行三个子模块，分别是编码器、解码器、码本。其中自监督损失可定义为：

![\<img alt="" data-attachment-key="IUE92QCV" width="756" height="69" src="attachments/IUE92QCV.png" ztype="zimage">](attachments/IUE92QCV.png)

<span style="color: rgb(18, 18, 18)"><span style="background-color: rgb(255, 255, 255)">其中</span></span>![\<img alt="" data-attachment-key="K3B2SZEB" width="184" height="41" src="attachments/K3B2SZEB.png" ztype="zimage">](attachments/K3B2SZEB.png)%3Cspan%20style%3D%22color%3A%20rgb(18%2C%2018%2C%2018)%22%3E%3Cspan%20style%3D%22background-color%3A%20rgb(255%2C%20255%2C%20255)%22%3E%E4%B8%BA%E9%87%8D%E5%BB%BA%E6%8D%9F%E5%A4%B1%EF%BC%88reconstruction%20loss%EF%BC%89%EF%BC%8C%E8%80%8C%C2%A0%3C%2Fspan%3E%3C%2Fspan%3E%3Cspan%20style%3D%22background-color%3A%20rgb(255%2C%20255%2C%20255)%22%3Esg%E2%81%A1%5B%C2%B7%5D%3C%2Fspan%3E%3Cspan%20style%3D%22color%3A%20rgb(18%2C%2018%2C%2018)%22%3E%3Cspan%20style%3D%22background-color%3A%20rgb(255%2C%20255%2C%20255)%22%3E%C2%A0%E4%B8%BA%E6%A2%AF%E5%BA%A6%E7%BB%88%E6%AD%A2%E6%93%8D%E4%BD%9C%EF%BC%88stop-gradient%20operation%EF%BC%89%E3%80%82%E4%B9%8B%E6%89%80%E4%BB%A5%E8%A6%81%E5%9C%A8E(x)%E5%92%8C%20zq%20%E4%B9%8B%E9%97%B4%E5%8A%A0%E5%85%A5%3C%2Fspan%3E%3C%2Fspan%3E%3Cspan%20style%3D%22background-color%3A%20rgb(255%2C%20255%2C%20255)%22%3Esg%E2%81%A1%5B%C2%B7%5D%3C%2Fspan%3E%3Cspan%20style%3D%22color%3A%20rgb(18%2C%2018%2C%2018)%22%3E%3Cspan%20style%3D%22background-color%3A%20rgb(255%2C%20255%2C%20255)%22%3E%E6%93%8D%E4%BD%9C%EF%BC%8C%E6%98%AF%E5%9B%A0%E4%B8%BA%E6%88%91%E4%BB%AC%E5%9C%A8%E8%BF%99%E4%B8%A4%E4%B8%AA%E7%89%B9%E5%BE%81%E9%97%B4%E8%BF%9B%E8%A1%8C%E4%BA%86%E7%A6%BB%E6%95%A3%E5%8C%96%E8%BD%AC%E6%8D%A2%EF%BC%8C%E5%A6%82%E6%9E%9C%E7%9B%B4%E6%8E%A5%E8%BF%9B%E8%A1%8CL2%E6%8D%9F%E5%A4%B1%E8%AE%A1%E7%AE%97%E7%9A%84%E8%AF%9D%EF%BC%8C%E4%BC%9A%E5%AF%BC%E8%87%B4%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E6%A2%AF%E5%BA%A6%E4%B8%8D%E8%83%BD%E5%9B%9E%E4%BC%A0%E3%80%82%E5%9B%A0%E6%AD%A4%E5%88%86%E5%88%AB%E5%B0%86%E4%B8%A4%E4%B8%AA%E7%89%B9%E5%BE%81%E7%9A%84%E6%A2%AF%E5%BA%A6%E7%BB%88%E6%AD%A2%EF%BC%8C%E5%B0%86%3C%2Fspan%3E%3C%2Fspan%3E%3Cspan%20style%3D%22background-color%3A%20rgb(255%2C%20255%2C%20255)%22%3Ezq%3C%2Fspan%3E%3Cspan%20style%3D%22color%3A%20rgb(18%2C%2018%2C%2018)%22%3E%3Cspan%20style%3D%22background-color%3A%20rgb(255%2C%20255%2C%20255)%22%3E%E7%9A%84%E6%A2%AF%E5%BA%A6%E7%9B%B4%E6%8E%A5%E5%A4%8D%E5%88%B6%E5%88%B0%3C%2Fspan%3E%3C%2Fspan%3E%3Cspan%20style%3D%22background-color%3A%20rgb(255%2C%20255%2C%20255)%22%3Ez%5E%3C%2Fspan%3E%3Cspan%20style%3D%22color%3A%20rgb(18%2C%2018%2C%2018)%22%3E%3Cspan%20style%3D%22background-color%3A%20rgb(255%2C%20255%2C%20255)">上，这两项损失分别训练了编码器和码本。

对于判别器来说，损失函数大致可以表示为

![\<img alt="" data-attachment-key="IGQLPSRY" width="564" height="64" src="attachments/IGQLPSRY.png" ztype="zimage">](attachments/IGQLPSRY.png)

在VQGAN中，Transformer主要是用作编码生成器使用用来生成高分辨率图像。迁移到VQGAN中，可以理解为先预测一个码本中的一个值，在一步步的通过预测好的值推断下一个值。

具体的训练过程是先输入图片，经过编码器得到中间特征变量，在经过码本离散编码得到离散变量。之后为了训练Transformer，将离散变量平展，得到降低了维度的变量。随机将其中一部分的编码值替换为随机生成的相同维度的向量。也就是在特征中加入噪声。用来提高Transformer的泛化能力。

![\<img alt="" data-attachment-key="VQNAX82T" width="854" height="484" src="attachments/VQNAX82T.png" ztype="zimage">](attachments/VQNAX82T.png)

[v2-0ff3d5ed6cdce24603c2e58091e3341c\_b.webp (854×484) (zhimg.com)](https://pic1.zhimg.com/v2-0ff3d5ed6cdce24603c2e58091e3341c_b.webp)

VQGAN模型是需要经过两步训练的，第一步是通过自监督学习训练解码器编码器和码本。第二部在已经训练好的编码器和码本之上，将码本中的部分编码随机替换，然后使用Transformer重建这部分编码来提高模型的泛化能力。

## Stable Diffusion
