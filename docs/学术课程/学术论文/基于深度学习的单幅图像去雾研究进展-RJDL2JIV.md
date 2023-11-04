---
tag: []
title: 基于深度学习的单幅图像去雾研究进展
category:
    - 文献综述
version: 4769
libraryID: 1
itemKey: RJDL2JIV

---
# 基于深度学习的单幅图像去雾研究进展

一、基于物理模型和先验知识去雾方法：

*   2011

    <span class="highlight" data-annotation="%7B%22attachmentURI%22%3A%22http%3A%2F%2Fzotero.org%2Fusers%2F10046823%2Fitems%2F9SIX2C8F%22%2C%22pageLabel%22%3A%223%22%2C%22position%22%3A%7B%22pageIndex%22%3A2%2C%22rects%22%3A%5B%5B286.316%2C699.917%2C296.116%2C711.171%5D%2C%5B62.462%2C685.383%2C219.717%2C697.338%5D%5D%7D%2C%22citationItem%22%3A%7B%22uris%22%3A%5B%22http%3A%2F%2Fzotero.org%2Fusers%2F10046823%2Fitems%2FSW4N67ZU%22%5D%2C%22locator%22%3A%223%22%7D%7D" ztype="zhighlight"><a href="zotero://open-pdf/library/items/9SIX2C8F?page=3">“暗通道先验（Dark Channel Prior，DCP）”</a></span>

    <span class="highlight" data-annotation="%7B%22attachmentURI%22%3A%22http%3A%2F%2Fzotero.org%2Fusers%2F10046823%2Fitems%2F9SIX2C8F%22%2C%22pageLabel%22%3A%223%22%2C%22position%22%3A%7B%22pageIndex%22%3A2%2C%22rects%22%3A%5B%5B222.018%2C685.383%2C296.116%2C697.338%5D%2C%5B62.462%2C670.849%2C296.117%2C682.104%5D%2C%5B62.462%2C656.23%2C282.228%2C667.485%5D%5D%7D%2C%22citationItem%22%3A%7B%22uris%22%3A%5B%22http%3A%2F%2Fzotero.org%2Fusers%2F10046823%2Fitems%2FSW4N67ZU%22%5D%2C%22locator%22%3A%223%22%7D%7D" ztype="zhighlight"><a href="zotero://open-pdf/library/items/9SIX2C8F?page=3">“He 等人［5］统计了 大量的无雾图像，发现在图像的大部分区域内，存在一 些像素点在至少一个颜色通道中具有非常低的值 .”</a></span>

    他们将这些颜色通道称为暗通道。

    <span class="highlight" data-annotation="%7B%22attachmentURI%22%3A%22http%3A%2F%2Fzotero.org%2Fusers%2F10046823%2Fitems%2F9SIX2C8F%22%2C%22pageLabel%22%3A%223%22%2C%22position%22%3A%7B%22pageIndex%22%3A2%2C%22rects%22%3A%5B%5B82.462%2C599.951%2C296.116%2C611.205%5D%2C%5B62.462%2C585.417%2C136.161%2C596.671%5D%5D%7D%2C%22citationItem%22%3A%7B%22uris%22%3A%5B%22http%3A%2F%2Fzotero.org%2Fusers%2F10046823%2Fitems%2FSW4N67ZU%22%5D%2C%22locator%22%3A%223%22%7D%7D" ztype="zhighlight"><a href="zotero://open-pdf/library/items/9SIX2C8F?page=3">“据此预测图像透射图，利用大气光散射模型得到 最终的去雾结果 .”</a></span>

    <span class="highlight" data-annotation="%7B%22attachmentURI%22%3A%22http%3A%2F%2Fzotero.org%2Fusers%2F10046823%2Fitems%2F9SIX2C8F%22%2C%22pageLabel%22%3A%223%22%2C%22position%22%3A%7B%22pageIndex%22%3A2%2C%22rects%22%3A%5B%5B82.462%2C570.883%2C296.117%2C582.138%5D%2C%5B62.462%2C556.349%2C284.718%2C567.604%5D%5D%7D%2C%22citationItem%22%3A%7B%22uris%22%3A%5B%22http%3A%2F%2Fzotero.org%2Fusers%2F10046823%2Fitems%2FSW4N67ZU%22%5D%2C%22locator%22%3A%223%22%7D%7D" ztype="zhighlight"><a href="zotero://open-pdf/library/items/9SIX2C8F?page=3">“但是，暗通道先验对大片天空等区域不够鲁棒，当 雾霾图像存在大片天空区域时，处理效果并不理想 .”</a></span>

*   2016 NL

*   2021 IDE

   

2\. 有监督和弱监督的神经网络方法：

2016 DehazeNet \[36], <span class="highlight" data-annotation="%7B%22attachmentURI%22%3A%22http%3A%2F%2Fzotero.org%2Fusers%2F10046823%2Fitems%2F9SIX2C8F%22%2C%22pageLabel%22%3A%223%22%2C%22position%22%3A%7B%22pageIndex%22%3A2%2C%22rects%22%3A%5B%5B193.717%2C265.675%2C296.116%2C276.929%5D%2C%5B62.462%2C251.141%2C296.116%2C262.396%5D%2C%5B62.462%2C236.608%2C296.115%2C247.862%5D%2C%5B62.462%2C222.074%2C296.116%2C233.582%5D%2C%5B62.462%2C206.819%2C296.116%2C218.073%5D%2C%5B62.462%2C192.285%2C226.16%2C203.539%5D%5D%7D%2C%22citationItem%22%3A%7B%22uris%22%3A%5B%22http%3A%2F%2Fzotero.org%2Fusers%2F10046823%2Fitems%2FSW4N67ZU%22%5D%2C%22locator%22%3A%223%22%7D%7D" ztype="zhighlight"><a href="zotero://open-pdf/library/items/9SIX2C8F?page=3">“该网络包含结合传统手 工特征的特征提取层、多尺度映射层、局部极值层以及 非线性回归层，通过学习雾霾退化模型中的介质透射 率 t(x) 进行去雾 . 计算时，假设大气光值 A 为固定经验 值与实际大气光值之间会有差异，通过退化模型求解 得到的去雾图像也相应地会产生偏差 .”</a></span> <span class="citation" data-citation="%7B%22citationItems%22%3A%5B%7B%22uris%22%3A%5B%22http%3A%2F%2Fzotero.org%2Fusers%2F10046823%2Fitems%2FSW4N67ZU%22%5D%7D%5D%2C%22properties%22%3A%7B%7D%7D" ztype="zcitation">(<span class="citation-item"><a href="zotero://select/library/items/SW4N67ZU">贾童瑶 等, 2023</a></span>)</span>

AOD-Net \[37], “Li 等人［15］在<span class="highlight" data-annotation="%7B%22attachmentURI%22%3A%22http%3A%2F%2Fzotero.org%2Fusers%2F10046823%2Fitems%2F9SIX2C8F%22%2C%22pageLabel%22%3A%222%22%2C%22position%22%3A%7B%22pageIndex%22%3A1%2C%22rects%22%3A%5B%5B433.818%2C441.218%2C519.572%2C452.473%5D%5D%7D%2C%22citationItem%22%3A%7B%22uris%22%3A%5B%22http%3A%2F%2Fzotero.org%2Fusers%2F10046823%2Fitems%2FSW4N67ZU%22%5D%2C%22locator%22%3A%222%22%7D%7D" ztype="zhighlight"><a href="zotero://open-pdf/library/items/9SIX2C8F?page=2">“大 气 光 单 散 射 模 型”</a></span>的基础上， 将介质透射率 t(x) 和大气光值 A 统一到一个变量 K(x) 中，只需要求解一个 K(x)就可以实现图像增强 .” <span class="citation" data-citation="%7B%22citationItems%22%3A%5B%7B%22uris%22%3A%5B%22http%3A%2F%2Fzotero.org%2Fusers%2F10046823%2Fitems%2FSW4N67ZU%22%5D%7D%5D%2C%22properties%22%3A%7B%7D%7D" ztype="zcitation">(<span class="citation-item"><a href="zotero://select/library/items/SW4N67ZU">贾童瑶 等, 2023</a></span>)</span>“，K(x) 由一体化去雾网络（All-in-One De⁃ hazing Network，AOD-Net）来求解，网络仅包含 5 个卷积 层，计算复杂度低，去雾效果有了进一步的提升 .”

2019 GCANet \[39],

2019 WaveletUnet \[33],

EPDN \[40], <span class="highlight" data-annotation="%7B%22attachmentURI%22%3A%22http%3A%2F%2Fzotero.org%2Fusers%2F10046823%2Fitems%2F9SIX2C8F%22%2C%22pageLabel%22%3A%225%22%2C%22position%22%3A%7B%22pageIndex%22%3A4%2C%22rects%22%3A%5B%5B82.462%2C526.534%2C296.116%2C538.488%5D%2C%5B62.462%2C512.076%2C296.183%2C523.331%5D%2C%5B62.462%2C497.619%2C296.116%2C508.873%5D%2C%5B62.462%2C483.161%2C296.116%2C494.416%5D%2C%5B62.462%2C468.704%2C296.117%2C479.958%5D%2C%5B62.462%2C454.246%2C296.117%2C465.5%5D%2C%5B62.462%2C439.788%2C296.116%2C451.043%5D%2C%5B62.462%2C425.331%2C296.116%2C436.585%5D%2C%5B62.462%2C410.873%2C296.116%2C422.128%5D%2C%5B62.462%2C396.416%2C191.16%2C407.67%5D%5D%7D%2C%22citationItem%22%3A%7B%22uris%22%3A%5B%22http%3A%2F%2Fzotero.org%2Fusers%2F10046823%2Fitems%2FSW4N67ZU%22%5D%2C%22locator%22%3A%225%22%7D%7D" ztype="zhighlight"><a href="zotero://open-pdf/library/items/9SIX2C8F?page=5">“Qu 等人［29］将图像去雾问题简化为图像到图像的 转 换 问 题 ，并 提 出 了 增 强 型 去 雾 网 络（Enhanced Pix2pix Dehazing Network，EPDN），在 不 依 赖 物 理 散 射 模型的情况下生成无雾图像 . EPDN 由多分辨率生成器 模块、增强器模块和多尺度判别器模块组成 . 多分辨率 生成器对雾霾图像在两个尺度上进行特征提取；增强 模块用于恢复去雾图像的颜色和细节信息；多尺度判 别器用于对生成的去雾结果进行鉴别 . 虽然算法在主 客观结果上都有了一定提升，但是对真实雾霾图像进 行处理时，会存在过增强现象 .”</a></span> <span class="citation" data-citation="%7B%22citationItems%22%3A%5B%7B%22uris%22%3A%5B%22http%3A%2F%2Fzotero.org%2Fusers%2F10046823%2Fitems%2FSW4N67ZU%22%5D%7D%5D%2C%22properties%22%3A%7B%7D%7D" ztype="zcitation">(<span class="citation-item"><a href="zotero://select/library/items/SW4N67ZU">贾童瑶 等, 2023</a></span>)</span>

RefineDNet \[43], <span class="highlight" data-annotation="%7B%22attachmentURI%22%3A%22http%3A%2F%2Fzotero.org%2Fusers%2F10046823%2Fitems%2F9SIX2C8F%22%2C%22pageLabel%22%3A%226%22%2C%22position%22%3A%7B%22pageIndex%22%3A5%2C%22rects%22%3A%5B%5B336.163%2C281.416%2C549.817%2C293.371%5D%2C%5B316.163%2C266.985%2C549.817%2C278.239%5D%2C%5B316.163%2C252.553%2C549.817%2C263.808%5D%2C%5B316.163%2C238.122%2C549.817%2C249.376%5D%2C%5B316.163%2C223.69%2C549.816%2C234.945%5D%2C%5B316.163%2C209.258%2C549.817%2C220.513%5D%2C%5B316.163%2C194.827%2C549.817%2C206.081%5D%2C%5B316.163%2C180.395%2C339.863%2C191.65%5D%5D%7D%2C%22citationItem%22%3A%7B%22uris%22%3A%5B%22http%3A%2F%2Fzotero.org%2Fusers%2F10046823%2Fitems%2FSW4N67ZU%22%5D%2C%22locator%22%3A%226%22%7D%7D" ztype="zhighlight"><a href="zotero://open-pdf/library/items/9SIX2C8F?page=6">“Zhao 等人［49］将基于先验和基于学习的方法结合起 来，提出了一个两阶段弱监督去雾框架 RefineDNet，以 发挥二者的优势 . 在第一阶段，采用暗通道先验恢复可 见性；在第二阶段，细化第一阶段的初步去雾结果，通 过非成对的雾霾和清晰图像的对抗学习来提高真实 性 . 为了获得更优越的结果，还提出了一种有效的感知 融合策略来混合不同的去雾输出，可以有效提升去雾 效果 .”</a></span> <span class="citation" data-citation="%7B%22citationItems%22%3A%5B%7B%22uris%22%3A%5B%22http%3A%2F%2Fzotero.org%2Fusers%2F10046823%2Fitems%2FSW4N67ZU%22%5D%7D%5D%2C%22properties%22%3A%7B%7D%7D" ztype="zcitation">(<span class="citation-item"><a href="zotero://select/library/items/SW4N67ZU">贾童瑶 等, 2023</a></span>)</span>

ZID，<span class="highlight" data-annotation="%7B%22attachmentURI%22%3A%22http%3A%2F%2Fzotero.org%2Fusers%2F10046823%2Fitems%2F9SIX2C8F%22%2C%22pageLabel%22%3A%227%22%2C%22position%22%3A%7B%22pageIndex%22%3A6%2C%22rects%22%3A%5B%5B82.462%2C656.785%2C296.116%2C668.739%5D%2C%5B62.462%2C642.362%2C296.116%2C653.889%5D%2C%5B62.462%2C627.939%2C296.116%2C639.193%5D%2C%5B62.462%2C613.516%2C296.117%2C624.77%5D%2C%5B62.462%2C599.093%2C296.116%2C610.347%5D%2C%5B62.462%2C584.67%2C296.115%2C595.924%5D%2C%5B62.462%2C570.247%2C296.116%2C581.501%5D%2C%5B62.462%2C555.824%2C231.16%2C567.078%5D%5D%7D%2C%22citationItem%22%3A%7B%22uris%22%3A%5B%22http%3A%2F%2Fzotero.org%2Fusers%2F10046823%2Fitems%2FSW4N67ZU%22%5D%2C%22locator%22%3A%227%22%7D%7D" ztype="zhighlight"><a href="zotero://open-pdf/library/items/9SIX2C8F?page=7">“Li 等人［50］提出了一种基于零样本学习的图像去雾 方法（Zero-shot Image Dehazing，ZID）. 利用解耦思想根 据先验模型将雾霾图像视为清晰图像、透射图和大气 光的融合 . 利用单个雾霾图像进行学习和推理，不遵循 在大规模数据集上训练深度模型的传统范式 . 这能够 避免数据收集和使用合成雾霾图像来解决现实世界图 像的域转移问题 . 但是，对一张图像需要 500 轮次迭 代，而且去雾结果中常常存在大量伪像 .”</a></span> <span class="citation" data-citation="%7B%22citationItems%22%3A%5B%7B%22uris%22%3A%5B%22http%3A%2F%2Fzotero.org%2Fusers%2F10046823%2Fitems%2FSW4N67ZU%22%5D%7D%5D%2C%22properties%22%3A%7B%7D%7D" ztype="zcitation">(<span class="citation-item"><a href="zotero://select/library/items/SW4N67ZU">贾童瑶 等, 2023</a></span>)</span>
