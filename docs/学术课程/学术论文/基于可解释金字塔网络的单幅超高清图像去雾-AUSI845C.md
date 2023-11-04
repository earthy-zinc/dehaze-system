---
tag:
    - 有代码
    - 待阅
    - 4KDehazing
title: 'Single UHD Image Dehazing via Interpretable Pyramid Network'
category:
    - 图像去雾
version: 4770
libraryID: 1
itemKey: AUSI845C

---
# 基于可解释金字塔网络的单幅超高清图像去雾

*   网络模型方面：受到图像处理领域泰勒定理的无限逼近原理和拉普拉斯金字塔的启发，在构建网络模型层面，建立了一种能够实时处理4K幅雾天图像的模型。

    *   金字塔网络的N个分支网络对应于泰勒定理中的N个约束项。
    *   低阶多项式重构图像(例如颜色、光照等)的低频信息。
    *   高阶多项式回归图像的高频信息(如纹理)。

<!---->

*   提出了一种正则化项：基于一个论文Tucker重构，作用于金字塔模型的每个分支网络。进一步约束了特征空间中异常信号的产生。
