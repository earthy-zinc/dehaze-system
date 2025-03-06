## 📢 项目介绍

雾霾的存在会导致图像的质量会急剧恶化，造成⾊彩失真、特征模糊、对⽐度降低等问题，针对当前图像去雾领域存在缺乏强⼤的先验知识、浓雾区域去雾不彻底问题，本系统基于深度学习⽅法研究设计了⼀种真实场景⾮均匀雾的环境条件下的图像去雾⽅法。基于该方法构建了一个基于深度学习的在线实时响应的图像去雾系统，从⽽实现最终端到端的图像去雾的⽬标。

本部分为图像去雾系统的 Python 后端，基于PyTorch构建深度学习模型，Flask作为Web服务框架提供API接口，通过Gunicorn进行生产级部署。是整个图像去雾系统最核心的部分，同时向外提供API接口以供Java后端调用。

## 💻 模块划分

* 算法模块：algorithm/目录下包含20+种去雾算法模型（如RIDCP、WPXNet、Dehamer等），每个算法有独立的模型定义、运行脚本和依赖配置。通过importPath配置动态加载不同算法模型。
* 服务模块：app/目录提供API接口层，包含文件处理、模型调用、结果返回等功能。
* 测试模块：tests/目录包含模型测试用例和数据集配置，如dir_test.py定义了各模型的权重路径。
* 部署模块：通过Docker容器化（Dockerfile）实现环境一致性，使用NVIDIA CUDA 12.1镜像支持GPU加速。

## 💡 项目亮点

1. 引⼊从清晰⽆雾图像训练得到离散码本，封装具有原有图像⾊彩和结构的⾼质量先验知识，构建⼀种双分⽀神经⽹络结构。
2. 针对浓雾和⾮均匀雾霾区域图像纹理和结构特征的提取，设计了⼀种⾦字塔空洞邻域注意⼒编码器，聚合不同层级的特征, 实现不同尺度的特征重⽤。
3. 将基于 Transformer 的邻域注意⼒和基于卷积的通道注意⼒结合，提取图像全局特征并学习浓雾区域与底层场景之间复杂交互特征，通过特征融合模块对两个分⽀提取的特征进⾏融合。进⽽对雾霾图像重建实现端到端的图像去雾流程
4. 去雾模型封装：利⽤ Flask 搭建的 web 框架，封装基于 Python 去雾模型进⽽通过 API 接⼝实现模型调⽤

## 🚨 项目难点

模型兼容性：
  * 部分模型（如CFENViTDehazing）因依赖未解决或代码问题无法运行。
  * 模型配置差异大（如RIDCP需BASICSR_JIT=True，WPXNet依赖CUDA扩展模块）。
跨平台问题：
  * 部分模型（如RIDCP、WPXNet）仅支持Linux，Windows环境需额外适配。
依赖管理：
  * Dockerfile中需精确指定PyTorch和Natten版本（如torchvision-0.16.0+cu121），升级时易引发兼容性问题。
性能瓶颈：
  * 多模型并行推理时GPU资源分配需优化（如Gunicorn工作进程数需根据显存调整）。

## 🚀 项目启动

```bash
# 克隆代码
git clone https://gitee.com/earthy-zinc/dehaze_python.git

# 切换目录
cd dehaze_python

# 安装 miniconda 并创建虚拟环境
conda env create -n dehaze_backend python=3.10
conda acticate dehaze_backend

# 安装依赖
conda install --yes --file requirements.txt

# 启动运行
python start.py
```

## 🌺 项目部署

在安装好项目依赖，启动项目成功之后。可以进行项目的部署操作。Gunicorn是一个流行的Python WSGI HTTP服务器，适用于生产环境。我们通过使用Gunicorn部署Flask应用。

```bash
gunicorn -w 4 start:app
```
* -w 4表示使用4个工作进程。
* start:app中的 start 是启动Flask应用所在的文件名（不包括.py扩展名），app是Flask实例的名称。

## 🌈 模型

```yml
name: "算法名称"
type: "算法类型"
description: "算法描述"
importPath: "算法代码导入路径"
children:
    - name: "子模型名称"
      type: "子模型类型"
      description: "子模型描述"
      path: "模型路径"


```

## 📥 注意事项
以下去雾模型由于一些原因无法运行
* AECRNet no
* CFENViTDehazing no
* DaclipUir no
* DCPDN no
* FCD no
* PSD no

以下模型准备调试
* TNN todo
* ImgRestorationSde todo
* MB-TaylorFormer todo

以下模型需要在linux系统中运行
* RIDCP 需要BASICSR_JIT = True
* WPXNet 需要natten
