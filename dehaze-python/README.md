## 📢 项目介绍

雾霾的存在会导致图像的质量会急剧恶化，造成⾊彩失真、特征模糊、对⽐度降低等问题，针对当前图像去雾领域存在缺乏强⼤的先验知识、浓雾区域去雾不彻底问题，本系统基于深度学习⽅法研究设计了⼀种真实场景⾮均匀雾的环境条件下的图像去雾⽅法。基于该方法构建了一个基于深度学习的在线实时响应的图像去雾系统，从⽽实现最终端到端的图像去雾的⽬标。

本部分为图像去雾系统的 Python 后端，基于 PyTorch 构建深度学习模型，Flask 作为 Web 服务框架提供 API 接口，通过 Gunicorn 进行生产级部署。是整个图像去雾系统最核心的部分，同时向外提供 API 接口以供 Java 后端调用。

## 💻 模块划分

- 算法模块：algorithm/目录下包含 20+种去雾算法模型（如 RIDCP、WPXNet、Dehamer 等），每个算法有独立的模型定义、运行脚本和依赖配置。通过 importPath 配置动态加载不同算法模型。
- 服务模块：app/目录提供 API 接口层，包含文件处理、模型调用、结果返回等功能。
- 测试模块：tests/目录包含模型测试用例和数据集配置，如 dir_test.py 定义了各模型的权重路径。
- 部署模块：通过 Docker 容器化（Dockerfile）实现环境一致性，使用 NVIDIA CUDA 12.1 镜像支持 GPU 加速。

## 🛞 模型介绍

1. 引⼊从清晰⽆雾图像训练得到离散码本，封装具有原有图像⾊彩和结构的⾼质量先验知识，构建⼀种双分⽀神经⽹络结构。
2. 针对浓雾和⾮均匀雾霾区域图像纹理和结构特征的提取，设计了⼀种⾦字塔空洞邻域注意⼒编码器，聚合不同层级的特征, 实现不同尺度的特征重⽤。
3. 将基于 Transformer 的邻域注意⼒和基于卷积的通道注意⼒结合，提取图像全局特征并学习浓雾区域与底层场景之间复杂交互特征，通过特征融合模块对两个分⽀提取的特征进⾏融合。进⽽对雾霾图像重建实现端到端的图像去雾流程

## 💡 项目亮点

1. 去雾模型封装：利⽤ Flask 搭建的 web 框架，封装基于 Python 去雾模型进⽽通过 API 接⼝实现模型调⽤
2. 分层架构设计：实现 Web 服务层（Flask）、模型推理层（PyTorch）、存储层（MinIO）分离，通过工厂模式动态加载模型算法实现模型可插拔架构
3. 跨平台与生产级部署：通过 Dockerfile 多阶段构建，减小最终镜像体积。通过健康检查（HEALTHCHECK）监控服务状态，实现高可用
4. 依赖管理：利用 requirements.txt 打包 docker 镜像，精准控制 CUDA、PyTorch 等依赖版本，确保环境一致性。
5. 监控预警：集成 Prometheus+Grafana 监控系统，实时监控 GPU 利用率、算法模型的推理耗时、准确率等指标，生成性能报告
6. 弹性伸缩：监控 GPU 利用率 Kubernetes 自动调整应用 Pod 数量，流量高峰时自动扩容，GPU 利用率稳定在 65%±5%，避免资源浪费
7. 请求限流：通过 Flask-Limiter 实现限流，防止 API 被滥用

## 🚨 项目难点

模型兼容性：

- 部分模型（如 CFENViTDehazing）因依赖未解决或代码问题无法运行。
- 模型配置差异大（如 RIDCP 需 BASICSR_JIT=True，WPXNet 依赖 CUDA 扩展模块）。
  跨平台问题：
- 部分模型（如 RIDCP、WPXNet）仅支持 Linux，Windows 环境需额外适配。
  依赖管理：
- Dockerfile 中需精确指定 PyTorch 和 Natten 版本（如 torchvision-0.16.0+cu121），升级时易引发兼容性问题。
  性能瓶颈：
- 多模型并行推理时 GPU 资源分配需优化（如 Gunicorn 工作进程数需根据显存调整）。

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

在安装好项目依赖，启动项目成功之后。可以进行项目的部署操作。Gunicorn 是一个流行的 Python WSGI HTTP 服务器，适用于生产环境。我们通过使用 Gunicorn 部署 Flask 应用。

```bash
gunicorn -w 4 start:app
```

- -w 4 表示使用 4 个工作进程。
- start:app 中的 start 是启动 Flask 应用所在的文件名（不包括.py 扩展名），app 是 Flask 实例的名称。

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

- AECRNet no
- CFENViTDehazing no
- DaclipUir no
- DCPDN no
- FCD no
- PSD no

以下模型准备调试

- TNN todo
- ImgRestorationSde todo
- MB-TaylorFormer todo

以下模型需要在 linux 系统中运行

- RIDCP 需要 BASICSR_JIT = True
- WPXNet 需要 natten
