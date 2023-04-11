# MMdetection

## 介绍

MMDetection是一个深度学习集成框架，预训练了许多目标检测相关的模型。

## 安装

```shell
#创建一个名为open-mmlab的虚拟开发环境，基于python3.7
conda create --name open-mmlab python=3.7 -y
#进入该虚拟环境
conda activate open-mmlab
#安装基于CUDA11.7版本的pytorch
conda install pytorch pytorch-cuda=11.7 torchvision
#使用mim命令安装MMDetection
pip install openmim
mim install mmdet
```

## 配置文件

在MMDetection中，一个训练好的深度学习神经网络模型被定义为：一个配置文件、对应的存储在检查点（checkpoint）文件内的模型参数。

