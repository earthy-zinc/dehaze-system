# 算法研究 (dehaze-algorithm)

基于高质量码本的双分支多尺度图像去雾算法源码实现。详细业务文档见 [dehaze-doc](../dehaze-doc/docs/05-子项目实现/学术论文文档.md)。

## 技术栈

- Python
- PyTorch
- BasicSR（基于 BasicSR 框架的训练/测试管线）

## 快速开始

### 1. 安装依赖

```bash
pip install -e .
```

> `setup.py` 会读取 `requirements.txt` 安装全部依赖。如需编译 CUDA 算子（如 DCN），请设置环境变量 `BASICSR_EXT=True`。

### 2. 推理

使用 `inference_ridcp.py` 进行单图或批量推理：

```bash
python inference_ridcp.py \
    -i inputs \
    -w path/to/model_weight.pth \
    -o results \
    --use_weight \
    --alpha 1.0
```

参数说明：

- `-i / --input`：输入图片或文件夹，默认 `inputs`
- `-w / --weight`：模型权重路径
- `-o / --output`：输出文件夹，默认 `results`
- `--use_weight`：启用权重融合
- `--alpha`：权重融合系数，默认 `1.0`
- `--max_size`：单张图片最大尺寸，超过则启用分块推理，默认 `10000`

### 3. 训练

通过 `basicsr/train.py` 启动训练，配置文件位于 `options/` 目录：

```bash
python basicsr/train.py -opt options/common/NH-HAZE-20.yml
```

### 4. 测试

通过 `basicsr/test.py` 跑测试集评估：

```bash
python basicsr/test.py -opt options/common/NH-HAZE-20.yml
```

> 消融实验与对比实验配置分别位于 `options/ablation/` 与 `options/compare/`。
