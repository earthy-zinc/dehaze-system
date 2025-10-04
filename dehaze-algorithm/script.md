```shell
cd /mnt/e/DeepLearningCopies/2023/RIDCP
cd /quzhong_fix/wpx/DeepLearningCopies/2023/RIDCP
cd /mnt/workspace/ridcp
cd /var/lib/docker/user1/wpx/DeepLearningCopies/2023/RIDCP

CUDA_VISIBLE_DEVICES=7
BASICSR_JIT=True python basicsr/train.py -opt options/ablation/a/Codebook-NH-HAZE-20.yml --auto_resume 
BASICSR_JIT=True python basicsr/train.py -opt options/ablation/a/Enhancer-NH-HAZE-20.yml --auto_resume 
BASICSR_JIT=True python basicsr/train.py -opt options/ablation/a/Enhancer-NH-HAZE-21.yml --auto_resume 
BASICSR_JIT=True python basicsr/train.py -opt options/ablation/a/NoRSTB-NH-HAZE-20.yml --auto_resume 
BASICSR_JIT=True python basicsr/train.py -opt options/ablation/a/RCAN-NH-HAZE-20.yml --auto_resume 
BASICSR_JIT=True python basicsr/train.py -opt options/ablation/a/RSTB-NH-HAZE-20.yml --auto_resume 
```

```shell
cd /var/lib/docker/user1/wpx/DeepLearningCopies/2023/RIDCP && \
CUDA_VISIBLE_DEVICES=7 BASICSR_JIT=True python basicsr/train.py -opt options/ablation/a/Enhancer-NH-HAZE-20.yml --auto_resume
```

```shell
python basicsr/train.py -opt options/compare/FFA/DENSE-HAZE.yml && \
python basicsr/train.py -opt options/compare/FFA/I-HAZE.yml && \
python basicsr/train.py -opt options/compare/FFA/O-HAZE.yml && \
python basicsr/train.py -opt options/compare/FFA/NH-HAZE-20.yml && \
python basicsr/train.py -opt options/compare/FFA/NH-HAZE-21.yml && \
python basicsr/train.py -opt options/compare/FFA/NH-HAZE-23.yml

```

# 测试方法
## 预备工作
```shell
# 切换到工作目录
cd /var/lib/docker/user1/wpx/DeepLearningCopies/2023/RIDCP
conda activate ridcp
```

## DCP
```shell
python basicsr/test.py -opt options/compare/DCP/RESIDE-IN.yml
python basicsr/test.py -opt options/compare/DCP/RESIDE-OUT.yml
python basicsr/test.py -opt options/compare/DCP/RESIDE-6K.yml
```

## AOD
```shell
python basicsr/test.py -opt options/compare/AOD/RESIDE-IN.yml
python basicsr/test.py -opt options/compare/AOD/RESIDE-OUT.yml
python basicsr/test.py -opt options/compare/AOD/RESIDE-6K.yml
```

## FFA
```shell
python basicsr/test.py -opt options/compare/FFA/RESIDE-IN.yml
python basicsr/test.py -opt options/compare/FFA/RESIDE-OUT.yml
python basicsr/test.py -opt options/compare/FFA/RESIDE-6K.yml
```

## 查看结果
结果保存在以 方法名-数据集名称 命名的文件夹中
```shell
cd /var/lib/docker/user1/wpx/DeepLearningCopies/2023/RIDCP/results
```
