```shell
cd /mnt/e/DeepLearningCopies/2023/RIDCP
cd /quzhong_fix/wpx/DeepLearningCopies/2023/RIDCP
cd /mnt/workspace/ridcp
cd /var/lib/docker/user1/wpx/DeepLearningCopies/2023/RIDCP

CUDA_VISIBLE_DEVICES=7
BASICSR_JIT=True python basicsr/train.py -opt options/ablation/a/Codebook-NH-HAZE-20.yml --auto_resume 
BASICSR_JIT=True python basicsr/train.py -opt options/ablation/a/Enhancer-NH-HAZE-20.yml --auto_resume 
BASICSR_JIT=True python basicsr/train.py -opt options/ablation/a/NoRSTB-NH-HAZE-20.yml --auto_resume 
BASICSR_JIT=True python basicsr/train.py -opt options/ablation/a/RCAN-NH-HAZE-20.yml --auto_resume 
BASICSR_JIT=True python basicsr/train.py -opt options/ablation/a/RSTB-NH-HAZE-20.yml --auto_resume 
```

```shell
cd /var/lib/docker/user1/wpx/DeepLearningCopies/2023/RIDCP && \
CUDA_VISIBLE_DEVICES=7 BASICSR_JIT=True python basicsr/train.py -opt options/ablation/a/Enhancer-NH-HAZE-20.yml --auto_resume
```
