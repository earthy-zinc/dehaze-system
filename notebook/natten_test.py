import sys
import torch
from torch import nn
sys.path.append("/var/lib/docker/user1/wpx/DeepLearningCopies/2023/RIDCP")
sys.path.append("/quzhong_fix/wpx/DeepLearningCopies/2023/RIDCP")
from torchsummary import summary
# from basicsr.archs.module.rcan import RCAN
from basicsr.archs.module.dinats import PyramidDiNAT
from basicsr.archs.ridcp.encoder import SwinLayers
from basicsr.archs.module.nat_ir import CascadeNAT

random_data = torch.randn(1, 256, 64, 64).cuda()
nat = CascadeNAT(
    depths=[3, 4, 18, 5],
    num_heads=[4, 8, 16, 32],
    embed_dim=256,
    mlp_ratio=2,
    drop_path_rate=0.3,
    layer_scale=1e-5,
    kernel_size=7
).cuda()
# summary(nat, input_size=(256, 64, 64), batch_size=1, device="cuda")
print(nat(random_data).shape)

dinat = PyramidDiNAT(
    depths=[2, 2, 18, 2],
    num_heads=[4, 8, 16, 32],

    mlp_ratio=4,
    drop_path_rate=0.5,
    kernel_size=7,
    dilations=[
        [1, 8],
        [1, 4],
        [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
        [1, 1],
    ],
).cuda()
# summary(dinat, input_size=(256, 64, 64), batch_size=1, device="cuda")
print(dinat(random_data).shape)

rstb = SwinLayers().cuda()
# summary(rstb, input_size=(256, 64, 64), batch_size=1, device="cuda")
print(rstb(random_data).shape)


