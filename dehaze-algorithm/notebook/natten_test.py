import sys
import torch
from torch import nn
sys.path.append("/var/lib/docker/user1/wpx/DeepLearningCopies/2023/RIDCP")
sys.path.append("/quzhong_fix/wpx/DeepLearningCopies/2023/RIDCP")
from torchsummary import summary
# from basicsr.archs.module.rcan import RCAN
from basicsr.archs.module.dinats import PyramidDiNAT_s, CascadeDiNAT_s
from basicsr.archs.ridcp.encoder import SwinLayers
from basicsr.archs.module.nat_ir import CascadeNAT, PyramidNAT

random_data = torch.randn(1, 256, 64, 88).cuda()
# ---------------------NAT------------------
pyramid_nat = PyramidNAT(
    embed_dim=[256, 256, 256, 512],
    output_dim=[256, 256, 512, 1024],
    mlp_ratio=2.0,
    depths=[3, 4, 18, 5],
    num_heads=[4, 8, 16, 32],
    drop_path_rate=0.5,
    kernel_size=7,
    layer_scale=1e-5,
).cuda()
print(pyramid_nat(random_data).shape)

nat = CascadeNAT(
    embed_dim=256,
    mlp_ratio=2,
    depths=[3, 4, 18, 5],
    num_heads=[4, 8, 16, 32],
    drop_path_rate=0.3,
    kernel_size=7,
    layer_scale=1e-5,
).cuda()
# summary(nat, input_size=(256, 64, 64), batch_size=1, device="cuda")
print(nat(random_data).shape)

# ---------------------DiNAT------------------
pyramid_dinat = PyramidNAT(
    embed_dim=[256, 256, 256, 512],
    output_dim=[256, 256, 512, 1024],
    mlp_ratio=2.0,
    depths=[3, 4, 18, 5],
    num_heads=[4, 8, 16, 32],
    drop_path_rate=0.5,
    kernel_size=7,
    layer_scale=1e-5,
    dilations=[[1, 16, 1], [1, 4, 1, 8], [1, 2, 1, 3, 1, 4, 1, 2, 1, 3, 1, 4, 1, 2, 1, 3, 1, 4], [1, 2, 1, 2, 1]],
).cuda()

# summary(pyramid_dinat, input_size=(256, 64, 88), batch_size=1, device="cuda")
print(pyramid_dinat(random_data).shape)

cascade_dinat = CascadeNAT(
    embed_dim=256,
    mlp_ratio=2,
    depths=[3, 4, 18, 5],
    num_heads=[4, 8, 16, 32],
    drop_path_rate=0.3,
    kernel_size=7,
    layer_scale=1e-5,
    dilations=[[1, 16, 1], [1, 4, 1, 8], [1, 2, 1, 3, 1, 4, 1, 2, 1, 3, 1, 4, 1, 2, 1, 3, 1, 4], [1, 2, 1, 2, 1]],
).cuda()
print(cascade_dinat(random_data).shape)

# ---------------------DiNAT_s------------------
pyramid_dinat_s = PyramidDiNAT_s(
    embed_dim=[256, 256, 256, 512],
    output_dim=[256, 256, 512, 1024],
    depths=[2, 2, 18, 2],
    num_heads=[4, 8, 16, 32],
    drop_path_rate=0.3,
    kernel_size=7,
    dilations=[[1, 16], [1, 8], [1, 2, 1, 3, 1, 4, 1, 2, 1, 3, 1, 4, 1, 2, 1, 3, 1, 4], [1, 2]],
).cuda()
# summary(pyramid_dinat_s, input_size=(256, 64, 64), batch_size=1, device="cuda")
print(pyramid_dinat_s(random_data).shape)

cascade_dinat_s = CascadeDiNAT_s(
    embed_dim=256,
    depths=[2, 2, 18, 2],
    num_heads=[4, 8, 16, 32],
    drop_path_rate=0.3,
    kernel_size=7,
    dilations=[[1, 16], [1, 8], [1, 2, 1, 3, 1, 4, 1, 2, 1, 3, 1, 4, 1, 2, 1, 3, 1, 4], [1, 2]],
).cuda()
print(cascade_dinat_s(random_data).shape)

# ---------------------rstb------------------
rstb = SwinLayers().cuda()
# summary(rstb, input_size=(256, 64, 64), batch_size=1, device="cuda")
print(rstb(random_data).shape)


