import sys
import torch
from torch import nn

sys.path.append("/var/lib/docker/user1/wpx/DeepLearningCopies/2023/RIDCP")
sys.path.append("/quzhong_fix/wpx/DeepLearningCopies/2023/RIDCP")
from torchsummary import summary
from basicsr.archs.module.rcan import RCAN
from basicsr.archs.module import RSTB
from basicsr.archs.module.dinats import DiNAT_s, BasicLayer
from basicsr.archs.module.nat_ir import NAT

# nat_layers = NATLayers()
# nat_layers = nat_layers.cuda()
# random_data = torch.randn(1, 3, 256, 256).cuda()
#
# data = nat_layers(random_data)
# print(data.shape)

# nat = NAT(256, 2.0, [3, 4, 18, 5], [4, 8, 16, 32], 0.5)
# nat = DiNAT_s(
#     depths=[2, 2, 18, 2],
#     num_heads=[4, 8, 16, 32],
#     embed_dim=256,
#     mlp_ratio=4,
#     drop_path_rate=0.5,
#     kernel_size=7,
#     dilations=[
#         [1, 8],
#         [1, 4],
#         [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
#         [1, 1],
#     ],
# )
depths = [2, 2, 18, 2]
dpr = [
    x.item() for x in torch.linspace(0, 0.5, sum(depths))
]  # stochastic depth decay rule
nat = BasicLayer(
    dim=256,
    depth=2,
    num_heads=4,
    kernel_size=7,
    dilations=[1, 8],
    mlp_ratio=4,
    qkv_bias=True,
    qk_scale=None,
    drop=0.0,
    attn_drop=0.0,
    drop_path=dpr[0:2],
    norm_layer=nn.LayerNorm,
    downsample=None,
)
# rcan = RCAN()
#
summary(nat, input_size=(3, 256, 256), batch_size=1, device="cpu")
#
#
# random_data = torch.randn(1, 3, 256, 256)
#
# data = rcan(random_data)
# print(data.shape)
