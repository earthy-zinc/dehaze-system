import sys
import torch
from torch import nn

sys.path.append("/var/lib/docker/user1/wpx/DeepLearningCopies/2023/RIDCP")
sys.path.append("/quzhong_fix/wpx/DeepLearningCopies/2023/RIDCP")
from torchsummary import summary
from basicsr.archs.module.rcan import RCAN
from basicsr.archs.module import RSTB
from basicsr.archs.module.dinats import DiNAT_s, BasicLayer, DiNAT, DiNAT_e
from basicsr.archs.module.nat_ir import NAT

# nat_layers = NATLayers()
# nat_layers = nat_layers.cuda()
# random_data = torch.randn(1, 3, 256, 256).cuda()
#
# data = nat_layers(random_data)
# print(data.shape)

# nat = NAT(256, 2.0, [3, 4, 18, 5], [4, 8, 16, 32], 0.5)
nat = DiNAT(
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
)


summary(nat, input_size=(3, 256, 256), batch_size=1, device="cpu")
#
#
# random_data = torch.randn(1, 3, 256, 256)
#
# data = rcan(random_data)
# print(data.shape)
