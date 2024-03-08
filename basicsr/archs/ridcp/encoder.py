import torch
from torch import nn

from basicsr.archs.module import RSTB
from basicsr.archs.module import ResBlock
from basicsr.archs.module.dinats import PyramidDiNAT
from basicsr.archs.module.nat_ir import CascadeNAT


class SwinLayers(nn.Module):
    def __init__(self, input_resolution=(32, 32), embed_dim=256,
                 blk_depth=6,
                 num_heads=8,
                 window_size=8,
                 **kwargs):
        super().__init__()
        self.swin_blks = nn.ModuleList()
        for i in range(4):
            layer = RSTB(embed_dim, input_resolution, blk_depth, num_heads, window_size, patch_size=1, **kwargs)
            self.swin_blks.append(layer)

    def forward(self, x):
        b, c, h, w = x.shape
        x = x.reshape(b, c, h*w).transpose(1, 2)
        for m in self.swin_blks:
            x = m(x, (h, w))
        x = x.transpose(1, 2).reshape(b, c, h, w)
        return x

class NATLayers(nn.Module):
    def __init__(self):
        super().__init__()
        self.nat_blks = nn.ModuleList()
        for i in range(4):
            self.nat_blks.append(
                CascadeNAT(
                    depths=[3, 4, 18, 5],
                    num_heads=[4, 8, 16, 32],
                    embed_dim=256,
                    mlp_ratio=2,
                    drop_path_rate=0.3,
                    layer_scale=1e-5,
                    kernel_size=7
                )
            )

    def forward(self, x):
        for m in self.nat_blks:
            x = m(x)
        return x


class MultiScaleEncoder(nn.Module):
    def __init__(self,
                 in_channel,
                 max_depth,
                 input_res=256,
                 channel_query_dict=None,
                 norm_type='gn',
                 act_type='leakyrelu',
                 LQ_stage=True,
                 additional_encoder="RSTB",
                 **swin_opts,
                 ):
        super().__init__()

        self.in_conv = nn.Conv2d(in_channel, channel_query_dict[input_res], 4, padding=1)

        self.blocks = nn.ModuleList()
        res = input_res
        for i in range(max_depth):
            in_ch, out_ch = channel_query_dict[res], channel_query_dict[res // 2]
            tmp_down_block = [
                nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1),
                ResBlock(out_ch, out_ch, norm_type, act_type),
                ResBlock(out_ch, out_ch, norm_type, act_type),
            ]
            self.blocks.append(nn.Sequential(*tmp_down_block))
            res = res // 2

        if LQ_stage:
            if additional_encoder == "DiNAT":
                self.blocks.append(PyramidDiNAT(
                    depths=[2, 2, 18, 2],
                    num_heads=[4, 8, 16, 32],
                    embed_dim=[256, 256, 256, 512],
                    mlp_ratio=4,
                    drop_path_rate=0.5,
                    kernel_size=7,
                    dilations=[
                        [1, 8],
                        [1, 4],
                        [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
                        [1, 1],
                    ],
                ))
            elif additional_encoder == "NAT":
                self.blocks.append(CascadeNAT(
                    depths=[3, 4, 18, 5],
                    num_heads=[4, 8, 16, 32],
                    embed_dim=256,
                    mlp_ratio=2,
                    drop_path_rate=0.3,
                    layer_scale=1e-5,
                    kernel_size=7
                ))
            elif additional_encoder == "RSTB":
                self.blocks.append(SwinLayers(**swin_opts))
            elif additional_encoder == "many_nats":
                self.blocks.append(NATLayers())
            else:
                pass

    def forward(self, input):
        x = self.in_conv(input)
        for idx, m in enumerate(self.blocks):
            with torch.backends.cudnn.flags(enabled=False):
                x = m(x)
        return x


class VQEncoder(nn.Module):
    def __init__(self,
                 in_channel,
                 max_depth,
                 input_res=256,
                 channel_query_dict=None,
                 norm_type='gn',
                 act_type='leakyrelu',
                 ):
        super().__init__()
        self.in_conv = nn.Conv2d(in_channel, channel_query_dict[input_res], 4, padding=1)

        res = input_res
        self.blocks = nn.ModuleList()
        for i in range(max_depth):
            in_ch, out_ch = channel_query_dict[res], channel_query_dict[res // 2]
            down_sample_block = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1),
                ResBlock(out_ch, out_ch, norm_type, act_type),
                ResBlock(out_ch, out_ch, norm_type, act_type),
            )
            self.blocks.append(down_sample_block)
            res = res // 2

    def forward(self, inputs):
        x = self.in_conv(inputs)
        for i, m in enumerate(self.blocks):
            with torch.backends.cudnn.flags(enabled=False):
                x = m(x)
        return x



