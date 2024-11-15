import torch
from torch import nn

from ..module import ResBlock, WarpBlock
from ..module.attention import DehazeBlock


class DecoderBlock(nn.Module):

    def __init__(self, in_channel, out_channel, norm_type='gn', act_type='leakyrelu'):
        super().__init__()
        self.block = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channel, out_channel, 3, stride=1, padding=1),
            ResBlock(out_channel, out_channel, norm_type, act_type),
            ResBlock(out_channel, out_channel, norm_type, act_type),
        )

    def forward(self, inputs):
        return self.block(inputs)


class MultiScaleDecoder(nn.Module):
    def __init__(self,
                 in_channel,
                 max_depth,
                 input_res=256,
                 channel_query_dict=None,
                 norm_type='gn',
                 act_type='leakyrelu',
                 only_residual=False,
                 use_warp=True
                 ):
        super().__init__()
        self.only_residual = only_residual
        self.use_warp = use_warp
        self.upsampler = nn.ModuleList()
        self.warp = nn.ModuleList()
        res = input_res // (2 ** max_depth)
        for i in range(max_depth):
            in_channel, out_channel = channel_query_dict[res], channel_query_dict[res * 2]
            self.upsampler.append(
                nn.Sequential(
                    nn.Upsample(scale_factor=2),
                    nn.Conv2d(in_channel, out_channel, 3, stride=1, padding=1),
                    ResBlock(out_channel, out_channel, norm_type, act_type),
                    ResBlock(out_channel, out_channel, norm_type, act_type),
                )
            )
            self.warp.append(WarpBlock(out_channel))
            res = res * 2

    def forward(self, x, code_decoder_output):
        for idx, m in enumerate(self.upsampler):
            with torch.backends.cudnn.flags(enabled=False):
                if not self.only_residual:
                    x = m(x)
                    if self.use_warp:
                        x_vq = self.warp[idx](code_decoder_output[idx], x)
                        x = x + x_vq * (x.mean() / x_vq.mean())
                    else:
                        x = x + code_decoder_output[idx]
                else:
                    x = m(x)
        return x


class VQDecoder(nn.Module):
    def __init__(self,
                 max_depth,
                 input_res=256,
                 channel_query_dict=None,
                 norm_type='gn',
                 act_type='leakyrelu',):
        super(VQDecoder, self).__init__()

        self.decoder_group = nn.ModuleList()
        for i in range(max_depth):
            res = input_res // 2 ** max_depth * 2 ** i
            in_ch, out_ch = channel_query_dict[res], channel_query_dict[res * 2]
            self.decoder_group.append(
                DecoderBlock(in_ch, out_ch, norm_type, act_type)
            )

    def forward(self, x):
        for idx, m in enumerate(self.decoder_group):
            x = self.decoder_group[idx](x)
        return x


class RIDCPDecoder(nn.Module):
    def __init__(self,
                 in_channel,
                 max_depth,
                 input_res=256,
                 channel_query_dict=None,
                 norm_type='gn',
                 act_type='leakyrelu',
                 only_residual=False,
                 use_warp=True,
                 additional_enhancer=True,
                 ):
        super().__init__()
        self.only_residual = only_residual
        self.use_warp = use_warp
        self.upsampler = nn.ModuleList()
        self.warp = nn.ModuleList()
        res = input_res // (2 ** max_depth)
        for i in range(max_depth):
            in_channel, out_channel = channel_query_dict[res], channel_query_dict[res * 2]
            # 消融实验4、删除多余的增强模块
            if additional_enhancer:
                self.upsampler.append(
                    nn.Sequential(
                        nn.Upsample(scale_factor=2),
                        nn.Conv2d(in_channel, out_channel, 3, stride=1, padding=1),
                        ResBlock(out_channel, out_channel, norm_type, act_type),
                        ResBlock(out_channel, out_channel, norm_type, act_type),
                        # 不知道有没有用
                        DehazeBlock(nn.Conv2d, out_channel, kernel_size=1),
                    )
                )
            else:
                self.upsampler.append(
                    nn.Sequential(
                        nn.Upsample(scale_factor=2),
                        nn.Conv2d(in_channel, out_channel, 3, stride=1, padding=1),
                        ResBlock(out_channel, out_channel, norm_type, act_type),
                        ResBlock(out_channel, out_channel, norm_type, act_type),
                    )
                )
            self.warp.append(WarpBlock(out_channel))
            res = res * 2

    def forward(self, x, code_decoder_output):
        # in x (batch_size, 256, w, h)
        # in code_decoder_output list length 2 shape (batch_size, 256, w, h)
        # out channel 256 -> 128 -> 64
        for idx, m in enumerate(self.upsampler):
            with torch.backends.cudnn.flags(enabled=False):
                if not self.only_residual:
                    x = m(x)
                    if self.use_warp:
                        x_vq = self.warp[idx](code_decoder_output[idx], x)
                        x = x + x_vq * (x.mean() / x_vq.mean())
                    else:
                        x = x + code_decoder_output[idx]
                else:
                    x = m(x)
        return x
