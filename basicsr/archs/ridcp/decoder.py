import torch
from torch import nn

from basicsr.archs.module import ResBlock, WarpBlock


class DecoderBlock(nn.Module):

    def __init__(self, in_channel, out_channel, norm_type='gn', act_type='leakyrelu'):
        super().__init__()

        self.block = []
        self.block += [
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channel, out_channel, 3, stride=1, padding=1),
            ResBlock(out_channel, out_channel, norm_type, act_type),
            ResBlock(out_channel, out_channel, norm_type, act_type),
        ]

        self.block = nn.Sequential(*self.block)

    def forward(self, input):
        return self.block(input)


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
