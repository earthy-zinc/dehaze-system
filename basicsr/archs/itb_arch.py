import torch
from torch import nn

from basicsr.archs.module.rcan import RCAN
from basicsr.archs.ridcp_arch import RIDCP
from basicsr.archs.ridcp_new_arch import RIDCPNew
from basicsr.utils.registry import ARCH_REGISTRY


@ARCH_REGISTRY.register()
class FusionRefine(nn.Module):
    def __init__(self, opt, **kwargs):
        super(FusionRefine, self).__init__()
        print(opt)
        print(kwargs)
        # first branch
        self.feature_extract = RIDCPNew(**opt)
        # second branch
        self.pre_trained_rcan = RCAN()
        self.tail = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(35, 3, kernel_size=7, padding=0),
            nn.Tanh()
        )

    def forward(self, inputs, gt_indices=None):
        out_img, out_img_residual, codebook_loss, feat_to_quant, z_quant, indices = self.feature_extract(inputs, gt_indices)
        rcan_out = self.pre_trained_rcan(inputs)
        x = torch.cat([out_img_residual, rcan_out], 1)
        # print("out_img_residual.shape")
        # print(rcan_out.shape)
        # print(out_img_residual.shape)
        # print(x.shape)
        feat_hazy = self.tail(x)
        return out_img, feat_hazy, codebook_loss, feat_to_quant, z_quant, indices

    @torch.no_grad()
    def test_tile(self, inputs):
        x = self.feature_extract.test_tile(inputs)
        y = self.pre_trained_rcan(inputs)
        out = torch.cat([x, y], 1)
        return self.tail(out)

    @torch.no_grad()
    def test(self, inputs):
        x, index = self.feature_extract.test(inputs)
        y = self.pre_trained_rcan(inputs)
        out = torch.cat([x, y], 1)
        return self.tail(out), index
