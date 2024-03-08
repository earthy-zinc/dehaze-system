import torch
from torch import nn

from basicsr.archs.module.rcan import RCAN
from basicsr.archs.ridcp_arch import RIDCP
from basicsr.archs.ridcp_new_arch import RIDCPNew
from basicsr.utils.registry import ARCH_REGISTRY


@ARCH_REGISTRY.register()
class FusionRefine(nn.Module):
    def __init__(self, opt, one_branch=False, **kwargs):
        """
        Args:
            opt:
                in_channel:
                codebook_scale:
                codebook_emb_num:
                codebook_emb_dim:
                gt_resolution:
                LQ_stage:
                norm_type:
                act_type:
                use_quantize: 消融实验5、True 去除码本和码本匹配操作 False 不去除
                use_residual: 消融实验5、True 去除码本和码本匹配操作 False 不去除
                only_residual:
                use_weight: True
                use_warp:
                weight_alpha: -21.25
                additional_encoder: 消融实验2、3
                    DiNAT 使用金字塔型的邻域注意力特征提取器
                    NAT 使用级联型的邻域注意力特征提取器
                    RSTB 使用Swin Transformer的特征提取器RSTB
                    Many_NATs 使用多个级联型的邻域注意力特征提取器
                    其他 不使用额外的特征提取器
                additional_enhancer: 消融实验4 是否启用额外的增强模块
            one_branch: 消融实验1、去掉残差通道注意力分支
            **kwargs:
        """
        super(FusionRefine, self).__init__()
        print(opt)
        print(kwargs)
        self.one_branch = one_branch
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
        # 消融实验1、去掉残差通道注意力分支
        if self.one_branch:
            return out_img, out_img_residual, codebook_loss, feat_to_quant, z_quant, indices
        else:
            rcan_out = self.pre_trained_rcan(inputs)
            x = torch.cat([out_img_residual, rcan_out], 1)
            feat_hazy = self.tail(x)
            return out_img, feat_hazy, codebook_loss, feat_to_quant, z_quant, indices

    @torch.no_grad()
    def test_tile(self, inputs):
        x = self.feature_extract.test_tile(inputs)
        # 消融实验1、去掉残差通道注意力分支
        if self.one_branch:
            return x
        else:
            y = self.pre_trained_rcan(inputs)
            out = torch.cat([x, y], 1)
            return self.tail(out)

    @torch.no_grad()
    def test(self, inputs):
        x, index = self.feature_extract.test(inputs)
        if self.one_branch:
            return x, index
        else:
            y = self.pre_trained_rcan(inputs)
            out = torch.cat([x, y], 1)
            return self.tail(out), index
