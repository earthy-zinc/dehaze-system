import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .module.attention import Enhancer
from .module.dinats import PyramidDiNAT_s, CascadeDiNAT_s
from .module.nat_ir import CascadeNAT, PyramidNAT
from .module.rcan import RCAN
from .ridcp.codebook import VectorQuantizer
from .ridcp.decoder import MultiScaleDecoder, DecoderBlock, RIDCPDecoder
from .ridcp.encoder import MultiScaleEncoder, VQEncoder, SwinLayers, NATLayers
from .module import CombineQuantBlock
from .module import VGGFeatureExtractor

class RIDCPNew(nn.Module):
    def __init__(self,
                 *,
                 in_channel=3,
                 codebook_scale=64,
                 codebook_emb_num=1024,
                 codebook_emb_dim=512,
                 gt_resolution=256,
                 LQ_stage=True,
                 norm_type='gn',
                 act_type='silu',
                 use_quantize=True,
                 use_residual=True,
                 only_residual=False,
                 use_weight=False,
                 use_warp=True,
                 weight_alpha=1.0,
                 additional_encoder="PyramidDiNAT",
                 additional_enhancer=True,
                 **ignore_kwargs):
        """

        Args:
            in_channel:
            codebook_scale:
            codebook_emb_num:
            codebook_emb_dim:
            gt_resolution:
            LQ_stage:
            norm_type:
            act_type:
            use_quantize:
            use_residual:
            only_residual:
            use_weight: True
            use_warp:
            weight_alpha: -21.25
            additional_encoder:
                PyramidDiNAT/PyramidNAT 使用金字塔型的(空洞)邻域注意力特征提取器
                CascadeDiNAT/CascadeNAT 使用级联型的邻域注意力特征提取器
                RSTB 使用Swin Transformer的特征提取器RSTB
                其他 不使用额外的特征提取器
            additional_enhancer: 是否启用额外的增强模块
            **ignore_kwargs:
        """
        super().__init__()

        self.codebook_scale = codebook_scale

        self.use_quantize = use_quantize
        self.in_channel = in_channel
        self.gt_res = gt_resolution
        self.LQ_stage = LQ_stage
        self.use_residual = use_residual
        self.only_residual = only_residual
        self.use_weight = use_weight
        self.use_warp = use_warp
        self.weight_alpha = weight_alpha
        self.additional_encoder = additional_encoder
        self.additional_enhancer = additional_enhancer

        channel_query_dict = {
            8: 256,
            16: 256,
            32: 256,
            64: 256,
            128: 128,
            256: 64,
            512: 32,
        }

        # build encoder
        self.max_depth = int(np.log2(gt_resolution // self.codebook_scale))
        self.vq_encoder = VQEncoder(
            in_channel,
            self.max_depth,
            self.gt_res,
            channel_query_dict,
            norm_type, act_type
        )
        if self.LQ_stage and self.use_residual:
            if additional_encoder == "PyramidNAT":
                self.ridcp_encoder = PyramidNAT(
                    embed_dim=[256, 256, 256, 512],
                    output_dim=[256, 256, 512, 1024],
                    mlp_ratio=2.0,
                    depths=[3, 4, 18, 5],
                    num_heads=[4, 8, 16, 32],
                    drop_path_rate=0.5,
                    kernel_size=7,
                    layer_scale=1e-5,
                )
            elif additional_encoder == "CascadeNAT":
                self.ridcp_encoder = CascadeNAT(
                    embed_dim=256,
                    mlp_ratio=2,
                    depths=[3, 4, 18, 5],
                    num_heads=[4, 8, 16, 32],
                    drop_path_rate=0.3,
                    kernel_size=7,
                    layer_scale=1e-5,
                )
            elif additional_encoder == "PyramidDiNAT":
                self.ridcp_encoder = PyramidNAT(
                    embed_dim=[256, 256, 256, 512],
                    output_dim=[256, 256, 512, 1024],
                    mlp_ratio=2.0,
                    depths=[3, 4, 18, 5],
                    num_heads=[4, 8, 16, 32],
                    drop_path_rate=0.5,
                    kernel_size=7,
                    layer_scale=1e-5,
                    dilations=[[1, 16, 1], [1, 4, 1, 8], [1, 2, 1, 3, 1, 4, 1, 2, 1, 3, 1, 4, 1, 2, 1, 3, 1, 4], [1, 2, 1, 2, 1]],
                )
            elif additional_encoder == "CascadeDiNAT":
                self.ridcp_encoder = CascadeNAT(
                    embed_dim=256,
                    mlp_ratio=2,
                    depths=[3, 4, 18, 5],
                    num_heads=[4, 8, 16, 32],
                    drop_path_rate=0.3,
                    kernel_size=7,
                    layer_scale=1e-5,
                    dilations=[[1, 16, 1], [1, 4, 1, 8], [1, 2, 1, 3, 1, 4, 1, 2, 1, 3, 1, 4, 1, 2, 1, 3, 1, 4], [1, 2, 1, 2, 1]],
                )
            elif additional_encoder == "PyramidDiNAT_s":
                self.ridcp_encoder = PyramidDiNAT_s(
                    embed_dim=[256, 256, 256, 512],
                    output_dim=[256, 256, 512, 1024],
                    depths=[2, 2, 18, 2],
                    num_heads=[4, 8, 16, 32],
                    drop_path_rate=0.3,
                    kernel_size=7,
                    dilations=[[1, 16], [1, 8], [1, 2, 1, 3, 1, 4, 1, 2, 1, 3, 1, 4, 1, 2, 1, 3, 1, 4], [1, 2]],
                )
            elif additional_encoder == "CascadeDiNAT_s":
                self.ridcp_encoder = CascadeDiNAT_s(
                    embed_dim=256,
                    depths=[2, 2, 18, 2],
                    num_heads=[4, 8, 16, 32],
                    drop_path_rate=0.3,
                    kernel_size=7,
                    dilations=[[1, 16], [1, 8], [1, 2, 1, 3, 1, 4, 1, 2, 1, 3, 1, 4, 1, 2, 1, 3, 1, 4], [1, 2]],
                )
            elif additional_encoder == "RSTB":
                self.ridcp_encoder = SwinLayers()
            else:
                self.ridcp_encoder = None
            self.ridcp_decoder = RIDCPDecoder(
                in_channel,
                self.max_depth,
                self.gt_res,
                channel_query_dict,
                norm_type, act_type,
                only_residual, use_warp=self.use_warp,
                additional_enhancer=additional_enhancer,
            )

        # build decoder
        out_ch = 0
        self.vq_decoder_group = nn.ModuleList()
        for i in range(self.max_depth):
            res = gt_resolution // 2 ** self.max_depth * 2 ** i
            in_ch, out_ch = channel_query_dict[res], channel_query_dict[res * 2]
            self.vq_decoder_group.append(DecoderBlock(in_ch, out_ch, norm_type, act_type))

        self.out_conv = nn.Conv2d(out_ch, 3, 3, 1, 1)
        if additional_enhancer:
            self.enhancer = Enhancer(3, 3)

        if self.LQ_stage:
            self.residual_conv = nn.Conv2d(out_ch, 3, 3, 1, 1)

        # build vector quantizers
        self.quantizer = VectorQuantizer(
            codebook_emb_num,
            codebook_emb_dim,
            LQ_stage=self.LQ_stage,
            use_weight=self.use_weight,
            weight_alpha=self.weight_alpha
        )
        scale_in_ch = channel_query_dict[self.codebook_scale]
        self.before_quant = nn.Conv2d(scale_in_ch, codebook_emb_dim, 1)
        self.after_quant = CombineQuantBlock(codebook_emb_dim, 0, scale_in_ch)

    def forward(self, inputs, gt_indices=None):
        return self.encode_and_decode(inputs, gt_indices)

    def encode_and_decode(self, inputs, gt_indices=None):
        enc_feats = self.vq_encoder(inputs)
        # 经过 vq_encoder 处理后的 enc_feats.shape [batch_size, 256, 64, 64]
        if self.LQ_stage and hasattr(self, 'ridcp_encoder'):
            enc_feats = self.ridcp_encoder(enc_feats)
            # 经过 ridcp_encoder 处理后的 enc_feats.shape [batch_size, 256, 64, 64]
        feat_to_quant = self.before_quant(enc_feats)
        # feat_to_quant.shape [batch_size, 512, 64, 64]
        # 消融实验5、去除码本和码本匹配操作
        codebook_loss, indices = None, None
        if self.use_quantize:
            z_quant, codebook_loss, indices = self.quantizer(feat_to_quant, gt_indices)
        else:
            z_quant = feat_to_quant
        after_quant_feat = self.after_quant(z_quant)

        # vq-gan解码后的输出集合，总共输出了两次，第一次是经过离散化的并解码一次的，第二次是解码两次的
        code_decoder_output = []
        x = after_quant_feat
        for i in range(self.max_depth):
            x = self.vq_decoder_group[i](x)
            code_decoder_output.append(x)

        if self.LQ_stage and self.use_residual:
            # 消融实验5、去除码本和码本匹配操作
            if self.only_residual:
                residual_feature = self.ridcp_decoder(enc_feats, code_decoder_output)
            else:
                residual_feature = self.ridcp_decoder(enc_feats.detach(), code_decoder_output)
            out_img_residual = self.residual_conv(residual_feature)
        else:
            out_img_residual = None

        out_img = self.out_conv(x)
        # 消融实验4、删除多余的增强模块
        if self.additional_enhancer:
            out_img = self.enhancer(out_img)
        # out_img 图像生成的重建输出
        # out_img_residual 图像去雾的结果输出
        return out_img, out_img_residual, codebook_loss, feat_to_quant, z_quant, indices

    @torch.no_grad()
    def test_tile(self, inputs, tile_size=240, tile_pad=16):
        """
        It will first crop input images to tiles, and then process each tile.
        Finally, all the processed tiles are merged into one images.
        Modified from: https://github.com/xinntao/Real-ESRGAN/blob/master/realesrgan/utils.py
        """
        batch, channel, height, width = inputs.shape
        output_height = height
        output_width = width
        output_shape = (batch, channel, output_height, output_width)

        # start with black image，构造一个全零张量
        output = inputs.new_zeros(output_shape)
        # 根据分割大小 tile_size 确定图片宽度和高度处分别分割多少次
        tiles_x = math.ceil(width / tile_size)
        tiles_y = math.ceil(height / tile_size)

        # loop over all tiles
        for y in range(tiles_y):
            for x in range(tiles_x):
                # extract tile from input image
                ofs_x = x * tile_size
                ofs_y = y * tile_size
                # input tile area on total image，获取分割区域在整个图片上的上下左右点的坐标
                input_start_x = ofs_x
                input_end_x = min(ofs_x + tile_size, width)
                input_start_y = ofs_y
                input_end_y = min(ofs_y + tile_size, height)

                # input tile area on total image with padding，加上边框后分割区域在整个图片上的上下左右点的坐标
                input_start_x_pad = max(input_start_x - tile_pad, 0)
                input_end_x_pad = min(input_end_x + tile_pad, width)
                input_start_y_pad = max(input_start_y - tile_pad, 0)
                input_end_y_pad = min(input_end_y + tile_pad, height)

                # input tile dimensions
                input_tile_width = input_end_x - input_start_x
                input_tile_height = input_end_y - input_start_y
                tile_idx = y * tiles_x + x + 1
                # 在图片上获取被分割的区域
                input_tile = inputs[:, :, input_start_y_pad:input_end_y_pad, input_start_x_pad:input_end_x_pad]

                # upscale tile，处理该区域
                output_tile, _ = self.test(input_tile)

                # output tile area on total image
                output_start_x = int(input_start_x)
                output_end_x = int(input_end_x)
                output_start_y = int(input_start_y)
                output_end_y = int(input_end_y)

                # output tile area without padding
                output_start_x_tile = int(input_start_x - input_start_x_pad)
                output_end_x_tile = int(output_start_x_tile + input_tile_width)
                output_start_y_tile = int(input_start_y - input_start_y_pad)
                output_end_y_tile = int(output_start_y_tile + input_tile_height)

                # put tile into output image
                output[:, :, output_start_y:output_end_y, output_start_x:output_end_x] \
                    = output_tile[:, :, output_start_y_tile:output_end_y_tile, output_start_x_tile:output_end_x_tile]
        return output

    @torch.no_grad()
    def test(self, inputs):
        # padding to multiple of window_size * 8
        wsz = 32
        _, _, h_old, w_old = inputs.shape
        h_pad = (h_old // wsz + 1) * wsz - h_old
        w_pad = (w_old // wsz + 1) * wsz - w_old
        inputs = torch.cat([inputs, torch.flip(inputs, [2])], 2)[:, :, :h_old + h_pad, :]
        inputs = torch.cat([inputs, torch.flip(inputs, [3])], 3)[:, :, :, :w_old + w_pad]

        output_vq, output, _, _, after_quant, index = self.encode_and_decode(inputs, None)

        if output is not None:
            output = output[..., :h_old, :w_old]
            return output, index
        if output_vq is not None:
            output_vq = output_vq[..., :h_old, :w_old]
            return output_vq, index

    def decode_indices(self, indices):
        assert len(indices.shape) == 4, f'shape of indices must be (b, 1, h, w), but got {indices.shape}'

        z_quant = self.quantizer.get_codebook_entry(indices)
        x = self.after_quant(z_quant)
        for m in self.decoder_group:
            x = m(x)
        out_img = self.out_conv(x)
        return out_img


class FusionRefine(nn.Module):
    def __init__(self):
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
                use_quantize: 消融实验5、False 去除码本和码本匹配操作
                use_residual: 消融实验5、False 去除码本和码本匹配操作
                only_residual:
                use_weight: True
                use_warp:
                weight_alpha: -21.25
                additional_encoder: 消融实验2、3
                    PyramidDiNAT/PyramidNAT 使用金字塔型的(空洞)邻域注意力特征提取器
                    CascadeDiNAT/CascadeNAT 使用级联型的邻域注意力特征提取器
                    RSTB 使用Swin Transformer的特征提取器RSTB
                    其他 不使用额外的特征提取器
                additional_enhancer: 消融实验4 是否启用额外的增强模块
            one_branch: 消融实验1、去掉残差通道注意力分支
            **kwargs:
        """
        super(FusionRefine, self).__init__()
        self.one_branch = False
        # first branch
        self.feature_extract = RIDCPNew()
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
