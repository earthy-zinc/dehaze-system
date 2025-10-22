import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .ridcp.codebook import VectorQuantizer
from .ridcp.decoder import MultiScaleDecoder, DecoderBlock
from .ridcp.encoder import MultiScaleEncoder
from .module import CombineQuantBlock
from .module import VGGFeatureExtractor
from ..utils.registry import ARCH_REGISTRY


@ARCH_REGISTRY.register()
class HQP(nn.Module):
    def __init__(self,
                 *,
                 in_channel=3,
                 codebook_scale=64,
                 codebook_emb_num=1024,
                 codebook_emb_dim=512,
                 gt_resolution=256,
                 norm_type='gn',
                 act_type='silu',
                 use_weight=False,
                 use_warp=True,
                 weight_alpha=1.0,
                 **ignore_kwargs):
        super().__init__()
        self.codebook_scale = codebook_scale

        self.in_channel = in_channel
        self.gt_res = gt_resolution
        self.use_weight = use_weight
        self.use_warp = use_warp
        self.weight_alpha = weight_alpha

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
        self.multiscale_encoder = MultiScaleEncoder(
            in_channel,
            self.max_depth,
            self.gt_res,
            channel_query_dict,
            norm_type, act_type,
            LQ_stage=False
        )

        # build decoder
        out_ch = 0
        self.decoder_group = nn.ModuleList()
        for i in range(self.max_depth):
            res = gt_resolution // 2 ** self.max_depth * 2 ** i
            in_ch, out_ch = channel_query_dict[res], channel_query_dict[res * 2]
            self.decoder_group.append(DecoderBlock(in_ch, out_ch, norm_type, act_type))
        self.out_conv = nn.Conv2d(out_ch, 3, 3, 1, 1)
        self.residual_conv = nn.Conv2d(out_ch, 3, 3, 1, 1)

        # build multi-scale vector quantizers
        self.quantizer = VectorQuantizer(
            codebook_emb_num,
            codebook_emb_dim,
            LQ_stage=False,
            use_weight=self.use_weight,
            weight_alpha=self.weight_alpha
        )
        scale_in_ch = channel_query_dict[self.codebook_scale]
        self.before_quant = nn.Conv2d(scale_in_ch, codebook_emb_dim, 1)
        self.after_quant = CombineQuantBlock(codebook_emb_dim, 0, scale_in_ch)

        # semantic loss for HQ pretrain stage
        self.conv_semantic = nn.Sequential(
            nn.Conv2d(512, 512, 1, 2, 0),
            nn.ReLU(),
        )
        self.vgg_feat_layer = 'relu4_4'
        self.vgg_feat_extractor = VGGFeatureExtractor([self.vgg_feat_layer])

    def forward(self, inputs):
        return self.encode_and_decode(inputs)

    def encode_and_decode(self, inputs):
        # code_decoder_output: vq-gan解码后的输出集合，总共输出了两次，第一次是经过离散化的并解码一次的，第二次是解码两次的
        code_decoder_output = []

        enc_feats = self.multiscale_encoder(inputs)
        feat_to_quant = self.before_quant(enc_feats)
        z_quant, codebook_loss, indices = self.quantizer(feat_to_quant)
        after_quant_feat = self.after_quant(z_quant)

        x = after_quant_feat
        for i in range(self.max_depth):
            x = self.decoder_group[i](x)
            code_decoder_output.append(x)

        out_img = self.out_conv(x)

        with torch.no_grad():
            vgg_feat = self.vgg_feat_extractor(inputs)[self.vgg_feat_layer]
        semantic_z_quant = self.conv_semantic(z_quant)
        # print("z_quant.shape")
        # print(z_quant.shape)
        # print("semantic_z_quant.shape")
        # print(semantic_z_quant.shape)
        # print("inputs.shape")
        # print(inputs.shape)
        # print("vgg_feat.shape")
        # print(vgg_feat.shape)
        semantic_loss = F.mse_loss(semantic_z_quant, vgg_feat)
        # out_img 图像生成的重建输出
        return out_img, codebook_loss, semantic_loss, feat_to_quant, z_quant, indices

    @torch.no_grad()
    def test_tile(self, input, tile_size=240, tile_pad=16):
        """
        It will first crop input images to tiles, and then process each tile.
        Finally, all the processed tiles are merged into one images.
        Modified from: https://github.com/xinntao/Real-ESRGAN/blob/master/realesrgan/utils.py
        """
        batch, channel, height, width = input.shape
        output_height = height
        output_width = width
        output_shape = (batch, channel, output_height, output_width)

        # start with black image
        output = input.new_zeros(output_shape)
        tiles_x = math.ceil(width / tile_size)
        tiles_y = math.ceil(height / tile_size)

        # loop over all tiles
        for y in range(tiles_y):
            for x in range(tiles_x):
                # extract tile from input image
                ofs_x = x * tile_size
                ofs_y = y * tile_size
                # input tile area on total image
                input_start_x = ofs_x
                input_end_x = min(ofs_x + tile_size, width)
                input_start_y = ofs_y
                input_end_y = min(ofs_y + tile_size, height)

                # input tile area on total image with padding
                input_start_x_pad = max(input_start_x - tile_pad, 0)
                input_end_x_pad = min(input_end_x + tile_pad, width)
                input_start_y_pad = max(input_start_y - tile_pad, 0)
                input_end_y_pad = min(input_end_y + tile_pad, height)

                # input tile dimensions
                input_tile_width = input_end_x - input_start_x
                input_tile_height = input_end_y - input_start_y
                tile_idx = y * tiles_x + x + 1
                input_tile = input[:, :, input_start_y_pad:input_end_y_pad, input_start_x_pad:input_end_x_pad]

                # upscale tile
                output_tile = self.test(input_tile)

                # output tile area on total image
                output_start_x = input_start_x
                output_end_x = input_end_x
                output_start_y = input_start_y
                output_end_y = input_end_y

                # output tile area without padding
                output_start_x_tile = (input_start_x - input_start_x_pad)
                output_end_x_tile = output_start_x_tile + input_tile_width
                output_start_y_tile = (input_start_y - input_start_y_pad)
                output_end_y_tile = output_start_y_tile + input_tile_height

                # put tile into output image
                output[:, :, output_start_y:output_end_y,
                output_start_x:output_end_x] = output_tile[:, :, output_start_y_tile:output_end_y_tile,
                                               output_start_x_tile:output_end_x_tile]
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

        output_vq, output, _, _, _, after_quant, index = self.encode_and_decode(inputs)
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


@ARCH_REGISTRY.register()
class RIDCP(nn.Module):
    def __init__(self,
                 *,
                 in_channel=3,
                 codebook_scale=64,
                 codebook_emb_num=1024,
                 codebook_emb_dim=512,
                 gt_resolution=256,
                 LQ_stage=False,
                 norm_type='gn',
                 act_type='silu',
                 use_quantize=True,
                 use_semantic_loss=False,
                 use_residual=True,
                 only_residual=False,
                 use_weight=False,
                 use_warp=True,
                 weight_alpha=1.0,
                 additional_encoder="RSTB",
                 **ignore_kwargs):
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
        self.multiscale_encoder = MultiScaleEncoder(
            in_channel,
            self.max_depth,
            self.gt_res,
            channel_query_dict,
            norm_type, act_type, LQ_stage,
            additional_encoder
        )
        if self.LQ_stage and self.use_residual:
            self.multiscale_decoder = MultiScaleDecoder(
                in_channel,
                self.max_depth,
                self.gt_res,
                channel_query_dict,
                norm_type, act_type, only_residual, use_warp=self.use_warp
            )

        # build decoder
        out_ch = 0
        self.decoder_group = nn.ModuleList()
        for i in range(self.max_depth):
            res = gt_resolution // 2 ** self.max_depth * 2 ** i
            in_ch, out_ch = channel_query_dict[res], channel_query_dict[res * 2]
            self.decoder_group.append(DecoderBlock(in_ch, out_ch, norm_type, act_type))

        self.out_conv = nn.Conv2d(out_ch, 3, 3, 1, 1)
        self.residual_conv = nn.Conv2d(out_ch, 3, 3, 1, 1)

        # build multi-scale vector quantizers
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

        # semantic loss for HQ pretrain stage
        self.use_semantic_loss = use_semantic_loss
        if self.use_semantic_loss:
            self.conv_semantic = nn.Sequential(
                nn.Conv2d(512, 512, 1, 1, 0),
                nn.ReLU(),
            )
            self.vgg_feat_layer = 'relu4_4'
            self.vgg_feat_extractor = VGGFeatureExtractor([self.vgg_feat_layer])

    def forward(self, inputs, gt_indices=None):
        return self.encode_and_decode(inputs, gt_indices)

    def encode_and_decode(self, inputs, gt_indices=None):
        # code_decoder_output: vq-gan解码后的输出集合，总共输出了两次，第一次是经过离散化的并解码一次的，第二次是解码两次的
        code_decoder_output = []
        semantic_loss = 0

        enc_feats = self.multiscale_encoder(inputs)
        feat_to_quant = self.before_quant(enc_feats)
        z_quant, codebook_loss, indices = self.quantizer(feat_to_quant, gt_indices)
        after_quant_feat = self.after_quant(z_quant)

        x = after_quant_feat
        for i in range(self.max_depth):
            x = self.decoder_group[i](x)
            code_decoder_output.append(x)

        out_img = self.out_conv(x)
        out_img_residual = None
        if self.LQ_stage and self.use_residual:
            if self.only_residual:
                residual_feature = self.multiscale_decoder(enc_feats, code_decoder_output)
            else:
                residual_feature = self.multiscale_decoder(enc_feats.detach(), code_decoder_output)
            out_img_residual = self.residual_conv(residual_feature)

        if self.use_semantic_loss:
            with torch.no_grad():
                vgg_feat = self.vgg_feat_extractor(inputs)[self.vgg_feat_layer]
            semantic_z_quant = self.conv_semantic(z_quant)
            semantic_loss = F.mse_loss(semantic_z_quant, vgg_feat)
        # out_img 图像生成的重建输出
        # out_img_residual 图像去雾的结果输出
        return out_img, out_img_residual, codebook_loss, semantic_loss, feat_to_quant, z_quant, indices

    @torch.no_grad()
    def test_tile(self, input, tile_size=240, tile_pad=16):
        # return self.test(input)
        """
        It will first crop input images to tiles, and then process each tile.
        Finally, all the processed tiles are merged into one images.
        Modified from: https://github.com/xinntao/Real-ESRGAN/blob/master/realesrgan/utils.py
        """
        batch, channel, height, width = input.shape
        output_height = height
        output_width = width
        output_shape = (batch, channel, output_height, output_width)

        # start with black image
        output = input.new_zeros(output_shape)
        tiles_x = math.ceil(width / tile_size)
        tiles_y = math.ceil(height / tile_size)

        # loop over all tiles
        for y in range(tiles_y):
            for x in range(tiles_x):
                # extract tile from input image
                ofs_x = x * tile_size
                ofs_y = y * tile_size
                # input tile area on total image
                input_start_x = ofs_x
                input_end_x = min(ofs_x + tile_size, width)
                input_start_y = ofs_y
                input_end_y = min(ofs_y + tile_size, height)

                # input tile area on total image with padding
                input_start_x_pad = max(input_start_x - tile_pad, 0)
                input_end_x_pad = min(input_end_x + tile_pad, width)
                input_start_y_pad = max(input_start_y - tile_pad, 0)
                input_end_y_pad = min(input_end_y + tile_pad, height)

                # input tile dimensions
                input_tile_width = input_end_x - input_start_x
                input_tile_height = input_end_y - input_start_y
                tile_idx = y * tiles_x + x + 1
                input_tile = input[:, :, input_start_y_pad:input_end_y_pad, input_start_x_pad:input_end_x_pad]

                # upscale tile
                output_tile = self.test(input_tile)

                # output tile area on total image
                output_start_x = input_start_x
                output_end_x = input_end_x
                output_start_y = input_start_y
                output_end_y = input_end_y

                # output tile area without padding
                output_start_x_tile = (input_start_x - input_start_x_pad)
                output_end_x_tile = output_start_x_tile + input_tile_width
                output_start_y_tile = (input_start_y - input_start_y_pad)
                output_end_y_tile = output_start_y_tile + input_tile_height

                # put tile into output image
                output[:, :, output_start_y:output_end_y,
                output_start_x:output_end_x] = output_tile[:, :, output_start_y_tile:output_end_y_tile,
                                               output_start_x_tile:output_end_x_tile]
        return output

    @torch.no_grad()
    def test(self, input):
        org_use_semantic_loss = self.use_semantic_loss
        self.use_semantic_loss = False

        # padding to multiple of window_size * 8
        wsz = 32
        _, _, h_old, w_old = input.shape
        h_pad = (h_old // wsz + 1) * wsz - h_old
        w_pad = (w_old // wsz + 1) * wsz - w_old
        input = torch.cat([input, torch.flip(input, [2])], 2)[:, :, :h_old + h_pad, :]
        input = torch.cat([input, torch.flip(input, [3])], 3)[:, :, :, :w_old + w_pad]

        output_vq, output, _, _, _, after_quant, index = self.encode_and_decode(input, None)
        self.use_semantic_loss = org_use_semantic_loss
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

