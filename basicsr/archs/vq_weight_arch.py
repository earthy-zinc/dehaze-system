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
class VQWeightNet(nn.Module):
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
            norm_type, act_type, LQ_stage
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
        self.decoder_group = nn.ModuleList()
        for i in range(self.max_depth):
            res = gt_resolution // 2**self.max_depth * 2**i
            in_ch, out_ch = channel_query_dict[res], channel_query_dict[res * 2]
            self.decoder_group.append(DecoderBlock(in_ch, out_ch, norm_type, act_type))

        self.out_conv = nn.Conv2d(out_ch, 3, 3, 1, 1)
        self.residual_conv = nn.Conv2d(out_ch, 3, 3, 1, 1)

        # build multi-scale vector quantizers
        self.quantize_group = nn.ModuleList()
        self.before_quant_group = nn.ModuleList()
        self.after_quant_group = nn.ModuleList()

        for scale in range(0, 1):
            quantize = VectorQuantizer(
                codebook_emb_num,
                codebook_emb_dim,
                LQ_stage=self.LQ_stage,
                use_weight=self.use_weight,
                weight_alpha=self.weight_alpha
            )
            self.quantize_group.append(quantize)

            scale_in_ch = channel_query_dict[self.codebook_scale]
            if scale == 0:
                quant_conv_in_ch = scale_in_ch
                comb_quant_in_ch1 = codebook_emb_dim
                comb_quant_in_ch2 = 0
            else:
                quant_conv_in_ch = scale_in_ch * 2
                comb_quant_in_ch1 = codebook_emb_dim
                comb_quant_in_ch2 = codebook_emb_dim

            self.before_quant_group.append(nn.Conv2d(quant_conv_in_ch, codebook_emb_dim, 1))
            self.after_quant_group.append(CombineQuantBlock(comb_quant_in_ch1, comb_quant_in_ch2, scale_in_ch))

        # semantic loss for HQ pretrain stage
        self.use_semantic_loss = use_semantic_loss
        if use_semantic_loss:
            self.conv_semantic = nn.Sequential(
                nn.Conv2d(512, 512, 1, 1, 0),
                nn.ReLU(),
            )
            self.vgg_feat_layer = 'relu4_4'
            self.vgg_feat_extractor = VGGFeatureExtractor([self.vgg_feat_layer])

    def encode_and_decode(self, input, gt_indices=None, current_iter=None, weight_alpha=None):
        # if self.training:
        #     for p in self.multiscale_encoder.parameters():
        #         p.requires_grad = True
        enc_feats = self.multiscale_encoder(input)

        if self.use_semantic_loss:
            with torch.no_grad():
                vgg_feat = self.vgg_feat_extractor(input)[self.vgg_feat_layer]

        codebook_loss_list = []
        indices_list = []
        semantic_loss_list = []
        code_decoder_output = []

        quant_idx = 0
        prev_dec_feat = None
        prev_quant_feat = None
        out_img = None
        out_img_residual = None

        x = enc_feats
        for i in range(self.max_depth):
            cur_res = self.gt_res // 2**self.max_depth * 2**i
            if cur_res == self.codebook_scale:  # needs to perform quantize
                if prev_dec_feat is not None:
                    before_quant_feat = torch.cat((x, prev_dec_feat), dim=1)
                else:
                    before_quant_feat = x
                feat_to_quant = self.before_quant_group[quant_idx](before_quant_feat)

                if weight_alpha is not None:
                    self.weight_alpha = weight_alpha
                if gt_indices is not None:
                    z_quant, codebook_loss, indices = self.quantize_group[quant_idx](feat_to_quant, gt_indices[quant_idx])
                else:
                    z_quant, codebook_loss, indices = self.quantize_group[quant_idx](feat_to_quant)

                if self.use_semantic_loss:
                    semantic_z_quant = self.conv_semantic(z_quant)
                    semantic_loss = F.mse_loss(semantic_z_quant, vgg_feat)
                    semantic_loss_list.append(semantic_loss)

                if not self.use_quantize:
                    z_quant = feat_to_quant

                after_quant_feat = self.after_quant_group[quant_idx](z_quant, prev_quant_feat)

                codebook_loss_list.append(codebook_loss)
                indices_list.append(indices)

                quant_idx += 1
                prev_quant_feat = z_quant
                x = after_quant_feat

            x = self.decoder_group[i](x)
            code_decoder_output.append(x)
            prev_dec_feat = x

        out_img = self.out_conv(x)

        if self.LQ_stage and self.use_residual:
            if self.only_residual:
                residual_feature = self.multiscale_decoder(enc_feats, code_decoder_output)
            else:
                residual_feature = self.multiscale_decoder(enc_feats.detach(), code_decoder_output)
            out_img_residual = self.residual_conv(residual_feature)

        if len(codebook_loss_list) > 0:
            codebook_loss = sum(codebook_loss_list)
        else:
            codebook_loss = 0
        semantic_loss = sum(semantic_loss_list) if len(semantic_loss_list) else codebook_loss * 0

        return out_img, out_img_residual, codebook_loss, semantic_loss, feat_to_quant, z_quant, indices_list

    def decode_indices(self, indices):
        assert len(indices.shape) == 4, f'shape of indices must be (b, 1, h, w), but got {indices.shape}'

        z_quant = self.quantize_group[0].get_codebook_entry(indices)
        x = self.after_quant_group[0](z_quant)

        for m in self.decoder_group:
            x = m(x)
        out_img = self.out_conv(x)
        return out_img

    @torch.no_grad()
    def test_tile(self, input, tile_size=240, tile_pad=16):
        # return self.test(input)
        """It will first crop input images to tiles, and then process each tile.
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
    def test(self, input, weight_alpha=None):
        org_use_semantic_loss = self.use_semantic_loss
        self.use_semantic_loss = False

        # padding to multiple of window_size * 8
        wsz = 32
        _, _, h_old, w_old = input.shape
        h_pad = (h_old // wsz + 1) * wsz - h_old
        w_pad = (w_old // wsz + 1) * wsz - w_old
        input = torch.cat([input, torch.flip(input, [2])], 2)[:, :, :h_old + h_pad, :]
        input = torch.cat([input, torch.flip(input, [3])], 3)[:, :, :, :w_old + w_pad]

        output_vq, output, _, _, _, after_quant, index = self.encode_and_decode(input, None, None, weight_alpha=weight_alpha)

        if output is not None:
            output = output[..., :h_old, :w_old]
        if output_vq is not None:
            output_vq = output_vq[..., :h_old, :w_old]

        self.use_semantic_loss = org_use_semantic_loss
        return output, index

    def forward(self, input, gt_indices=None, weight_alpha=None):

        if gt_indices is not None:
            # in LQ training stage, need to pass GT indices for supervise.
            dec, dec_residual, codebook_loss, semantic_loss, quant_before_feature, quant_after_feature, indices = self.encode_and_decode(input, gt_indices, weight_alpha=weight_alpha)
        else:
            # in HQ stage, or LQ test stage, no GT indices needed.
            dec, dec_residual, codebook_loss, semantic_loss, quant_before_feature, quant_after_feature, indices = self.encode_and_decode(input, weight_alpha=weight_alpha)

        return dec, dec_residual, codebook_loss, semantic_loss, quant_before_feature, quant_after_feature, indices
