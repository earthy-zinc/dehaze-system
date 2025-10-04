# ------------------------------------------------------------------------------------
# Enhancing Transformers
# Copyright (c) 2022 Thuan H. Nguyen. All Rights Reserved.
# Licensed under the MIT License [see LICENSE for details]
# ------------------------------------------------------------------------------------
# Modified from Taming Transformers (https://github.com/CompVis/taming-transformers)
# Copyright (c) 2020 Patrick Esser and Robin Rombach and Björn Ommer. All Rights Reserved.
# ------------------------------------------------------------------------------------

from typing import List, Tuple, Dict, Any, Optional
from omegaconf import OmegaConf

import PIL
import torch
import torch.nn as nn
from torch import FloatTensor
from torch.optim import lr_scheduler
from torchvision import transforms as T
import pytorch_lightning as pl
from layer import ViTEncoder, ViTDecoder
from quantizer import VectorQuantizer, GumbelQuantizer


class ViTVQ(pl.LightningModule):
    def __init__(self, image_key: str, image_size: int, patch_size: int,
                 encoder: Dict, decoder: Dict, quantizer: Dict,
                 path: Optional[str] = None, ignore_keys=None) -> None:
        super().__init__()
        if ignore_keys is None:
            ignore_keys = list()
        self.path = path
        self.ignore_keys = ignore_keys
        self.image_key = image_key

        self.encoder = ViTEncoder(image_size=image_size, patch_size=patch_size, **encoder)
        self.decoder = ViTDecoder(image_size=image_size, patch_size=patch_size, **decoder)
        self.quantizer = VectorQuantizer(**quantizer)
        self.pre_quant = nn.Linear(encoder.dim, quantizer.embed_dim)
        self.post_quant = nn.Linear(quantizer.embed_dim, decoder.dim)

        if path is not None:
            self.init_from_ckpt(path, ignore_keys)

    def forward(self, x: torch.FloatTensor) -> Tuple[FloatTensor, FloatTensor]:
        quant, diff = self.encode(x)
        dec = self.decode(quant)
        return dec, diff

    def init_from_ckpt(self, path: str, ignore_keys: List[str] = list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def encode(self, x: torch.FloatTensor) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        h = self.encoder(x)
        h = self.pre_quant(h)
        quant, emb_loss, _ = self.quantizer(h)
        return quant, emb_loss

    def decode(self, quant: torch.FloatTensor) -> torch.FloatTensor:
        quant = self.post_quant(quant)
        dec = self.decoder(quant)
        return dec

    def encode_codes(self, x: torch.FloatTensor) -> torch.LongTensor:
        h = self.encoder(x)
        h = self.pre_quant(h)
        _, _, codes = self.quantizer(h)
        return codes

    def decode_codes(self, code: torch.LongTensor) -> torch.FloatTensor:
        quant = self.quantizer.embedding(code)
        quant = self.quantizer.norm(quant)
        if self.quantizer.use_residual:
            quant = quant.sum(-2)
        dec = self.decode(quant)
        return dec


def base_vit_vq():
    encoder = {
        'dim': 768,
        'depth': 12,
        'head': 12,
        'mlp_dim': 3072,
    }
    decoder = encoder.copy()
    quantizer = {
        'embed_dim': 32,
        'n_embed': 8192,
    }
    return ViTVQ(image_key='image', image_size=256, patch_size=8,
                 encoder=encoder, decoder=decoder, quantizer=quantizer)
