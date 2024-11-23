from io import BytesIO

import torch
import torch.nn.functional as F

from app.utils.image import preprocess_image, postprocess_image
from config import Config
from .model.backbone import Backbone


def get_model(model_path: str):
    net = Backbone().to(Config.DEVICE)
    ckpt = torch.load(model_path, map_location='cpu')
    net.load_state_dict(ckpt)
    net.eval()
    return net


def pad_img(x, patch_size):
    _, _, h, w = x.size()
    mod_pad_h = (patch_size - h % patch_size) % patch_size
    mod_pad_w = (patch_size - w % patch_size) % patch_size
    x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
    return x


def dehaze(haze_image: BytesIO, model_path: str) -> BytesIO:
    net = get_model(model_path)
    haze = preprocess_image(haze_image)
    with torch.no_grad():
        hazy_img = pad_img(haze, 4)
        output = net(hazy_img)
    return postprocess_image(output)
