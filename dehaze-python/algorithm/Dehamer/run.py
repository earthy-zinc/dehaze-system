from io import BytesIO

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Normalize

from app.utils.image import postprocess_image
from config import Config
from .swin_unet import UNet_emb


def get_model(model_path: str):
    net = UNet_emb()
    net = net.to(Config.DEVICE)
    net.load_state_dict(torch.load(model_path), strict=False)
    net.eval()
    return net


# TODO 有点问题
def dehaze(haze_image: BytesIO, model_path: str) -> BytesIO:
    net = get_model(model_path)

    haze = Image.open(haze_image).convert('RGB')
    a = haze.size
    a_0 =a[1] - np.mod(a[1],16)
    a_1 =a[0] - np.mod(a[0],16)
    haze_crop_img = haze.crop((0, 0, 0 + a_1, 0+a_0))
    transform_haze = Compose([
        ToTensor() ,
        Normalize((0.64, 0.6, 0.58), (0.14,0.15, 0.152))
    ])
    haze = transform_haze(haze_crop_img)[None, ::]
    haze = haze.to(Config.DEVICE)

    with torch.no_grad():
        out = net(haze)
    return postprocess_image(out)
