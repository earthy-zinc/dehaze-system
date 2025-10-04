from io import BytesIO

import torch

from app.utils.image import preprocess_image, postprocess_image
from config import Config
from .lightdehazeNet import LightDehaze_Net


def get_model(model_path: str):
    net = LightDehaze_Net()
    net = net.to(Config.DEVICE)
    net.load_state_dict(torch.load(model_path))
    net.eval()
    return net


def dehaze(haze_image: BytesIO, model_path: str) -> BytesIO:
    net = get_model(model_path)
    haze = preprocess_image(haze_image).clip(0, 1)
    with torch.no_grad():
        out = net(haze)
    return postprocess_image(out)
