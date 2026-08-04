from io import BytesIO

import torch

from algorithm.image_utils import preprocess_image, postprocess_image
from algorithm.config import Config
from .lightdehazeNet import LightDehaze_Net


def get_model(model_path: str):
    net = LightDehaze_Net()
    net = net.to(Config.DEVICE)
    net.load_state_dict(torch.load(model_path, weights_only=False))
    net.eval()
    return net


def dehaze(haze_image: BytesIO, model_path: str) -> BytesIO:
    net = get_model(model_path)
    haze = preprocess_image(haze_image).clip(0, 1)
    with torch.no_grad():
        out = net(haze)
    return postprocess_image(out)
