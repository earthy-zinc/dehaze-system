from io import BytesIO

import torch

from app.utils.image import preprocess_image, postprocess_image
from config import Config
from .model import C2PNet


def get_model(model_path: str):
    net = C2PNet(gps=3, blocks=19)
    ckp = torch.load(model_path)
    net = net.to(Config.DEVICE)
    net.load_state_dict(ckp['model'])
    net.eval()
    return net


def dehaze(haze_image: BytesIO, model_path: str) -> BytesIO:
    net = get_model(model_path)
    haze = preprocess_image(haze_image)
    with torch.no_grad():
        pred = net(haze)
    return postprocess_image(pred)
