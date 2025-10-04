from io import BytesIO

import torch

from app.utils.image import preprocess_image, postprocess_image
from config import Config
from .config import get_config
from .model import fusion_refine
from .models import build_model


def get_model(model_path: str):
    config = get_config()
    swv2_model = build_model(config)
    net = fusion_refine(swv2_model, '')
    net = net.to(Config.DEVICE)
    net.load_state_dict(torch.load(model_path))
    net.eval()
    return net


def dehaze(haze_image: BytesIO, model_path: str) -> BytesIO:
    net = get_model(model_path)
    haze = preprocess_image(haze_image)
    with torch.no_grad():
        pred = net(haze)
    return postprocess_image(pred)
