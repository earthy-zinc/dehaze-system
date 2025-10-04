from io import BytesIO

import torch

from app.utils.image import postprocess_image, preprocess_image
from config import Config
from .net import dehaze_net


def get_model(model_path: str):
    # 构造模型文件的绝对路径
    net = dehaze_net().to(Config.DEVICE)
    net.load_state_dict(torch.load(model_path))
    net.eval()
    return net


def dehaze(haze_image: BytesIO, model_path: str) -> BytesIO:
    net = get_model(model_path)
    haze = preprocess_image(haze_image)
    with torch.no_grad():
        out = net(haze)
    return postprocess_image(out)
