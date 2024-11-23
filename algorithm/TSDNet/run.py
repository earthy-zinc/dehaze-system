from io import BytesIO

import torch

from app.utils.image import preprocess_image, postprocess_image
from config import Config
from .model import GNet


def get_model(model_path: str):
    net = GNet()
    net.to(Config.DEVICE)
    net = torch.nn.DataParallel(net, device_ids=Config.DEVICE_ID)
    # 将模型参数赋值进net
    model_info = torch.load(model_path)
    net.load_state_dict(model_info['state_dict'])
    net.eval()
    return net


def dehaze(haze_image: BytesIO, model_path: str) -> BytesIO:
    net = get_model(model_path)
    haze = preprocess_image(haze_image).clip(0, 1)
    with torch.no_grad():
        _, _, _, out = net(haze)
    return postprocess_image(out)
