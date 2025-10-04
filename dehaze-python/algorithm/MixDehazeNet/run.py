import os
from collections import OrderedDict
from io import BytesIO

import torch
from config import Config

from app.utils.image import preprocess_image, postprocess_image


def get_model(model_path: str):
    from .model import MixDehazeNet_t, MixDehazeNet_s, MixDehazeNet_b, MixDehazeNet_l
    net = eval(os.path.basename(model_path).replace('-', '_').replace('.pth', ''))()
    net = net.to(Config.DEVICE)
    state_dict = torch.load(model_path)['state_dict']
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]
        new_state_dict[name] = v

    net.load_state_dict(new_state_dict)
    net.eval()
    return net


def dehaze(haze_image: BytesIO, model_path: str) -> BytesIO:
    net = get_model(model_path)
    haze = preprocess_image(haze_image)
    with torch.no_grad():
        pred = net(haze)
    return postprocess_image(pred)
