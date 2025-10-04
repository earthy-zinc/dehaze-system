import os
from collections import OrderedDict
from io import BytesIO

import torch

from app.utils.image import preprocess_image, postprocess_image
from config import Config


def single(save_dir):
    state_dict = torch.load(save_dir, map_location=torch.device(Config.DEVICE))
    # print(state_dict)
    new_state_dict = OrderedDict()

    for k, v in state_dict.items():
        name = k[7:]
        new_state_dict[name] = v

    return new_state_dict


def get_model(model_path: str):

    from .models.LKD import LKD_b, LKD_l, LKD_s, LKD_t
    net = eval(os.path.basename(model_path).replace('-', '_').replace('.pth', ''))()
    net = net.to(Config.DEVICE)
    net.load_state_dict(single(model_path), strict=False)
    torch.cuda.empty_cache()
    net.eval()
    return net


def dehaze(haze_image: BytesIO, model_path: str) -> BytesIO:
    net = get_model(model_path)
    haze = preprocess_image(haze_image)
    with torch.no_grad():
        output = net(haze).clamp_(-1, 1)
        # [-1, 1] to [0, 1]
        output = output * 0.5 + 0.5
    return postprocess_image(output)
