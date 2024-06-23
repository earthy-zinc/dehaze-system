import os

import numpy as np
import torch
import torchvision.utils
from PIL import Image

from net import dehaze_net
from global_variable import DEVICE


def get_model(model_path: str):
    # 构造模型文件的绝对路径
    net = dehaze_net().to(DEVICE)
    net.load_state_dict(torch.load(model_path))
    net.eval()
    return net


def dehaze(haze_image_path: str, output_image_path: str, model_path: str = ''):
    net = get_model(model_path)

    haze = np.array(Image.open(haze_image_path).convert('RGB')) / 255.0
    haze = torch.from_numpy(haze).float()
    haze = haze.permute(2, 0, 1)
    haze = haze.cuda().unsqueeze(0)

    with torch.no_grad():
        out = net(haze)
        torchvision.utils.save_image(out, output_image_path)
