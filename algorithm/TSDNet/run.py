import os

import numpy as np
import torch
from PIL import Image
from torchvision.utils import save_image

from .model import GNet
from global_variable import DEVICE, DEVICE_ID


def get_model(model_path: str):
    net = GNet()
    net.to(DEVICE)
    net = torch.nn.DataParallel(net, device_ids=DEVICE_ID)
    # 将模型参数赋值进net
    model_info = torch.load(model_path)
    net.load_state_dict(model_info['state_dict'])
    net.eval()
    return net


def dehaze(haze_image_path: str, output_image_path: str, model_path: str):
    net = get_model(model_path)
    haze = np.array(Image.open(haze_image_path)) / 255
    haze = torch.Tensor(haze.transpose(2, 0, 1)[np.newaxis, :, :, :]).to(DEVICE)
    with torch.no_grad():
        _, _, _, out = net(haze)
    save_image(out, output_image_path)
