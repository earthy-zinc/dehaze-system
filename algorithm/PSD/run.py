import os
from io import BytesIO

import numpy as np
import torch
import torchvision.utils as torch_utils
from PIL import Image
from torch import nn
from torchvision.transforms import Compose, ToTensor, Normalize

from app.utils.image import postprocess_image
from .FFA import FFANet
from .GCA import GCANet
from .MSBDN import MSBDNNet
from config import Config


def get_model(model_path: str):
    if model_path.find("MSBDN") != -1:
        print(model_path, '加载模型MSBDN')
        net = MSBDNNet()
        name = 'MSBDN'
    elif model_path.find("FFANET") != -1:
        print(model_path, '加载FFANET')
        net = FFANet(3, 19)
        name = 'FFANET'
    else:
        print(model_path, '加载GCANET')
        net = GCANet(in_c=4, out_c=3, only_residual=True)
        name = 'GCANET'
    net = net.to(Config.DEVICE)
    net = nn.DataParallel(net, device_ids=Config.DEVICE_ID)
    net.load_state_dict(torch.load(model_path))
    net.eval()
    return net, name

# TODO 不知道为何，在神经网络中间层有张量形状不一致的问题，等待进一步处理
def dehaze(haze_image: BytesIO, model_path: str) -> BytesIO:
    net, name = get_model(model_path)

    haze_img = Image.open(haze_image).convert('RGB')
    haze_reshaped = haze_img.resize((256, 256), Image.LANCZOS)
    transform_haze = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    haze = transform_haze(haze_img)[None, ::]
    haze_reshaped = transform_haze(haze_reshaped)[None, ::]
    haze.to(Config.DEVICE)
    haze_reshaped.to(Config.DEVICE)

    with torch.no_grad():
        if name == 'GCANET':
            print('GCANET测试中')
            pred = net(haze, 0, True, False)
            dehaze = pred.float().round().clamp(0, 255)
            return postprocess_image(dehaze[0])
        else:
            _, pred, _, _, _ = net(haze, haze_reshaped, True)
            return postprocess_image(pred)

