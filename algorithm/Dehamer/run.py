import os

import numpy as np
import torch
import torchvision.utils
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Normalize

from benchmark.Dehamer.swin_unet import UNet_emb
from global_variable import MODEL_PATH, DEVICE


def get_model(model_name: str):
    # 构造模型文件的绝对路径
    model_dir = os.path.join(MODEL_PATH, model_name)
    net = UNet_emb()
    net = net.to(DEVICE)
    net.load_state_dict(torch.load(model_dir), strict=False)
    net.eval()
    return net


# TODO 有点问题
def dehaze(haze_image_path: str, output_image_path: str, model_name: str = ''):
    net = get_model(model_name)

    haze = Image.open(haze_image_path).convert('RGB')
    a = haze.size
    a_0 =a[1] - np.mod(a[1],16)
    a_1 =a[0] - np.mod(a[0],16)
    haze_crop_img = haze.crop((0, 0, 0 + a_1, 0+a_0))
    transform_haze = Compose([
        ToTensor() ,
        Normalize((0.64, 0.6, 0.58), (0.14,0.15, 0.152))
    ])
    haze = transform_haze(haze_crop_img)[None, ::]

    haze = haze.to(DEVICE)
    with torch.no_grad():
        out = net(haze)
        out = torch.squeeze(out.clamp(0, 1).cpu())
        torchvision.utils.save_image(out, output_image_path)
