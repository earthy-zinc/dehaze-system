import os

import numpy as np
import torch
import torchvision.utils
from PIL import Image

from benchmark.AODNet.net import dehaze_net
from global_variable import MODEL_PATH, DEVICE


def get_model(model_name: str):
    # 构造模型文件的绝对路径
    model_dir = os.path.join(MODEL_PATH, model_name)
    net = dehaze_net().to(DEVICE)
    net.load_state_dict(torch.load(model_dir))
    net.eval()
    return net


def dehaze(haze_image_path: str, output_image_path: str, model_name: str = ''):
    net = get_model(model_name)

    haze = np.array(Image.open(haze_image_path).convert('RGB')) / 255.0
    haze = torch.from_numpy(haze).float()
    haze = haze.permute(2, 0, 1)
    haze = haze.cuda().unsqueeze(0)

    with torch.no_grad():
        out = net(haze)
        torchvision.utils.save_image(out, output_image_path)
