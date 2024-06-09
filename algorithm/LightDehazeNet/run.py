import os

import numpy as np
import torch
import torchvision.utils as torch_utils
from PIL import Image

from benchmark.LightDehazeNet.lightdehazeNet import LightDehaze_Net
from global_variable import MODEL_PATH, DEVICE


def get_model(model_name: str):
    # 构造模型文件的绝对路径
    model_dir = os.path.join(MODEL_PATH, model_name)
    net = LightDehaze_Net()
    net = net.to(DEVICE)
    net.load_state_dict(torch.load(model_dir))
    net.eval()
    return net


def dehaze(haze_image_path: str, output_image_path: str, model_name: str = 'C2PNet/OTS.pkl'):
    net = get_model(model_name)

    haze = Image.open(haze_image_path).convert('RGB')
    hazy_image = (np.asarray(haze) / 255.0)
    hazy_image = torch.from_numpy(hazy_image).float()
    hazy_image = hazy_image.permute(2, 0, 1)
    hazy_image = hazy_image.cuda().unsqueeze(0)
    with torch.no_grad():
        out = net(hazy_image)
    ts = torch.squeeze(out.clamp(0, 1).cpu())
    torch_utils.save_image(ts, output_image_path)
