import os

import numpy as np
import torch
from PIL import Image
from torchvision.utils import save_image

from benchmark.TSDNet.model import GNet
from global_variable import MODEL_PATH, DEVICE, DEVICE_ID


def get_model(model_name: str):
    # 构造模型文件的绝对路径
    model_dir = os.path.join(MODEL_PATH, model_name)

    net = GNet()
    net.to(DEVICE)
    net = torch.nn.DataParallel(net, device_ids=DEVICE_ID)
    # 将模型参数赋值进net
    model_info = torch.load(model_dir)
    net.load_state_dict(model_info['state_dict'])
    net.eval()
    return net


def dehaze(haze_image_path: str, output_image_path: str, model_name: str = ''):
    net = get_model(model_name)
    haze = np.array(Image.open(haze_image_path)) / 255
    haze = torch.Tensor(haze.transpose(2, 0, 1)[np.newaxis, :, :, :]).to(DEVICE)
    with torch.no_grad():
        _, _, _, out = net(haze)
    save_image(out, output_image_path)
