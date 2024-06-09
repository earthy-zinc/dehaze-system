import os

import torch
import torchvision.utils
from PIL import Image

from benchmark.DCPDN.dehaze22 import dehaze as DCPDN
from global_variable import MODEL_PATH, DEVICE


def get_model(model_name: str):
    # 构造模型文件的绝对路径
    model_dir = os.path.join(MODEL_PATH, model_name)
    net = DCPDN(3, 3, 64)
    net.to(DEVICE)
    net.load_state_dict(torch.load(model_dir))
    net.eval()
    return net


# TODO 该模型torch版本太低，无法正确加载预训练模型
def dehaze(haze_image_path: str, output_image_path: str, model_name: str = ''):
    net = get_model(model_name)

    with torch.no_grad():
        haze = Image.open(haze_image_path).convert('RGB')
        haze = torchvision.transforms.ToTensor()(haze)[None, ::]
        haze = haze.to(DEVICE)
        out, _, _, _ = net(haze)
        torch.squeeze(out.clamp(0, 1).cpu())
        torchvision.utils.save_image(out, output_image_path)
