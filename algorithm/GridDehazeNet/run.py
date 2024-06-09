import os

import torch
import torchvision.utils
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Normalize

from benchmark.GridDehazeNet.model import GridDehazeNet
from global_variable import MODEL_PATH, DEVICE, DEVICE_ID


def get_model(model_name: str):
    # 构造模型文件的绝对路径
    model_dir = os.path.join(MODEL_PATH, model_name)
    net = GridDehazeNet(height=3, width=6, num_dense_layer=4, growth_rate=16)
    net = net.to(DEVICE)

    net = torch.nn.DataParallel(net, device_ids=DEVICE_ID)

    net.load_state_dict(torch.load(model_dir))
    net.eval()
    return net


def dehaze(haze_image_path: str, output_image_path: str, model_name: str = ''):
    net = get_model(model_name)
    haze = Image.open(haze_image_path).convert('RGB')
    transform_haze = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    haze = transform_haze(haze)[None, ::]
    haze = haze.to(DEVICE)
    with torch.no_grad():
        out = net(haze)
    out = torch.squeeze(out.clamp(0, 1).cpu())
    torchvision.utils.save_image(out, output_image_path)




