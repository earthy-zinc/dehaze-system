import os
from collections import OrderedDict

import torch
from PIL import Image
from torchvision.transforms import Compose, CenterCrop, ToTensor

from .model.CMFNet import CMFNet
from global_variable import MODEL_PATH, DEVICE
import torchvision.utils as torch_utils


def get_model(model_path: str):
    net = CMFNet()
    net = net.to(DEVICE)

    checkpoint = torch.load(model_path)
    try:
        net.load_state_dict(checkpoint["state_dict"])
    except RuntimeError:
        state_dict = checkpoint["state_dict"]
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        net.load_state_dict(new_state_dict)
    net.eval()
    return net


def dehaze(haze_image_path: str, output_image_path: str, model_path: str):
    net = get_model(model_path)
    haze = Image.open(haze_image_path).convert('RGB')
    width, height = haze.width, haze.height
    min_size = min(width, height)
    print(width, height, min_size)
    haze = Compose([
        CenterCrop((min_size, min_size)),
        ToTensor()
    ])(haze)[None, ::]
    haze = haze.to(DEVICE)
    with torch.no_grad():
        pred = net(haze)
    ts = torch.squeeze(torch.clamp(pred[0], 0, 1).cpu())
    torch_utils.save_image(ts, output_image_path)
