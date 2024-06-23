import os

import torch
import torchvision
from PIL import Image

from .FFA import FFA
from global_variable import MODEL_PATH, DEVICE


def get_model(model_path: str):
    net = FFA(gps=3, blocks=19)
    net.to(DEVICE)
    net = torch.nn.DataParallel(net)
    ckp = torch.load(model_path)
    net.load_state_dict(ckp['model'])
    net.eval()
    return net


def dehaze(haze_image_path: str, output_image_path: str, model_path: str):
    net = get_model(model_path)
    haze = Image.open(haze_image_path).convert('RGB')
    haze = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.64, 0.6, 0.58],std=[0.14,0.15, 0.152])
    ])(haze)[None, ::]

    haze = haze.to(DEVICE)
    with torch.no_grad():
        pred = net(haze)
    ts = torch.squeeze(pred.clamp(0, 1).cpu())
    torchvision.utils.save_image(ts, output_image_path)
