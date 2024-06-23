import os

import torch
from PIL import Image

import torchvision.transforms as tfs
import torchvision.utils as torch_utils

from .config import get_config
from .model import fusion_refine
from global_variable import DEVICE, MODEL_PATH
from .models import build_model


def get_model(model_path: str):
    config = get_config()
    swv2_model = build_model(config)
    net = fusion_refine(swv2_model, '')
    net = net.to(DEVICE)
    net.load_state_dict(torch.load(model_path))
    net.eval()
    return net


def dehaze(haze_image_path: str, output_image_path: str, model_path: str):
    net = get_model(model_path)
    haze = Image.open(haze_image_path).convert('RGB')
    haze = tfs.ToTensor()(haze)[None, ::]
    haze = haze.to(DEVICE)
    with torch.no_grad():
        pred = net(haze)
    ts = torch.squeeze(pred.clamp(0, 1).cpu())
    torch_utils.save_image(ts, output_image_path)
