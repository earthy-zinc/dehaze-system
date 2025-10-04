from collections import OrderedDict
from io import BytesIO

import torch
from PIL import Image
from torchvision.transforms import Compose, CenterCrop, ToTensor

from app.utils.image import postprocess_image
from config import Config
from .model.CMFNet import CMFNet


def get_model(model_path: str):
    net = CMFNet()
    net = net.to(Config.DEVICE)

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


def dehaze(haze_image: BytesIO, model_path: str) -> BytesIO:
    net = get_model(model_path)

    haze = Image.open(haze_image).convert('RGB')
    width, height = haze.width, haze.height
    min_size = min(width, height)
    haze = Compose([
        CenterCrop((min_size, min_size)),
        ToTensor()
    ])(haze)[None, ::]
    haze = haze.to(Config.DEVICE)

    with torch.no_grad():
        pred = net(haze)
    return postprocess_image(pred[0])
