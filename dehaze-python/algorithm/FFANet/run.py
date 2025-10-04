from io import BytesIO

import torch
import torchvision
from PIL import Image

from app.utils.image import postprocess_image
from config import Config
from .FFA import FFA


def get_model(model_path: str):
    net = FFA(gps=3, blocks=19)
    net.to(Config.DEVICE)
    net = torch.nn.DataParallel(net)
    ckp = torch.load(model_path)
    net.load_state_dict(ckp['model'])
    net.eval()
    return net


def dehaze(haze_image: BytesIO, model_path: str) -> BytesIO:
    net = get_model(model_path)
    haze = Image.open(haze_image).convert('RGB')
    haze = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.64, 0.6, 0.58],std=[0.14,0.15, 0.152])
    ])(haze)[None, ::]

    haze = haze.to(Config.DEVICE)
    with torch.no_grad():
        pred = net(haze)

    return postprocess_image(pred)
