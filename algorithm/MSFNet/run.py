from io import BytesIO

import torch
import torchvision.transforms as tfs
from PIL import Image
from torch.autograd import Variable

from app.utils.image import postprocess_image
from config import Config
from .net import final_Net


def get_model(model_path: str):
    net = final_Net()
    net.to(Config.DEVICE)
    # Load pretrained models
    net.load_state_dict(torch.load(model_path)["state_dict"])
    net.eval()
    return net


def dehaze(haze_image: BytesIO, model_path: str) -> BytesIO:
    net = get_model(model_path)
    haze = Image.open(haze_image).convert('RGB')
    transform = tfs.Compose([
        tfs.ToTensor(),
        tfs.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    haze = transform(haze).unsqueeze_(0)
    haze = Variable(haze.to(Config.DEVICE))
    with torch.no_grad():
        _, _, output = net(haze)
    return postprocess_image(output)
