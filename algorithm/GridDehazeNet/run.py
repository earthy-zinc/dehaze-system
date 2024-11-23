from io import BytesIO

import torch
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Normalize

from app.utils.image import postprocess_image
from config import Config
from .model import GridDehazeNet


def get_model(model_path: str):
    net = GridDehazeNet(height=3, width=6, num_dense_layer=4, growth_rate=16)
    net = net.to(Config.DEVICE)

    net = torch.nn.DataParallel(net, device_ids=Config.DEVICE_ID)

    net.load_state_dict(torch.load(model_path))
    net.eval()
    return net


def dehaze(haze_image: BytesIO, model_path: str) -> BytesIO:
    net = get_model(model_path)
    haze = Image.open(haze_image).convert('RGB')
    transform_haze = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    haze = transform_haze(haze)[None, ::]
    haze = haze.to(DEVICE)
    with torch.no_grad():
        out = net(haze)
    return postprocess_image(out)




