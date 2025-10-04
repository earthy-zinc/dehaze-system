from io import BytesIO

import torch
import torchvision.transforms as transforms
from PIL import Image

from app.utils.image import postprocess_image
from .model.networks import ResnetGenerator
from config import Config


def get_model(model_path: str, img_size):
    params = torch.load(model_path)
    genA2B = ResnetGenerator(input_nc=3, output_nc=3, ngf=64, n_blocks=4, img_size=img_size,
                             light=True).to(Config.DEVICE)
    genA2B.load_state_dict(params['genA2B'])
    genA2B.eval()
    return genA2B


def dehaze(haze_image: BytesIO, model_path: str) -> BytesIO:
    haze = Image.open(haze_image).convert('RGB')
    img_size = min(haze.width, haze.height)
    haze = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])(haze).unsqueeze(0).to(Config.DEVICE)
    net = get_model(model_path, img_size)
    with torch.no_grad():
        output, _, _ = net(haze)
    return postprocess_image(output)
