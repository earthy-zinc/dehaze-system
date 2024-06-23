import os

import torch
import torchvision.transforms as tfs
import torchvision.utils as torch_utils
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from .model import C2PNet
from global_variable import MODEL_PATH, DEVICE


def get_model(model_path: str):
    net = C2PNet(gps=3, blocks=19)
    ckp = torch.load(model_path)
    net = net.to(DEVICE)
    net.load_state_dict(ckp['model'])
    net.eval()
    return net


def dehaze(haze_image_path: str, output_image_path: str, model_path: str = 'C2PNet/OTS.pkl'):
    net = get_model(model_path)
    haze = Image.open(haze_image_path).convert('RGB')
    haze = tfs.ToTensor()(haze)[None, ::]
    haze = haze.to(DEVICE)
    with torch.no_grad():
        pred = net(haze)
    ts = torch.squeeze(pred.clamp(0, 1).cpu())
    torch_utils.save_image(ts, output_image_path)
