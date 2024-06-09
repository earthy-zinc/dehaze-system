import os

import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as tfs
import torchvision.utils as torch_utils

from global_variable import MODEL_PATH, DEVICE
from .model.backbone import Backbone


def get_model(model_name: str):
    model_dir = os.path.join(MODEL_PATH, model_name)
    net = Backbone().to(DEVICE)
    ckpt = torch.load(model_dir, map_location='cpu')
    net.load_state_dict(ckpt)
    net.eval()
    return net


def pad_img(x, patch_size):
    _, _, h, w = x.size()
    mod_pad_h = (patch_size - h % patch_size) % patch_size
    mod_pad_w = (patch_size - w % patch_size) % patch_size
    x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
    return x


def dehaze(haze_image_path: str, output_image_path: str, model_name: str = 'DEA-Net/HAZE4K/PSNR3426_SSIM9885.pth'):
    net = get_model(model_name)
    haze = Image.open(haze_image_path).convert('RGB')
    haze = tfs.ToTensor()(haze)[None, ::]
    haze = haze.to(DEVICE)

    with torch.no_grad():
        H, W = haze.shape[2:]
        hazy_img = pad_img(haze, 4)
        output = net(hazy_img)
        output = output.clamp(0, 1)
        output = output[:, :, :H, :W]
        torch_utils.save_image(output, output_image_path)
