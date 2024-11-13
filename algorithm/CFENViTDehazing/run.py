import os

import torch
import torchvision.transforms as tfs
import torchvision.utils as torch_utils
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from .models.ipt import define_G
from .models.common import default_conv
from global_variable import MODEL_PATH, DEVICE
from .test_options import TestOptions


def get_model(model_path: str):
    opt = TestOptions().parse()
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip
    opt.display_id = -1  # no visdom display
    net = define_G(opt, default_conv)
    ckp = torch.load(model_path)
    net = net.to(DEVICE)
    net.load_state_dict(ckp)
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

