import os

import torch
import torchvision.transforms as tfs
import torchvision.utils as torch_utils
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from benchmark.C2PNet.model import C2PNet
from global_variable import MODEL_PATH, DEVICE


def get_model(model_name: str):
    # 构造模型文件的绝对路径
    model_dir = os.path.join(MODEL_PATH, model_name)
    net = C2PNet(gps=3, blocks=19)
    ckp = torch.load(model_dir)
    net = net.to(DEVICE)
    net.load_state_dict(ckp['model'])
    net.eval()
    return net


def dehaze(haze_image_path: str, output_image_path: str, model_name: str = 'C2PNet/OTS.pkl'):
    net = get_model(model_name)
    haze = Image.open(haze_image_path).convert('RGB')
    haze = tfs.ToTensor()(haze)[None, ::]
    haze = haze.to(DEVICE)
    with torch.no_grad():
        pred = net(haze)
    ts = torch.squeeze(pred.clamp(0, 1).cpu())
    torch_utils.save_image(ts, output_image_path)


def calculate(haze_image_path: str, clear_image_path: str):
    haze = Image.open(haze_image_path).convert('RGB')
    clear = Image.open(clear_image_path).convert('RGB')
    haze = tfs.ToTensor()(haze)
    clear = tfs.ToTensor()(clear)
    current_psnr = peak_signal_noise_ratio(haze, clear, data_range=1.0)
    current_ssim = structural_similarity(haze, clear, data_range=1.0, channel_axis=0)
    return current_psnr, current_ssim
