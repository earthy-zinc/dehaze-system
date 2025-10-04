import os

import pyiqa
import torch
from PIL import Image
from torchvision.transforms import ToTensor
from tqdm import tqdm

from basicsr.archs.itb_arch import FusionRefine

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
pretrained_ridcp_path = "/mnt/e/DeepLearningCopies/2023/RIDCP/experiments/itb_o_haze/models/net_g_60000.pth"
param_key = 'params'
load_net = torch.load(pretrained_ridcp_path)[param_key]
itb = FusionRefine(opt={"LQ_stage": True})
itb.load_state_dict(load_net, strict=False)
itb.eval()

psnr = pyiqa.create_metric("psnr")
ssim = pyiqa.create_metric("ssim")

images = []
images_metric = []
count = 0
paths = ""
pbar = tqdm(total=len(paths), unit='image')
for idx, path in enumerate(paths):
    img_name = os.path.basename(path)
    pbar.set_description(f'处理图像 {img_name} 中')

    input_img = ToTensor()(Image.open(path).convert('RGB')).to(device)[None, ::]
    h, w = input_img.shape[2:]

    input_img = torch.nn.UpsamplingBilinear2d((h//3, w//3))(input_img)
    output, _ = itb.test(input_img)
    output = torch.nn.UpsamplingBilinear2d((h, w))(output)

    psnr_hl = psnr(input_img, output).item()
    ssim_hl = ssim(input_img, output).item()

    clear_out = input_img.squeeze().permute(1, 2, 0)
    hazy_out = output.squeeze().permute(1, 2, 0)
    images_metric.append({
        "Name": img_name,
        "PSNR": psnr_hl,
        "SSIM": ssim_hl,
    })
    images.append(clear_out)
    images.append(hazy_out)
    count += 1
    pbar.update(1)
pbar.close()

