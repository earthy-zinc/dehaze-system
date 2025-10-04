from io import BytesIO

import torch
from torchvision import transforms

from app.utils.image import postprocess_image, preprocess_image
from config import Config
from .model.dehaze_sgid_pff import DEHAZE_SGID_PFF


def get_model(model_path: str):
    net = DEHAZE_SGID_PFF(img_channels=3, t_channels=1, n_resblock=3, n_feat=32, device=Config.DEVICE)
    net.load_state_dict(torch.load(model_path), strict=False)
    net = net.to(Config.DEVICE)
    net.eval()
    return net

def nearest_multiple_of_4(size):
    return size - (size % 4)

def dehaze(haze_image: BytesIO, model_path: str) -> BytesIO:
    net = get_model(model_path)

    haze = preprocess_image(haze_image)

    # 使用 CenterCrop 进行中心裁剪
    _, _, h, w = haze.shape
    new_h = nearest_multiple_of_4(h)
    new_w = nearest_multiple_of_4(w)
    center_crop = transforms.CenterCrop((new_h, new_w))
    haze = center_crop(haze)

    with torch.no_grad():
        _, output, _, _, _ = net(haze)

    return postprocess_image(output)
