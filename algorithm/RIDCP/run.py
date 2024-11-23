from io import BytesIO

import torch

from app.utils.image import preprocess_image, postprocess_image
from config import Config
from .dehaze_vq_weight_arch import VQWeightDehazeNet


def get_model(model_path: str):
    net = VQWeightDehazeNet(codebook_params=[[64, 1024, 512]],
                            LQ_stage=True, use_weight=True,
                            weight_alpha=-21.25)
    net.to(Config.DEVICE)
    net.load_state_dict(torch.load(model_path)['params'], strict=False)
    net.eval()
    return net


def dehaze(haze_image: BytesIO, model_path: str) -> BytesIO:
    net = get_model(model_path)
    haze = preprocess_image(haze_image).clip(0, 1)

    max_size = 1500 ** 2
    h, w = haze.shape[2:]
    if h * w < max_size:
        output, _ = net.test(haze)
    else:
        down_img = torch.nn.UpsamplingBilinear2d((h // 2, w // 2))(haze)
        output, _ = net.test(down_img)
        output = torch.nn.UpsamplingBilinear2d((h, w))(output)
    return postprocess_image(output)
