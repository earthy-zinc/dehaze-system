from io import BytesIO

import torch

from app.utils.image import postprocess_image, preprocess_image
from config import Config
from .ridcp_new_arch import FusionRefine


def get_model(model_path: str):
    net = FusionRefine()
    net.to(Config.DEVICE)
    net.load_state_dict(torch.load(model_path)['params'], strict=False)
    net.eval()
    return net

def dehaze(haze_image: BytesIO, model_path: str) -> BytesIO:
    net = get_model(model_path)
    haze = preprocess_image(haze_image)
    with torch.no_grad():
        output, _ = net.test(haze)
    return postprocess_image(output)


