from io import BytesIO

import torch

from algorithm.image_utils import postprocess_image, preprocess_image
from algorithm.config import Config
from .ridcp_new_arch import FusionRefine


def get_model(model_path: str):
    net = FusionRefine()
    net.to(Config.DEVICE)
    net.load_state_dict(torch.load(model_path, weights_only=False)['params'], strict=False)
    net.eval()
    return net

def dehaze(haze_image: BytesIO, model_path: str) -> BytesIO:
    net = get_model(model_path)
    haze = preprocess_image(haze_image)
    with torch.no_grad():
        output, _ = net.test(haze)
    return postprocess_image(output)


