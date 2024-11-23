from io import BytesIO

import torch

from app.utils.image import postprocess_image
from config import Config
from .calculate import imgOperation
from .ridcp_new_arch import RIDCPNew


def get_model(model_path: str):
    net = RIDCPNew()
    net.to(Config.DEVICE)
    net.load_state_dict(torch.load(model_path)['params'], strict=False)
    net.eval()
    return net

def dehaze(haze_image: BytesIO, model_path: str) -> BytesIO:
    net = get_model(model_path)
    haze = imgOperation(haze_image)
    with torch.no_grad():
        _, output, _, _, _, _ = net(haze)
    return postprocess_image(output)


