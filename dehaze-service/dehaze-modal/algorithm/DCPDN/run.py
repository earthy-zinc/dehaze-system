from io import BytesIO

import torch

from algorithm.image_utils import preprocess_image, postprocess_image
from algorithm.config import Config
from .dehaze22 import dehaze as DCPDN


def get_model(model_path: str):
    net = DCPDN(3, 3, 64)
    net.to(Config.DEVICE)
    net.load_state_dict(torch.load(model_path, weights_only=False))
    net.eval()
    return net


# TODO 该模型torch版本太低，无法正确加载预训练模型
def dehaze(haze_image: BytesIO, model_path: str) -> BytesIO:
    net = get_model(model_path)
    haze = preprocess_image(haze_image)
    with torch.no_grad():
        out, _, _, _ = net(haze)
    return postprocess_image(out)
