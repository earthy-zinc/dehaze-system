from io import BytesIO

import torch

from app.utils.image import preprocess_image, postprocess_image
from config import Config
from .models.common import default_conv
from .models.ipt import define_G
from .test_options import TestOptions


def get_model(model_path: str):
    opt = TestOptions().parse()
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip
    opt.display_id = -1  # no visdom display
    net = define_G(opt, default_conv)
    ckp = torch.load(model_path)
    net = net.to(Config.DEVICE)
    net.load_state_dict(ckp)
    net.eval()
    return net


def dehaze(haze_image: BytesIO, model_path: str) -> BytesIO:
    net = get_model(model_path)
    haze = preprocess_image(haze_image)
    with torch.no_grad():
        pred = net(haze)
    return postprocess_image(pred)

