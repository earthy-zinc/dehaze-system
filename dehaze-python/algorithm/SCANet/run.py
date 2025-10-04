from io import BytesIO

import numpy as np
import torch
from PIL import Image
from torch.autograd import Variable

from app.utils.image import postprocess_image
from config import Config
from .models import Generator


def get_model(model_path: str):
    net = Generator()
    net.to(Config.DEVICE)

    net = torch.nn.DataParallel(net, device_ids=Config.DEVICE_ID).cuda()

    model_info = torch.load(model_path)

    net.load_state_dict(model_info['state_dict'])
    net.eval()
    return net


def dehaze(haze_image: BytesIO, model_path: str) -> BytesIO:
    net = get_model(model_path)

    norm = lambda x: (x - 0.5) / 0.5
    denorm = lambda x: (x + 1) / 2

    haze = np.array(Image.open(haze_image).convert('RGB')) / 255
    with torch.no_grad():
        haze = torch.Tensor(haze.transpose(2, 0, 1)[np.newaxis, :, :, :]).cuda()
        haze = Variable(haze, requires_grad=True)
        haze = norm(haze)
        out, _ = net(haze)
        out = denorm(out)
    return postprocess_image(out)
