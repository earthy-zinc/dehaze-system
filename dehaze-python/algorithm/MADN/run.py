import os
from io import BytesIO

import cv2
import numpy as np
import torch
from config import Config
from torch.autograd import Variable
from torchvision.transforms import transforms

from app.utils.image import preprocess_image, postprocess_image
from .model200314 import TransformNet
from .modelmeta import Meta
from .pregrocess import pre


def get_model(model_path: str):
    meta_path = os.path.join(os.path.dirname(model_path), "dehaze_80_state_dict.pth")

    meta = Meta(8)
    meta.to(Config.DEVICE)
    meta.load_state_dict(torch.load(meta_path, map_location=Config.DEVICE))
    meta.eval()

    net = TransformNet(32)
    net.to(Config.DEVICE)
    net.load_state_dict(torch.load(model_path))
    net.eval()
    return net, meta


def dehaze(haze_image: BytesIO, model_path: str) -> BytesIO:
    net, meta = get_model(model_path)
    prepro = pre(0.5, 0)

    haze = preprocess_image(haze_image)

    lr_patch = Variable(haze, requires_grad=False)

    image = haze.cpu().clone().squeeze(0)
    haze = transforms.ToPILImage()(image)
    x = cv2.cvtColor(np.asarray(haze), cv2.COLOR_RGB2BGR)

    with torch.no_grad():
        pre_haze = prepro(x).to(Config.DEVICE)
        fea1, fea2, fea3, fea4, fea5, _ = meta(pre_haze)
        haze_features = torch.cat((fea1, fea2, fea3, fea4, fea5), 1)
        output = net(lr_patch, haze_features, pre_haze)

    return postprocess_image(output)
