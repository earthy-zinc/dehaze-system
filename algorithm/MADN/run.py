import os

import cv2
import numpy as np
import torch
import torchvision.transforms as tfs
import torchvision.utils as torch_utils
from PIL import Image
from torch.autograd import Variable
from torchvision.transforms import transforms

from model200314 import TransformNet
from modelmeta import Meta
from pregrocess import pre
from global_variable import MODEL_PATH, DEVICE


def get_model(model_path: str):
    meta_path = os.path.join(os.path.dirname(model_path), "dehaze_80_state_dict.pth")

    meta = Meta(8)
    meta.to(DEVICE)
    meta.load_state_dict(torch.load(meta_path, map_location=DEVICE))
    meta.eval()

    net = TransformNet(32)
    net.to(DEVICE)
    net.load_state_dict(torch.load(model_path))
    net.eval()
    return net, meta


def dehaze(haze_image_path: str, output_image_path: str, model_path: str):
    net, meta = get_model(model_path)
    prepro = pre(0.5, 0)

    haze = Image.open(haze_image_path)
    haze = tfs.ToTensor()(haze)[None, ::]
    haze = haze.to(DEVICE)

    lr_patch = Variable(haze, requires_grad=False)

    image = haze.cpu().clone().squeeze(0)
    haze = transforms.ToPILImage()(image)
    x = cv2.cvtColor(np.asarray(haze), cv2.COLOR_RGB2BGR)

    with torch.no_grad():
        pre_haze = prepro(x).to(DEVICE)
        fea1, fea2, fea3, fea4, fea5, _ = meta(pre_haze)
        haze_features = torch.cat((fea1, fea2, fea3, fea4, fea5), 1)
        output = net(lr_patch, haze_features, pre_haze)

    torch_utils.save_image(output, output_image_path)
