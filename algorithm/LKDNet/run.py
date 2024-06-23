import os
from collections import OrderedDict

import cv2
import numpy as np
import torch
import torchvision.transforms as tfs
from PIL import Image

from global_variable import MODEL_PATH, DEVICE


def single(save_dir):
    state_dict = torch.load(save_dir, map_location=torch.device(DEVICE))
    # print(state_dict)
    new_state_dict = OrderedDict()

    for k, v in state_dict.items():
        name = k[7:]
        new_state_dict[name] = v

    return new_state_dict


def get_model(model_path: str):
    net = eval(os.path.basename(model_path).replace('-', '_').replace('.pth', ''))()
    net = net.to(DEVICE)
    net.load_state_dict(single(model_path), strict=False)
    torch.cuda.empty_cache()
    net.eval()
    return net


def dehaze(haze_image_path: str, output_image_path: str, model_path: str):
    net = get_model(model_path)
    haze = Image.open(haze_image_path).convert('RGB')
    haze = tfs.ToTensor()(haze)[None, ::]
    haze = haze.to(DEVICE)
    with torch.no_grad():
        output = net(haze).clamp_(-1, 1)
        # [-1, 1] to [0, 1]
        output = output * 0.5 + 0.5
    out_img = np.transpose(output.detach().cpu().squeeze(0).numpy(), axes=[1, 2, 0]).copy()
    out_img = np.round((out_img[:, :, ::-1].copy() * 255.0)).astype('uint8')
    cv2.imwrite(output_image_path, out_img)
