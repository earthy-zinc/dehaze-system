import cv2
import numpy as np
import torch
import torchvision.utils
from PIL import Image

from .ridcp_new_arch import RIDCPNew
from global_variable import MODEL_PATH, DEVICE
from .calculate import imgOperation

def get_model(model_path: str):
    net = RIDCPNew()
    net.to(DEVICE)
    net.load_state_dict(torch.load(model_path)['params'], strict=False)
    net.eval()
    return net

def dehaze(haze_image_path: str, output_image_path: str, model_path: str):
    net = get_model(model_path)
    haze = imgOperation(haze_image_path)
    _, output, _, _, _, _ = net(haze)
    output = torch.squeeze(output.clamp(0, 1).cpu())
    torchvision.utils.save_image(output, output_image_path)


