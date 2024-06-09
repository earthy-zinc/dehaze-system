import os
from collections import OrderedDict

import torch
from PIL import Image
import torchvision.transforms as tfs
from global_variable import MODEL_PATH, DEVICE
import torchvision.utils as torch_utils
from .model import MixDehazeNet_t, MixDehazeNet_s, MixDehazeNet_b, MixDehazeNet_l


def get_model(model_name: str):
    # 构造模型文件的绝对路径
    model_dir = os.path.join(MODEL_PATH, model_name)
    net = eval(model_name.split("/")[2].replace('-', '_').replace('.pth', ''))()
    net = net.to(DEVICE)
    state_dict = torch.load(model_dir)['state_dict']
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]
        new_state_dict[name] = v

    net.load_state_dict(new_state_dict)
    net.eval()
    return net


def dehaze(haze_image_path: str, output_image_path: str, model_name: str = 'DehazeFormer/indoor/dehazeformer-b.pth'):
    net = get_model(model_name)
    haze = Image.open(haze_image_path).convert('RGB')
    haze = tfs.ToTensor()(haze)[None, ::]
    haze = haze.to(DEVICE)
    with torch.no_grad():
        pred = net(haze)
    ts = torch.squeeze(pred.clamp(0, 1).cpu())
    torch_utils.save_image(ts, output_image_path)
