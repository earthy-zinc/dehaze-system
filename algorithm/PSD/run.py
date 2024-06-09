import os

import numpy as np
import torch
import torchvision.utils as torch_utils
from PIL import Image
from torch import nn
from torchvision.transforms import Compose, ToTensor, Normalize

from benchmark.PSD.FFA import FFANet
from benchmark.PSD.GCA import GCANet
from benchmark.PSD.MSBDN import MSBDNNet
from global_variable import MODEL_PATH, DEVICE, DEVICE_ID


def get_model(model_name: str):
    if model_name.find("MSBDN") != -1:
        print(model_name, '加载模型MSBDN')
        net = MSBDNNet()
        name = 'MSBDN'
    elif model_name.find("FFANET") != -1:
        print(model_name, '加载FFANET')
        net = FFANet(3, 19)
        name = 'FFANET'
    else:
        print(model_name, '加载GCANET')
        net = GCANet(in_c=4, out_c=3, only_residual=True)
        name = 'GCANET'
    # 构造模型文件的绝对路径
    model_dir = os.path.join(MODEL_PATH, model_name)
    net = net.to(DEVICE)
    net = nn.DataParallel(net, device_ids=DEVICE_ID)
    net.load_state_dict(torch.load(model_dir))
    net.eval()
    return net, name

# TODO 不知道为何，在神经网络中间层有张量形状不一致的问题，等待进一步处理
def dehaze(haze_image_path: str, output_image_path: str, model_name: str = 'C2PNet/OTS.pkl'):
    net, name = get_model(model_name)

    haze_img = Image.open(haze_image_path).convert('RGB')
    haze_reshaped = haze_img.resize((256, 256), Image.LANCZOS)
    transform_haze = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    haze = transform_haze(haze_img)[None, ::]
    haze_reshaped = transform_haze(haze_reshaped)[None, ::]
    haze.to(DEVICE)
    haze_reshaped.to(DEVICE)
    print(haze.size())
    print(haze_reshaped.size())
    with torch.no_grad():
        if name == 'GCANET':
            print('GCANET测试中')
            pred = net(haze, 0, True, False)
            dehaze = pred.float().round().clamp(0, 255)
            out_img = Image.fromarray(dehaze[0].cpu().numpy().astype(np.uint8).transpose(1, 2, 0))
            out_img.save(output_image_path)

        else:
            _, pred, _, _, _ = net(haze, haze_reshaped, True)
            ts = torch.squeeze(pred.clamp(0, 1).cpu())
            torch_utils.save_image(ts, output_image_path)

