import os

import torch
import torchvision.transforms
from PIL import Image

from benchmark.DehazeNet.model import DehazeNet
from global_variable import MODEL_PATH, DEVICE


def get_model(model_name: str):
    # 构造模型文件的绝对路径
    model_dir = os.path.join(MODEL_PATH, model_name)
    net = DehazeNet()
    net.to(DEVICE)
    net.load_state_dict(torch.load(model_dir))
    net.eval()
    return net


def dehaze(haze_image_path: str, output_image_path: str, model_name: str = ''):
    net = get_model(model_name)
    haze = Image.open(haze_image_path).convert('RGB')
    loader = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img1 = loader(haze).cuda()
    img2 = torchvision.transforms.ToTensor()(haze).cuda()
    c, h, w = img1.shape
    patch_size = 16
    num_w = int(w / patch_size)
    num_h = int(h / patch_size)
    t_list = []
    for i in range(0, num_w):
        for j in range(0, num_h):
            patch = img1[:, 0 + j * patch_size:patch_size + j * patch_size,
                    0 + i * patch_size:patch_size + i * patch_size]
            patch = torch.unsqueeze(patch, dim=0).cuda()
            t = net(patch)
            t_list.append([i, j, t])

    t_list = sorted(t_list, key=lambda t_list: t_list[2])
    a_list = t_list[:len(t_list) // 100]
    a0 = 0
    for k in range(0, len(a_list)):
        patch = img2[:, 0 + a_list[k][1] * patch_size:patch_size + a_list[k][1] * patch_size,
                0 + a_list[k][0] * patch_size:patch_size + a_list[k][0] * patch_size]
        a = torch.max(patch)
        if a0 < a.item():
            a0 = a.item()
    for k in range(0, len(t_list)):
        img2[:, 0 + t_list[k][1] * patch_size:patch_size + t_list[k][1] * patch_size,
        0 + t_list[k][0] * patch_size:patch_size + t_list[k][0] * patch_size] = (img2[:,
             0 + t_list[k][
                 1] * patch_size:patch_size +
                                 t_list[k][
                                     1] * patch_size,
             0 + t_list[k][
                 0] * patch_size:patch_size +
                                 t_list[k][
                                     0] * patch_size] - a0 * (
                     1 - t_list[k][2])) / \
            t_list[k][2]
    defog_img = torchvision.transforms.ToPILImage()(img2)
    defog_img.save(output_image_path)
