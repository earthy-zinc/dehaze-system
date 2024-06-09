import os

import numpy as np
import torch
from PIL import Image
from torch.autograd import Variable

from benchmark.GCANet.GCANet import GCANet
from global_variable import MODEL_PATH, DEVICE


def get_model(model_name: str):
    # 构造模型文件的绝对路径
    model_dir = os.path.join(MODEL_PATH, model_name)
    net = GCANet(in_c=4, out_c=3, only_residual=True)
    net = net.to(DEVICE)
    net.load_state_dict(torch.load(model_dir))
    net.eval()
    return net


def edge_compute(x):
    x_diffx = torch.abs(x[:,:,1:] - x[:,:,:-1])
    x_diffy = torch.abs(x[:,1:,:] - x[:,:-1,:])

    y = x.new(x.size())
    y.fill_(0)
    y[:,:,1:] += x_diffx
    y[:,:,:-1] += x_diffx
    y[:,1:,:] += x_diffy
    y[:,:-1,:] += x_diffy
    y = torch.sum(y,0,keepdim=True)/3
    y /= 4
    return y


def dehaze(haze_image_path: str, output_image_path: str, model_name: str = ''):
    net = get_model(model_name)

    with torch.no_grad():
        img = Image.open(haze_image_path).convert('RGB')
        im_w, im_h = img.size
        if im_w % 4 != 0 or im_h % 4 != 0:
            img = img.resize((int(im_w // 4 * 4), int(im_h // 4 * 4)))
        img = np.array(img).astype('float')
        img_data = torch.from_numpy(img.transpose((2, 0, 1))).float().to(DEVICE)
        edge_data = edge_compute(img_data).to(DEVICE)
        in_data = torch.cat((img_data, edge_data), dim=0).unsqueeze(0) - 128
        in_data.to(DEVICE)
        pred = net(Variable(in_data))
        out_img_data = (pred.data[0].cpu().float() + img_data.cpu()).round().clamp(0, 255)
        out_img = Image.fromarray(out_img_data.numpy().astype(np.uint8).transpose(1, 2, 0))
        out_img.save(output_image_path)

