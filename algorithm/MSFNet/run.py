import os

import torch
import torchvision.transforms as tfs
from PIL import Image
from torch.autograd import Variable

from benchmark.MSFNet.net import final_Net
from global_variable import MODEL_PATH, DEVICE


def get_model(model_name: str):
    # 构造模型文件的绝对路径
    model_dir = os.path.join(MODEL_PATH, model_name)
    net = final_Net()
    net.to(DEVICE)
    # Load pretrained models
    net.load_state_dict(torch.load(model_dir)["state_dict"])
    net.eval()
    return net


def dehaze(haze_image_path: str, output_image_path: str, model_name: str = ''):
    net = get_model(model_name)
    haze = Image.open(haze_image_path).convert('RGB')
    transform = tfs.Compose([
        tfs.ToTensor(),
        tfs.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    haze = transform(haze).unsqueeze_(0)
    haze = Variable(haze.to(DEVICE))
    with torch.no_grad():
        _, _, output = net(haze)
    output = torch.clamp(output, 0., 1.)
    prediction = output.data.cpu().numpy().squeeze().transpose((1, 2, 0))
    prediction = (prediction*255.0).astype("uint8")
    im = Image.fromarray(prediction)
    im.save(output_image_path)
