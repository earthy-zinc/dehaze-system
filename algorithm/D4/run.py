import os
import random

import cv2
import numpy as np
import torch
import torchvision
import torchvision.transforms.functional
from PIL import Image

from benchmark.D4.config import Config
from benchmark.D4.models import Model
from global_variable import MODEL_PATH, DEVICE


def get_model(model_name: str):
    # 构造模型文件的绝对路径
    model_dir = os.path.join(MODEL_PATH, model_name)

    # load config file
    # mode (int): 1: train, 2: test, 3: eval, reads from config file if not specified
    # model (int): 1: reconstruct
    config_path = os.path.join(model_dir, 'config.yml')
    config = Config(config_path)
    config.MODE = 3
    config.MODEL = 1
    net = Model(config).to(DEVICE)

    # set cv2 running threads to 1 (prevents deadlocks with pytorch dataloader)
    cv2.setNumThreads(0)

    # initialize random seed
    torch.manual_seed(config.SEED)
    torch.cuda.manual_seed_all(config.SEED)
    np.random.seed(config.SEED)
    random.seed(config.SEED)

    net.load()
    net.eval()
    return net


def pad_input(input: torch.Tensor, times=32):
    input_h, input_w = input.shape[2:]
    pad_h = pad_w = 0

    if input_h % times != 0:
        pad_h = times - (input_h % times)

    if input_w % times != 0:
        pad_w = times - (input_w % times)

    return torch.nn.functional.pad(input, (0, pad_w, 0, pad_h), mode='reflect')


def get_square_img(img):
    h, w = img.size
    if h < w:
        return torchvision.transforms.functional.crop(img, random.randint(0,w-h), 0,  h, h)
    elif h >= w:
        return torchvision.transforms.functional.crop(img, 0, random.randint(0,h-w), w, w)


def stitch_images(inputs, *outputs, img_per_row=2):
    gap = 5
    columns = len(outputs) + 1

    height, width = inputs[0][:, :, 0].shape
    img = Image.new('RGB', (width * img_per_row * columns + gap * (img_per_row - 1), height * int(len(inputs) / img_per_row +0.5)))
    images = [inputs, *outputs]

    for ix in range(len(inputs)):
        xoffset = int(ix % img_per_row) * width * columns + int(ix % img_per_row) * gap
        yoffset = int(ix / img_per_row) * height

        for cat in range(len(images)):
            im = np.array((images[cat][ix]).cpu()).astype(np.uint8).squeeze()
            im = Image.fromarray(im)
            img.paste(im, (xoffset + cat * width, yoffset))

    return img


def crop_result(result, input_h, input_w, times=32):
    crop_h = crop_w = 0
    if input_h % times != 0:
        crop_h = times - (input_h % times)

    if input_w % times != 0:
        crop_w = times - (input_w % times)

    if crop_h != 0:
        print(crop_h)
        result = result[...,:-crop_h, :]

    if crop_w != 0:
        result = result[...,:-crop_w]
    return result


def postprocess(img, size=None):
    # [0, 1] => [0, 255]
    if size is not None:
        img = torch.nn.functional.interpolate(img, size, mode='bicubic')
    img = img * 255.0
    img = img.permute(0, 2, 3, 1)
    return img.int()


def dehaze(haze_image_path: str, output_image_path: str, model_name: str = 'C2PNet/OTS.pkl'):
    net = get_model(model_name)

    haze = Image.open(haze_image_path).convert('RGB')
    haze = get_square_img(haze)
    haze = torchvision.transforms.functional.resize(haze, size=[256], interpolation=Image.BICUBIC)
    transforms = torchvision.transforms.Compose(
        ([torchvision.transforms.RandomCrop((256, 256))]) +
        [torchvision.transforms.ToTensor()]
    )
    haze = transforms(haze)[None, ::]
    haze = haze.to(DEVICE)

    h, w = haze.shape[2:4]
    noisy_images_input = pad_input(haze)
    clean_images_h2c, _ = net.forward_h2c(noisy_images_input)
    predicted_results = crop_result(clean_images_h2c, h, w)
    predicted_results = postprocess(predicted_results)[0]
    im = Image.fromarray(predicted_results.cpu().numpy().astype(np.uint8).squeeze())
    im.save(output_image_path)
