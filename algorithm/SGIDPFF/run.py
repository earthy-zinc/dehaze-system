import os

import imageio.v2 as imageio
import numpy as np
import torch

from benchmark.SGIDPFF.model.dehaze_sgid_pff import DEHAZE_SGID_PFF
from global_variable import MODEL_PATH, DEVICE


def get_model(model_name: str):
    # 构造模型文件的绝对路径
    model_dir = os.path.join(MODEL_PATH, model_name)
    net = DEHAZE_SGID_PFF(img_channels=3, t_channels=1, n_resblock=3, n_feat=32, device=DEVICE)
    net.load_state_dict(torch.load(model_dir), strict=False)
    net = net.to(DEVICE)
    net.eval()
    return net


def numpy2tensor(input, rgb_range=1.):
    img = np.array(input).astype('float64')
    np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))  # HWC -> CHW
    tensor = torch.from_numpy(np_transpose).float()  # numpy -> tensor
    tensor.mul_(rgb_range / 255)  # (0,255) -> (0,1)
    tensor = tensor.unsqueeze(0)
    return tensor


def tensor2numpy(tensor, rgb_range=1.):
    rgb_coefficient = 255 / rgb_range
    img = tensor.mul(rgb_coefficient).clamp(0, 255).round()
    img = img[0].data
    img = np.transpose(img.cpu().numpy(), (1, 2, 0)).astype(np.uint8)
    return img


def dehaze(haze_image_path: str, output_image_path: str, model_name: str = 'DehazeFormer/indoor/dehazeformer-b.pth'):
    net = get_model(model_name)

    inputs = imageio.imread(haze_image_path)
    h, w, _ = inputs.shape
    new_h, new_w = h - h % 4, w - w % 4
    inputs = inputs[:new_h, :new_w, :]
    in_tensor = numpy2tensor(inputs).to(DEVICE)

    with torch.no_grad():
        _, output, _, _, _ = net(in_tensor)

    output_img = tensor2numpy(output)
    imageio.imwrite(output_image_path, output_img)
