import os

import cv2
import torch
import torchvision.utils

from benchmark.RIDCP.dehaze_vq_weight_arch import VQWeightDehazeNet
from global_variable import MODEL_PATH, DEVICE


def get_model(model_name: str):
    # 构造模型文件的绝对路径
    model_dir = os.path.join(MODEL_PATH, model_name)

    net = VQWeightDehazeNet(codebook_params=[[64, 1024, 512]],
                            LQ_stage=True, use_weight=True,
                            weight_alpha=-21.25)
    net.to(DEVICE)
    net.load_state_dict(torch.load(model_dir)['params'], strict=False)
    net.eval()
    return net


def img2tensor(imgs, bgr2rgb=True, float32=True):
    """Numpy array to tensor.

    Args:
        imgs (list[ndarray] | ndarray): Input images.
        bgr2rgb (bool): Whether to change bgr to rgb.
        float32 (bool): Whether to change to float32.

    Returns:
        list[tensor] | tensor: Tensor images. If returned results only have
            one element, just return tensor.
    """

    def _totensor(img, bgr2rgb, float32):
        if img.shape[2] == 3 and bgr2rgb:
            if img.dtype == 'float64':
                img = img.astype('float32')
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img.transpose(2, 0, 1))
        if float32:
            img = img.float()
        return img

    if isinstance(imgs, list):
        return [_totensor(img, bgr2rgb, float32) for img in imgs]
    else:
        return _totensor(imgs, bgr2rgb, float32)


def dehaze(haze_image_path: str, output_image_path: str, model_name: str = ''):
    net = get_model(model_name)
    img = cv2.imread(haze_image_path, cv2.IMREAD_UNCHANGED)
    if img.max() > 255.0:
        img = img / 255.0
    if img.shape[-1] > 3:
        img = img[:, :, :3]
    img_tensor = img2tensor(img).to(DEVICE) / 255.
    img_tensor = img_tensor.unsqueeze(0)

    max_size = 1500 ** 2
    h, w = img_tensor.shape[2:]
    if h * w < max_size:
        output, _ = net.test(img_tensor)
    else:
        down_img = torch.nn.UpsamplingBilinear2d((h // 2, w // 2))(img_tensor)
        output, _ = net.test(down_img)
        output = torch.nn.UpsamplingBilinear2d((h, w))(output)
    output = torch.squeeze(output.clamp(0, 1).cpu())
    torchvision.utils.save_image(output, output_image_path)
