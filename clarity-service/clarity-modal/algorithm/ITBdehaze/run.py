from io import BytesIO

import torch
import torchvision.transforms.functional as F
from PIL import Image

from algorithm.image_utils import preprocess_image, postprocess_image
from algorithm.config import Config
from .config import get_config
from .model import fusion_refine
from .models import build_model


def get_model(model_path: str):
    config = get_config()
    swv2_model = build_model(config)
    net = fusion_refine(swv2_model, '')
    net = net.to(Config.DEVICE)
    ckpt = torch.load(model_path, weights_only=False)
    state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    net.load_state_dict(state_dict)
    net.eval()
    return net


def dehaze(haze_image: BytesIO, model_path: str) -> BytesIO:
    net = get_model(model_path)
    # Swin V2 训练尺寸为 256x256，先缩放推理再还原，避免小图窗口划分为 0
    img = Image.open(haze_image).convert('RGB')
    orig_size = img.size
    resized = img.resize((256, 256), Image.BICUBIC)
    buf = BytesIO()
    resized.save(buf, format="JPEG")
    buf.seek(0)
    haze = preprocess_image(buf)
    with torch.no_grad():
        pred = net(haze)
    out = F.resize(pred.squeeze(0), [orig_size[1], orig_size[0]], interpolation=Image.BICUBIC)
    return postprocess_image(out.unsqueeze(0))
