from io import BytesIO

import torch
import torch.nn.functional as F

from algorithm.config import Config
from algorithm.image_utils import postprocess_image, preprocess_image
from .MB_TaylorFormer import MB_TaylorFormer

# 官方 MB-TaylorFormer-B.yml / -L.yml 的 network_g 配置（差异项）
_MODEL_SIZES = {
    "B": {"num_blocks": [2, 3, 3, 4], "num_refinement_blocks": 2, "num_path": [2, 2, 2, 2]},
    "L": {"num_blocks": [4, 6, 6, 8], "num_refinement_blocks": 4, "num_path": [2, 3, 3, 3]},
}
_COMMON_ARGS = {
    "inp_channels": 3,
    "out_channels": 3,
    "dim": [24, 48, 72, 96],
    "heads": [1, 2, 4, 8],
    "ffn_expansion_factor": 2.66,
    "bias": False,
    "LayerNorm_type": "WithBias",
    "dual_pixel_task": False,
    "qk_norm": 0.5,
    "offset_clamp": (-3, 3),
}


def get_model(model_path: str):
    # 权重文件名含 "-MB-TaylorFormer-L." 为 L 档，其余为 B 档
    size = "L" if "-MB-TaylorFormer-L." in model_path else "B"
    net = MB_TaylorFormer(**_COMMON_ARGS, **_MODEL_SIZES[size])
    net = net.to(Config.DEVICE)
    state_dict = torch.load(model_path, weights_only=False, map_location=Config.DEVICE)
    net.load_state_dict(state_dict["params"])
    net.eval()
    return net


def dehaze(haze_image: BytesIO, model_path: str) -> BytesIO:
    net = get_model(model_path)
    haze = preprocess_image(haze_image)  # (1,3,H,W)，像素 [0,1]

    # 官方推理约束：输入需为 8 的倍数，reflect padding 后裁回
    factor = 8
    _, _, h, w = haze.shape
    pad_h = (factor - h % factor) % factor
    pad_w = (factor - w % factor) % factor
    haze = F.pad(haze, (0, pad_w, 0, pad_h), "reflect")

    with torch.no_grad():
        out = net(haze)

    out = out[:, :, :h, :w]
    return postprocess_image(out)
