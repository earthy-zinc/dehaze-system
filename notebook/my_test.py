from copy import deepcopy

import cv2
import torch
import torchvision
import yaml
from PIL import Image
import torchvision.utils as torch_utils
from basicsr import build_network
from basicsr.archs.dehaze_vq_weight_arch import VQWeightDehazeNet
from basicsr.utils.options import ordered_yaml
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

print("-------load net-----------")
load_net = torch.load("E://DeepLearningCopies//2023//RIDCP//pretrained_models//pretrained_HQPs_New.pth")
param_key = 'params'
load_net = load_net[param_key]
prefix = 'feature_extract.'
# 使用字典推导式为每个键添加前缀
load_net = {prefix + key: value for key, value in load_net.items()}
for key, param in load_net.items():
    print(key)

with open("E://DeepLearningCopies//2023//RIDCP//options//ITB-pei.yml", mode='r') as f:
    opt = yaml.load(f, Loader=ordered_yaml()[0])
net_g = build_network(opt['network_g']).to(DEVICE)
model_dict = net_g.state_dict()
state_dict = {k: v for k, v in load_net.items() if k in model_dict.keys()}
model_dict.update(state_dict)
net_g.load_state_dict(model_dict)
print("---------loaded net-----------")

# net_g.load_state_dict(load_net, strict=False)
save_dict = {}
state_dict = net_g.state_dict()
for key, param in state_dict.items():
    if key.startswith('module.'):  # remove unnecessary 'module.'
        key = key[7:]
    state_dict[key] = param.cpu()
save_dict[param_key] = state_dict
torch.save(save_dict, "E://DeepLearningCopies//2023//RIDCP//pretrained_models//ITB_init_weight.pth")

# net_hq.eval()
# with torch.no_grad():
#     for name, module in net_hq.named_modules():
#         # print(name)
#         if name == 'quantize_group.0.embedding':
#             print(module.weight)
#             print(module.weight.shape)
# net_hq.eval()
# with torch.no_grad():
#     img = cv2.imread("D://DeepLearning//dataset//RESIDE//OTS//haze//0025_0.8_0.2.jpg", cv2.IMREAD_UNCHANGED)
#     if img.max() > 255.0:
#         img = img / 255.0
#     if img.shape[-1] > 3:
#         img = img[:, :, :3]
#     img_tensor = img2tensor(img).to(DEVICE) / 255.
#     img_tensor = img_tensor.unsqueeze(0)
#
#     max_size = 1500 ** 2
#     h, w = img_tensor.shape[2:]
#     if h * w < max_size:
#         gt_rec, dec_residual, codebook_loss, semantic_loss, quant_before_feature, quant_gt, gt_indices = net_hq(img_tensor)
#         output = gt_rec
#     else:
#         down_img = torch.nn.UpsamplingBilinear2d((h // 2, w // 2))(img_tensor)
#         gt_rec, dec_residual, codebook_loss, semantic_loss, quant_before_feature, quant_gt, gt_indices = net_hq(img_tensor)
#         output = torch.nn.UpsamplingBilinear2d((h, w))(gt_rec)
#
#
# print(quant_before_feature.shape)
# print("---------")
# print(quant_gt.shape)
# print("---------")
# print(gt_indices)
# # net = VQWeightDehazeNet(codebook_params=[[64, 1024, 512]], LQ_stage=True, use_semantic_loss=True)
# # net.load_state_dict(torch.load("./pretrained_models/pretrained_HQPs.pth"))
# # print(net)
# ts = torch.squeeze(output.clamp(0, 1).cpu())
# torch_utils.save_image(ts, "./img.jpg")
