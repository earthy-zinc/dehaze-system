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


with open("./options/RIDCP.yml", mode='r') as f:
    opt = yaml.load(f, Loader=ordered_yaml()[0])

hq_opt = opt['network_g'].copy()
hq_opt['LQ_stage'] = False

net_hq = build_network(hq_opt).to(DEVICE)
net_hq.eval()
with torch.no_grad():
    for name, module in net_hq.named_modules():
        # print(name)
        if name == 'quantize_group.0.embedding':
            print(module.weight)
            print(module.weight.shape)
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
