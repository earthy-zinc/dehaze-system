import cv2
import numpy as np
import torch
import torchvision
from torch import nn

from basicsr import img2tensor, GANLoss
from basicsr.archs.ridcp.discriminator import UNetDiscriminatorSN
from basicsr.archs.ridcp_new_arch import RIDCPNew

discriminator = UNetDiscriminatorSN(512)
discriminator.cuda()
discriminator.load_state_dict(torch.load("E://DeepLearningCopies//2023//RIDCP//experiments//ridcp_first_train//models//net_d_best_.pth")["params"])
discriminator.eval()

hqp = RIDCPNew(LQ_stage=False)
hqp.cuda()
hqp.load_state_dict(torch.load("E://DeepLearningCopies//2023//RIDCP//pretrained_models//pretrained_HQPs_renamed.pth")["params"])
hqp.eval()

ridcp = RIDCPNew(LQ_stage=True)
ridcp.cuda()
ridcp.load_state_dict(torch.load("E://DeepLearningCopies//2023//RIDCP//pretrained_models//pretrained_RIDCP_renamed.pth")["params"])
ridcp.eval()

img = cv2.imread("E://DeepLearningCopies//2023/RIDCP//datasets//rgb_500//0001.jpg", cv2.IMREAD_UNCHANGED)
if img.max() > 255.0:
    img = img / 255.0
if img.shape[-1] > 3:
    img = img[:, :, :3]
img_tensor = img2tensor(img).cuda() / 255.
inputs = img_tensor.unsqueeze(0)


out_img, out_img_residual, codebook_loss, feat_to_quant, z_quant, indices = ridcp(inputs)
_, _, _, hqp_feat, _, _ = hqp(inputs)

out = discriminator(feat_to_quant)
hqp_out = discriminator(hqp_feat)

print(out.mean().item())
print(hqp_out.mean().item())

gan_loss = GANLoss('hinge')

print(gan_loss(out, False, True))
print(gan_loss(hqp_out, True, True))


