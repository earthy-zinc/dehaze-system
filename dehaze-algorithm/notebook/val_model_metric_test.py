from basicsr.archs.dehaze_vq_weight_arch import VQWeightDehazeNet
from basicsr.archs.itb_arch import FusionRefine
import torch
import os
import pyiqa
import glob
from tqdm import tqdm
import torchvision
from torchvision.transforms import ToTensor
from matplotlib import pyplot as plt
from PIL import Image
import math

# 指定模型运算设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 指定预训练模型存放位置
pretrained_net_path = "/mnt/e/DeepLearningCopies/2023/RIDCP/pretrained_models/ridcp_o_haze.pth"
# 指定待评估图片路径
haze_img_path = "/mnt/d/DeepLearning/dataset/test/hazy"
clean_img_path = "/mnt/d/DeepLearning/dataset/test/clean"
# 指定输出图像保存路径
output_img_path = "/mnt/e/DeepLearningCopies/2023/RIDCP/ohaze_results/NH-HAZE-23"
compare_img_path = "/mnt/e/DeepLearningCopies/2023/RIDCP/ohaze_results/NH-HAZE-23/compare_results"
# 指定图像最大分辨率
# 分辨率过大容易爆显存，超过最大分辨率的将会降采样后交由模型处理
max_size = 1024 * 1024

# 构建模型，加载预训练权重
# opt = {
#     "LQ_stage": True,
#     "use_weight": True,
#     "weight_alpha": -21.25,
#     "additional_encoder": "DiNAT"
# }
# sr_model = FusionRefine(opt=opt).to(device)
sr_model = VQWeightDehazeNet(LQ_stage=True, use_weight=True, weight_alpha=-21.25, codebook_params=[[64, 1024, 512]]).to(device)
sr_model.load_state_dict(torch.load(pretrained_net_path)['params'], strict=False)
sr_model.eval()

psnr = pyiqa.create_metric("psnr", device=device)
ssim = pyiqa.create_metric("ssim", device=device)

if os.path.isfile(haze_img_path):
    paths = [haze_img_path]
else:
    paths = sorted(glob.glob(os.path.join(haze_img_path, '*')))

total_psnr = 0
total_ssim = 0
count = 0
pbar = tqdm(total=len(paths), unit='image')
for idx, path in enumerate(paths):
    images = []
    images_metric = []
    img_name = os.path.basename(path)
    save_path = os.path.join(output_img_path, f'{img_name}')
    pbar.set_description(f'处理图像 {img_name} 中')

    input_img = ToTensor()(Image.open(path).convert('RGB')).to(device)[None, ::]
    clean_img = ToTensor()(Image.open(os.path.join(clean_img_path, img_name.replace('hazy', 'GT')))
                           .convert('RGB')).to(device)[None, ::]
    h, w = input_img.shape[2:]
    if h * w < max_size:
        output, _ = sr_model.test(input_img)
    elif h * w > max_size:
        num = math.floor(math.sqrt(h * w) / 1000)
        input_img = torch.nn.UpsamplingBilinear2d((h//num, w//num))(input_img)
        # clean_img = torch.nn.UpsamplingBilinear2d((h//4, w//4))(clean_img)
        output, _ = sr_model.test(input_img)
        output = torch.nn.UpsamplingBilinear2d((h, w))(output)
    else:
        input_img = torch.nn.UpsamplingBilinear2d((h//2, w//2))(input_img)
        # clean_img = torch.nn.UpsamplingBilinear2d((h//2, w//2))(clean_img)
        output, _ = sr_model.test(input_img)
        output = torch.nn.UpsamplingBilinear2d((h, w))(output)

    torchvision.utils.save_image(output, save_path)
    # .replace('hazy', 'GT')

    psnr_hl = psnr(output, clean_img).item()
    ssim_hl = ssim(output, clean_img).item()

    clear_out = clean_img.squeeze().permute(1, 2, 0)
    hazy_img = input_img.squeeze().permute(1, 2, 0)
    hazy_out = output.squeeze().permute(1, 2, 0)

    total_psnr += psnr_hl
    total_ssim += ssim_hl
    count += 1
    images.append(hazy_img.cpu().detach().numpy())
    images.append(hazy_out.cpu().detach().numpy())
    images.append(clear_out.cpu().detach().numpy())
    fig, axs = plt.subplots(1, 3)
    for j, ax in enumerate(axs.flat):
        ax.imshow(images[j])
        ax.axis('off')
    plt.suptitle("Name: {} SSIM: {:.2f} PSNR: {:.2f}".format(
        img_name, ssim_hl, psnr_hl
    ))
    plt.savefig(os.path.join(compare_img_path, img_name))
    plt.close()
    pbar.update(1)

pbar.close()
print('PSNR: {:.2f}, SSIM: {:.4f}'.format(total_psnr / count, total_ssim / count))



