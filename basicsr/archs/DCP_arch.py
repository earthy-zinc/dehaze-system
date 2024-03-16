import os

import cv2
import math
import numpy as np
import pyiqa
import torchvision.utils
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader

from basicsr.utils import tensor2img, img2tensor, mkdir_and_rename
from basicsr.data.ntire_dataset import NtireH5Dataset
from basicsr.utils.registry import ARCH_REGISTRY


def DarkChannel(im, sz):
    b, g, r = cv2.split(im)
    dc = cv2.min(cv2.min(r, g), b);
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (sz, sz))
    return cv2.erode(dc, kernel)


def AtmLight(im, dark):
    [h, w] = im.shape[:2]
    imsz = h * w
    numpx = int(max(math.floor(imsz / 1000), 1))
    darkvec = dark.reshape(imsz);
    imvec = im.reshape(imsz, 3);

    indices = darkvec.argsort();
    indices = indices[imsz - numpx::]

    atmsum = np.zeros([1, 3])
    for ind in range(1, numpx):
        atmsum = atmsum + imvec[indices[ind]]

    A = atmsum / numpx
    return A


def TransmissionEstimate(im, A, sz):
    omega = 0.95
    im3 = np.empty(im.shape, im.dtype);

    for ind in range(0, 3):
        im3[:, :, ind] = im[:, :, ind] / A[0, ind]

    transmission = 1 - omega * DarkChannel(im3, sz)
    return transmission


def Guidedfilter(im, p, r, eps):
    mean_I = cv2.boxFilter(im, cv2.CV_64F, (r, r))
    mean_p = cv2.boxFilter(p, cv2.CV_64F, (r, r))
    mean_Ip = cv2.boxFilter(im * p, cv2.CV_64F, (r, r))
    cov_Ip = mean_Ip - mean_I * mean_p;

    mean_II = cv2.boxFilter(im * im, cv2.CV_64F, (r, r))
    var_I = mean_II - mean_I * mean_I

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = cv2.boxFilter(a, cv2.CV_64F, (r, r))
    mean_b = cv2.boxFilter(b, cv2.CV_64F, (r, r))

    q = mean_a * im + mean_b
    return q


def TransmissionRefine(im, et):
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    gray = np.float64(gray) / 255
    r = 60
    eps = 0.0001
    t = Guidedfilter(gray, et, r, eps)

    return t


def Recover(im, t, A, tx=0.1):
    res = np.empty(im.shape, im.dtype)
    t = cv2.max(t, tx)

    for ind in range(0, 3):
        res[:, :, ind] = (im[:, :, ind] - A[0, ind]) / t + A[0, ind]

    return res


@ARCH_REGISTRY.register()
class DCP(nn.Module):
    def __init__(self):
        super(DCP, self).__init__()

    def forward(self, hazy):
        hazy_cv2 = tensor2img(hazy)
        I = hazy_cv2.astype('float64') / 255
        dark = DarkChannel(I, 15)
        A = AtmLight(I, dark)
        te = TransmissionEstimate(I, A, 15)
        t = TransmissionRefine(hazy_cv2, te)
        J = Recover(I, t, A, 0.1)
        recover = img2tensor(J)[None, ::]
        return recover


if __name__ == '__main__':
    dataset = "DENSE-HAZE"
    dataset_path = "/var/lib/docker/user1/wpx/DeepLearningCopies/2023/RIDCP/datasets/{}-test.h5".format(dataset)
    img_save_path = "/mnt/e/DeepLearningCopies/2023/RIDCP/experiments/DCP/visualization/{}/".format(dataset)
    mkdir_and_rename(img_save_path)

    c
    lpips = pyiqa.create_metric("lpips")
    niqe = pyiqa.create_metric("niqe")
    nima = pyiqa.create_metric("nima")
    brisque = pyiqa.create_metric("brisque")

    dataloader = DataLoader(dataset=NtireH5Dataset(opt={"data_path": dataset_path, "shuffle": False}),
                            num_workers=4, pin_memory=True, batch_size=1, shuffle=False)
    iteration = 0
    psnr_value = 0
    ssim_value = 0
    lpips_value = 0
    niqe_value = 0
    brisque_value = 0
    nima_value = 0
    for data in dataloader:
        clean = data["gt"]
        hazy = data["lq"]

        hazy_cv2 = tensor2img(hazy)

        I = hazy_cv2.astype('float64') / 255
        dark = DarkChannel(I, 15)
        A = AtmLight(I, dark)
        te = TransmissionEstimate(I, A, 15)
        t = TransmissionRefine(hazy_cv2, te)
        J = Recover(I, t, A, 0.1)
        recover = img2tensor(J)[None, ::]

        psnr_value = psnr(clean, recover).item() + psnr_value
        ssim_value = ssim(clean, recover).item() + ssim_value
        lpips_value = lpips(clean, recover).item() + lpips_value
        niqe_value = niqe(recover).item() + niqe_value
        nima_value = nima(recover).item() + nima_value
        brisque_value = brisque(recover).item() + brisque_value

        img_name = os.path.splitext(os.path.basename(data['gt_path'][0]))[0] + ".png"
        torchvision.utils.save_image(recover, os.path.join(img_save_path, img_name), normalize=False)
        iteration += 1

    print("PSNR: {}, SSIM: {}, LPIPS: {}, "
          "NIQE: {}, NIMA: {}, BRISQUE: {}"
          .format(psnr_value / iteration,
                  ssim_value / iteration,
                  lpips_value / iteration,
                  niqe_value / iteration,
                  nima_value / iteration,
                  brisque_value / iteration))
