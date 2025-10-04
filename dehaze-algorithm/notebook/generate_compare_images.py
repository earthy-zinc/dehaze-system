import pyiqa
from PIL import Image
from torchvision.transforms import ToTensor
import glob
import os
from matplotlib import pyplot as plt

niqe = pyiqa.create_metric("niqe")
nima = pyiqa.create_metric("nima")
brisque = pyiqa.create_metric("brisque")
psnr = pyiqa.create_metric("psnr")
ssim = pyiqa.create_metric("ssim")
lpips = pyiqa.create_metric("lpips")

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    '.tif', '.TIF', '.tiff', '.TIFF', '.npy'
]
order_list = [
    "hazy",
    "DCP",
    "AOD",
    "GCA",
    "FFA",
    "TNN",
    "Dehamer",
    "FogRemoval",
    "ITB",
    "SCA",
    "ours",
    "clean"
]


def custom_order(x):
    # 定义自定义排序规则
    if "hazy" in x:
        return 1
    if "DCP-" in x:
        return 2
    if "AOD" in x:
        return 3
    if "GCA" in x:
        return 4
    if "FFA" in x:
        return 5
    if "TNN" in x:
        return 6
    if "Dehamer" in x:
        return 7
    if "FogRemoval" in x:
        return 8
    if "ITB" in x:
        return 9
    if "SCA" in x:
        return 10
    if "clean" in x:
        return 12
    return 11


def custom_order_for_img(x):
    _, filename = os.path.split(x)
    if "_" in filename:
        return int(filename.split("_")[0])
    else:
        return int(filename.split(".")[0])


def get_image_list(root_path, dataset_name, clean_path, haze_path):
    exp_dir = []
    for root, dirs, files in os.walk(root_path):
        exp_dir = dirs
        break

    exp_dir = [os.path.join(root_path, x) for x in exp_dir if dataset_name in x]
    exp_dir.append(os.path.abspath(haze_path))
    exp_dir.append(os.path.abspath(clean_path))

    exp_dir = sorted(exp_dir, key=custom_order)
    exp_imgs = []
    for _, x in enumerate(exp_dir):
        temp_list = glob.glob(x + "/**/*.*", recursive=True)
        temp_list = [x for x in temp_list if os.path.splitext(x)[-1] in IMG_EXTENSIONS]
        exp_imgs.append(sorted(temp_list, key=custom_order_for_img))
    assert len(exp_imgs) == 12, "Number of experiments should be 12: " + str(len(exp_dir))
    for i, sublist in enumerate(exp_imgs):
        assert len(sublist) == len(exp_imgs[0]), "第" + str(i) + "项中，" + str(len(sublist)) + "不等于" + str(len(exp_imgs[0]))
    return exp_imgs


def generate_comparison_images(root_path, dataset_name, clean_path, haze_path):
    result_path = os.path.abspath("../compare/" + dataset_name)
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    exp_imgs = get_image_list(root_path, dataset_name, clean_path, haze_path)

    for count in range(len(exp_imgs[0])):
        plt.rcParams['figure.dpi'] = 600
        fig, axs = plt.subplots(1, 12)
        # clean_img = ToTensor()(Image.open(exp_imgs[0][count]).convert('RGB'))[None, ::]
        for i in range(len(exp_imgs)):
            # current_img = ToTensor()(Image.open(exp_imgs[i][count]).convert('RGB'))[None, ::]
            # title = """
            # {:s}\n
            # NIQE: {:.2f}\n
            # NIMA: {:.2f}\n
            # BRISQUE: {:.2f}\n
            # PSNR: {:.2f}\n
            # SSIM: {:.2f}\n
            # LPIPS: {:.2f}
            # """.format(
            #     order_list[i],
            #     niqe(current_img).item(),
            #     nima(current_img).item(),
            #     brisque(current_img).item(),
            #     psnr(clean_img, current_img).item(),
            #     ssim(clean_img, current_img).item(),
            #     lpips(clean_img, current_img).item()
            # )
            #
            axs[i].imshow(plt.imread(exp_imgs[i][count]))
            axs[i].set_title(order_list[i], fontproperties={'size': 4})
            axs[i].axis('off')
        plt.tight_layout()
        plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0.01, hspace=0)
        plt.savefig(os.path.join(result_path, 'COMPARE-' + str(count) + '.png'), bbox_inches='tight', pad_inches=0)
        # plt.show()
        plt.close()


if __name__ == '__main__':
    # generate_comparison_images("/mnt/e/DeepLearningCopies/2023/RIDCP/results",
    #                            "NH-HAZE-20",
    #                            "/mnt/d/DeepLearning/dataset/NH-HAZE-2020/clean",
    #                            "/mnt/d/DeepLearning/dataset/NH-HAZE-2020/hazy")
    generate_comparison_images("/mnt/e/DeepLearningCopies/2023/RIDCP/results",
                               "NH-HAZE-21",
                               "/mnt/d/DeepLearning/dataset/NH-HAZE-2021/clean",
                               "/mnt/d/DeepLearning/dataset/NH-HAZE-2021/hazy")
    # generate_comparison_images("/mnt/e/DeepLearningCopies/2023/RIDCP/results",
    #                            "NH-HAZE-23",
    #                            "/mnt/d/DeepLearning/dataset/NH-HAZE-2023/clean",
    #                            "/mnt/d/DeepLearning/dataset/NH-HAZE-2023/hazy")
    # generate_comparison_images("/mnt/e/DeepLearningCopies/2023/RIDCP/results",
    #                            "O-HAZE",
    #                            "/mnt/d/DeepLearning/dataset/O-HAZE/clean",
    #                            "/mnt/d/DeepLearning/dataset/O-HAZE/hazy")
    # generate_comparison_images("/mnt/e/DeepLearningCopies/2023/RIDCP/results",
    #                            "I-HAZE",
    #                            "/mnt/d/DeepLearning/dataset/I-HAZE/clean",
    #                            "/mnt/d/DeepLearning/dataset/I-HAZE/hazy")
