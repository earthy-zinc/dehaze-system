import os
from typing import Tuple

from PIL import Image
from natsort import natsort
from torch.utils import data as data
from torchvision.transforms import Compose, CenterCrop, ToTensor, ToPILImage, RandomCrop, RandomResizedCrop, \
    RandomHorizontalFlip, RandomVerticalFlip, RandomChoice

from basicsr.data.data_util import is_image_file
from basicsr.utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class Flickr2KDataset(data.Dataset):
    """
    从文件夹中加载用于神经网络训练使用的去雾数据集，主要用于Flickr2K、DIV2K数据集的加载。
    """

    def __init__(self, opt):
        """
        dataset_dir, 无雾图像数据集文件夹路径
        crop_size: 图像的裁剪尺寸

        由于每张图片大小不一，因此在输入神经网络之前需要将图片进行裁剪、旋转等变换(transform)。
        因此会有相应的裁剪方法。
        使用数据集会返回裁剪后的图片张量。
        """
        super(Flickr2KDataset, self).__init__()
        self.opt = opt
        dataset_dir = opt['dataset_dir']
        crop_size = opt['crop_size']
        self.image_path = [os.path.join(dataset_dir, x)
                           for x in natsort.natsorted(os.listdir(dataset_dir))
                           if is_image_file(x)]

        self.img_transform = self.clear_img_transform(crop_size)

    def __getitem__(self, index):
        """
        获取数据集中的一个图像和其对应的路径。
        采用python字典的形式返回
        """
        gt_image = self.img_transform(Image.open(self.image_path[index]))
        return {
            'gt': gt_image,
            'gt_path': self.image_path[index]
        }

    def __len__(self):
        """
            返回数据集中数据的大小
        """
        return len(self.image_path)

    @staticmethod
    def clear_img_transform(crop_size):
        """
            图像变换方法
            将清晰无雾图像按照crop_size给定的尺寸进行中心裁剪，
            然后将图像转换为张量。
        """
        return Compose([
            RandomResizedCrop(crop_size),
            # RandomChoice([
            #     RandomHorizontalFlip(p=0.5),
            #     RandomVerticalFlip(p=0.2)
            # ]),
            ToTensor(),
        ])

    def show_img_example(self, size=1, index=0):
        """
            以可视化的方式查看数据集中的部分图像
            size: 查看图像的数量，默认查看第一个
            index: 图像在数据集中的位置，默认为位置下标为0的图像
        """
        if index + size >= self.__len__():
            raise IndexError("请求的图像下标或数量超出数据集大小")
        pil_transforms = ToPILImage()
        for num in range(index, index + size):
            haze_img = pil_transforms(self[num][0])
            clear_img = pil_transforms(self[num][1])
            print("第%d有雾图像：" % num)
            haze_img.show()
            print("第%d清晰无雾图像：" % num)
            clear_img.show()
