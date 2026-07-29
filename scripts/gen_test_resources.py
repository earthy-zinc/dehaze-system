"""生成 dehaze-sdk-js 集成测试所需的占位图片资源。

包含两类输出：
1. dehaze-sdk-js/test/resources/  本地上传用占位图片
2. datasets/  模拟数据集图片，经根目录 datasets 符号链接/挂载由 nginx-dataset
   容器提供 http://{host}:9000/datasets/... 访问，供预测/评测接口按 URL 下载

图片带渐变 + 高斯噪声纹理（非纯色），确保：
- 每张字节内容不同，避免后端文件 hash 去重导致 fileName 复用
- 含足够局部方差，NIQE 等无参考图像质量评估指标可正常计算
  （纯色图片协方差矩阵奇异会导致 linalg.svd 不收敛）

用法（需 Pillow + numpy）：
    python scripts/gen_test_resources.py
幂等，重复执行覆盖写入。
"""

import os

import numpy as np
from PIL import Image

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESOURCES_DIR = os.path.join(ROOT_DIR, "dehaze-sdk-js", "test", "resources")
DATASETS_DIR = os.path.join(ROOT_DIR, "datasets")

# (相对路径, 随机种子, 是否加雾)
IMAGES = [
    ("test/clean/41_outdoor_GT.jpg", 41, False),
    ("test/clean/42_outdoor_GT.jpg", 42, False),
    ("test/clean/43_outdoor_GT.jpg", 43, False),
    ("test/clean/44_outdoor_GT.jpg", 44, False),
    ("test/clean/45_outdoor_GT.jpg", 45, False),
    ("test/hazy/41_outdoor_hazy.jpg", 41, True),
    ("test/hazy/42_outdoor_hazy.jpg", 42, True),
    ("test/hazy/43_outdoor_hazy.jpg", 43, True),
    ("test2/clean/0025.jpg", 25, False),
    ("test2/hazy/0025_0.8_0.04.jpg", 251, True),
    ("test2/hazy/0025_0.8_0.08.jpg", 252, True),
    ("test2/hazy/0025_0.9_0.12.jpg", 253, True),
    # model.test.ts 专用：与 item-file 的图片 MD5 不同，避免并行上传时唯一索引冲突（B0405）
    ("test/model/hazy.jpg", 941, True),
    ("test/model/clear.jpg", 942, False),
]

PNG_IMAGES = [
    ("test3/cqupt.png", 100),
]

# 模拟数据集占位图片：写入 datasets/，由 nginx-dataset 容器挂载提供 URL 访问
# 对应 test/factories/model.ts 中引用的 NH-HAZE-2023 数据集路径
DATASET_IMAGES = [
    ("NH-HAZE-2023/hazy/001.JPG", 1011, True),
    ("NH-HAZE-2023/clean/001.JPG", 1012, False),
]


def make_image(seed: int, hazy: bool) -> Image.Image:
    rng = np.random.RandomState(seed)
    x = np.linspace(0, 255, 128).reshape(1, -1, 1)
    y = np.linspace(0, 200, 128).reshape(-1, 1, 1)
    grad = np.repeat((x + y) / 2, 3, axis=2)
    img = grad + rng.normal(0, 25, (128, 128, 3)) + rng.randint(0, 50)
    if hazy:
        img = img * 0.6 + 100
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8), "RGB")


def main() -> None:
    for rel, seed, hazy in IMAGES:
        path = os.path.join(RESOURCES_DIR, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        make_image(seed, hazy).save(path, "JPEG", quality=90)
        print(f"{rel}: {os.path.getsize(path)} bytes")

    for rel, seed in PNG_IMAGES:
        path = os.path.join(RESOURCES_DIR, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        make_image(seed, hazy=False).save(path, "PNG")
        print(f"{rel}: {os.path.getsize(path)} bytes")

    for rel, seed, hazy in DATASET_IMAGES:
        path = os.path.join(DATASETS_DIR, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        make_image(seed, hazy).save(path, "JPEG", quality=90)
        print(f"datasets/{rel}: {os.path.getsize(path)} bytes")


if __name__ == "__main__":
    main()
