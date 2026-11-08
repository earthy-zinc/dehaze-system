import argparse
from pathlib import Path

import torch
import torchvision
from PIL import Image
from torchvision.transforms import ToTensor
from tqdm import tqdm

from basicsr.archs.itb_arch import FusionRefine


def main():
    """Inference demo for FeMaSR"""
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, default="inputs", help="Input image or folder")
    parser.add_argument("-w", "--weight", type=str, default=None, help="path for model weights")
    parser.add_argument("-o", "--output", type=str, default="results", help="Output folder")
    parser.add_argument("--use_weight", action="store_true")
    parser.add_argument("--alpha", type=float, default=1.0, help="value of alpha")
    parser.add_argument("--suffix", type=str, default="", help="Suffix of the restored image")
    parser.add_argument(
        "--max_size",
        type=int,
        default=10000,
        help="Max image size for whole image inference, otherwise use tiled_test",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight_path = args.weight

    # set up the model
    # sr_model = VQWeightNet(LQ_stage=True, use_weight=args.use_weight,
    #                       weight_alpha=args.alpha).to(device)
    # sr_model = VQWeightDehazeNet(codebook_params=[[64, 1024, 512]], LQ_stage=True,
    #                            use_weight=args.use_weight, weight_alpha=args.alpha).to(device)
    # sr_model = RIDCP(LQ_stage=True, use_weight=args.use_weight, weight_alpha=args.alpha).to(device)
    # sr_model = RIDCPNew(LQ_stage=True, use_weight=args.use_weight,
    #                    weight_alpha=args.alpha).to(device)
    sr_model = FusionRefine(
        opt={"LQ_stage": True, "use_weight": args.use_weight, "weight_alpha": args.alpha}
    ).to(device)
    sr_model.load_state_dict(torch.load(weight_path)["params"], strict=False)
    sr_model.eval()

    Path(args.output).mkdir(parents=True, exist_ok=True)
    input_path = Path(args.input)
    paths = [input_path] if input_path.is_file() else sorted(input_path.glob("*"))

    pbar = tqdm(total=len(paths), unit="image")
    for path in paths:
        img_name = path.name
        save_path = Path(args.output) / img_name
        pbar.set_description(f"测试 {img_name}")

        img_tensor = ToTensor()(Image.open(path).convert("RGB")).to(device)[None, ::]
        h, w = img_tensor.shape[2:]
        if h * w < args.max_size**2:
            output, _ = sr_model.test(img_tensor)
        elif h * w > (args.max_size * 2) ** 2:
            down_img = torch.nn.UpsamplingBilinear2d((h // 3, w // 3))(img_tensor)
            output = sr_model.test_tile(down_img, tile_size=960, tile_pad=64)
            output = torch.nn.UpsamplingBilinear2d((h, w))(output)
        else:
            down_img = torch.nn.UpsamplingBilinear2d((h // 2, w // 2))(img_tensor)
            output, _ = sr_model.test(down_img)
            output = torch.nn.UpsamplingBilinear2d((h, w))(output)
        torchvision.utils.save_image(output, save_path)
        pbar.update(1)
    pbar.close()


if __name__ == "__main__":
    main()
