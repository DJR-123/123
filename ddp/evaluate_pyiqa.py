import os
import glob
import argparse
import torch
import pyiqa
from PIL import Image
from tqdm import tqdm
from torchvision.transforms import ToTensor


def load_image(path):
    # 读取图片并归一化到 [0, 1]
    img = Image.open(path).convert("RGB")
    return ToTensor()(img).unsqueeze(0).cuda()


def main():
    parser = argparse.ArgumentParser()
    # 你的生成结果文件夹
    parser.add_argument(
        "--input_dir", type=str, required=True, help="Path to generated images"
    )
    # 你的真值(GT)文件夹，如果是无参考指标(NIQE)可以不传，但计算PSNR/SSIM/LPIPS必须传
    parser.add_argument(
        "--target_dir", type=str, default=None, help="Path to GT images"
    )
    # 如果生成图和GT混在一个文件夹，可以通过后缀区分
    parser.add_argument(
        "--input_suffix",
        type=str,
        default="",
        help="Suffix for input images (e.g., _sample.png)",
    )
    parser.add_argument(
        "--target_suffix",
        type=str,
        default="",
        help="Suffix for target images (e.g., _gt.png)",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 初始化指标
    print("Initializing metrics...")
    # Full Reference Metrics
    metric_psnr = pyiqa.create_metric("psnr", device=device)
    metric_ssim = pyiqa.create_metric("ssim", device=device)
    metric_lpips = pyiqa.create_metric("lpips", device=device)
    # No Reference Metric
    metric_niqe = pyiqa.create_metric("niqe", device=device)

    # 2. 获取文件列表
    # 假设都在 input_dir 里，或者分两个文件夹
    if args.target_dir is None:
        # 假设生成图和GT都在 input_dir，通过后缀区分
        # 例如：test_0_results_sample.png 和 test_0_lin_gt.png
        # 此时 input_dir 就是那个 images 文件夹
        search_path = os.path.join(args.input_dir, "*" + args.input_suffix)
        input_paths = sorted(glob.glob(search_path))

        # 构造对应的 GT 路径
        target_paths = []
        for p in input_paths:
            # 简单替换后缀来找 GT
            gt_p = p.replace(args.input_suffix, args.target_suffix)
            if os.path.exists(gt_p):
                target_paths.append(gt_p)
            else:
                print(f"Warning: GT not found for {p}, expected {gt_p}")
                target_paths.append(None)
    else:
        # 分两个文件夹的情况
        input_paths = sorted(glob.glob(os.path.join(args.input_dir, "*")))
        target_paths = sorted(glob.glob(os.path.join(args.target_dir, "*")))

    # 3. 开始评估
    avg_psnr, avg_ssim, avg_lpips, avg_niqe = 0.0, 0.0, 0.0, 0.0
    count = 0

    print(f"Found {len(input_paths)} images. Starting evaluation...")

    for i, input_p in tqdm(enumerate(input_paths), total=len(input_paths)):
        try:
            input_tensor = load_image(input_p)

            # 计算 NIQE (不需要 GT)
            score_niqe = metric_niqe(input_tensor).item()
            avg_niqe += score_niqe

            # 如果有 GT，计算全参考指标
            target_p = target_paths[i]
            if target_p is not None:
                target_tensor = load_image(target_p)

                # 确保尺寸一致，不一致稍微 resize 一下 target
                if input_tensor.shape != target_tensor.shape:
                    target_tensor = torch.nn.functional.interpolate(
                        target_tensor, size=input_tensor.shape[2:]
                    )

                score_psnr = metric_psnr(input_tensor, target_tensor).item()
                score_ssim = metric_ssim(input_tensor, target_tensor).item()
                score_lpips = metric_lpips(input_tensor, target_tensor).item()

                avg_psnr += score_psnr
                avg_ssim += score_ssim
                avg_lpips += score_lpips

            count += 1

        except Exception as e:
            print(f"Error processing {input_p}: {e}")

    if count > 0:
        print("\n" + "=" * 20 + " Results " + "=" * 20)
        print(f"Avg PSNR:  {avg_psnr / count:.4f}")
        print(f"Avg SSIM:  {avg_ssim / count:.4f}")
        print(f"Avg LPIPS: {avg_lpips / count:.4f}")
        print(f"Avg NIQE:  {avg_niqe / count:.4f}")
        print("=" * 49)
    else:
        print("No valid image pairs found.")


if __name__ == "__main__":
    main()


# 生成图和 GT 都在同一个文件夹里，运行命令：
# python evaluate_pyiqa.py --input_dir /home/dujieru/my_project/hdrev-diff/results/Test_123456-2026-03-18T15-32-34/images --input_suffix _results.jpg --target_suffix _new_gt.jpg(替换input路径即可)
