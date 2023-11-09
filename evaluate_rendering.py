import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from pytorch_msssim import ms_ssim
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from tqdm import tqdm



def sort_by_prefix(paths):
    return sorted(paths, key=lambda p: str(p).split('_')[0])


def eval_rendering(render_path, output_path, device="cuda"):
    rendered_color_paths = [p for p in (render_path / "tracking_vis").glob('*') if "gt" not in str(p)]
    gt_color_paths = [p for p in (render_path / "tracking_vis").glob('*') if "gt" in str(p)]

    rendered_color_paths = sort_by_prefix(rendered_color_paths)
    gt_color_paths = sort_by_prefix(gt_color_paths)
    
    num_frames = len(rendered_color_paths)
    lpips_model = LearnedPerceptualImagePatchSimilarity(net_type='alex', normalize=True).to(device)

    psnr, lpips, ssim = [], [], []
    for i in tqdm(range(num_frames)):
        rendered_color = torch.load(rendered_color_paths[i]).to(device)
        gt_color = torch.load(gt_color_paths[i]).to(device)

        rendered_color = rendered_color.permute(2, 0, 1).float()
        gt_color = gt_color[0].permute(2, 0, 1).float()

        mse_loss = torch.nn.functional.mse_loss(rendered_color, gt_color)
        psnr_value = (-10. * torch.log10(mse_loss)).item()
        lpips_value = lpips_model(rendered_color[None], gt_color[None]).item()
        ssim_value = ms_ssim(rendered_color[None], gt_color[None], data_range=1.0, size_average=True).item()
        
        psnr.append(psnr_value)
        lpips.append(lpips_value)
        ssim.append(ssim_value)
    
    with open(str(output_path / "metrics.json"), "w") as metrics_file:
        json.dump({
            "psnr": sum(psnr) / num_frames,
            "lpips": sum(lpips) / num_frames,
            "ssim": sum(ssim) / num_frames
        }, metrics_file)
    
    x = list(range(len(psnr)))
    plt.figure(figsize=(10, 10))
    plt.scatter(x, psnr, label="PSNR")
    plt.scatter(x, lpips, label="LPIPS")
    plt.scatter(x, ssim, label="SSIM")
    plt.legend()
    plt.savefig(str(output_path / "metrics.png"))


def get_args():
    parser = argparse.ArgumentParser(description='Arguments to compute the mesh')
    parser.add_argument('--checkpoint_path', type=str, help='SLAM checkpoint path',
                        default="output/slam/full_experiment/")
    parser.add_argument('--output_path', type=str, help='Output path for the mesh', default="")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    output_path = args.output_path if args.output_path != "" else args.checkpoint_path
    ckpt_path = Path(args.checkpoint_path)

    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    device = "cuda"
    eval_rendering(ckpt_path, output_path, device=device)
