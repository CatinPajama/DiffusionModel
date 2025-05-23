import os
import argparse
import torch
import torchvision

from diffusion.gaussian import (
    GausianDiffusion,
    CosineNoiseSchedule,
)
from diffusion.model import DiT
from diffusion.config import Config


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-o", "--output", type=str, default="./", help="output directory"
    )
    parser.add_argument("-w", "--guidance", type=int, default=1)
    parser.add_argument("-n", "--num_images", type=int)

    args = parser.parse_args()

    batch = args.num_images
    checkpoint = torch.load("model.pth", weights_only=True)
    config = Config(**checkpoint["config"])

    print(config)

    model = DiT(config).to(config.device)

    model.load_state_dict(checkpoint["model"])

    model.eval()

    schedule = CosineNoiseSchedule(config.timesteps, device=config.device)
    # schedule = LinearNoiseSchedule(config.timesteps, 1e-4, 0.02, device=config.device)

    gaussian_diffusion = GausianDiffusion(schedule)

    with torch.inference_mode():
        x = gaussian_diffusion.sample(model, batch, config.num_classes, args.guidance)

        samples = (x.cpu() + 1) / 2

        # PIECE OF SSHIT TORCHVISION REQUIRES [0,1] and not [0,255] >:(
        for i, sample in enumerate(samples):
            out_path = os.path.join(args.output, f"{i}.png")
            torchvision.utils.save_image(sample, out_path)
