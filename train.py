from dataclasses import asdict
import argparse
import torch

from diffusion.model import DiT
from diffusion.model import Config
from diffusion.gaussian import (
    GausianDiffusion,
    CosineNoiseSchedule,
)
from utils.dataset import read_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    parser.add_argument(
        "-t", "--train_directory", type=str, help="path to train directory"
    )
    parser.add_argument(
        "-e", "--epochs", type=int, default=100, help="path to train directory"
    )
    parser.add_argument("-d", "--device", type=str, default=DEFAULT_DEVICE)

    parser.add_argument("-b", "--batch_size", type=int, default=32)
    parser.add_argument("-w", "--workers", type=int, default=1)
    parser.add_argument("-c", "--conditional", type=int)

    args = parser.parse_args()
    conditional = bool(args.conditional)

    dataloader, num_classes = read_dataset(
        args.train_directory, conditional, args.batch_size, args.workers
    )

    images, _ = next(iter(dataloader))

    config = Config(
        img_size=(images.shape[2], images.shape[3]),
        in_channels=images.shape[1],
        num_classes=num_classes,
        patch_size=(2, 2),
        out_channels=images.shape[1],
        device=args.device,
    )

    model = DiT(config)

    model = model.to(args.device)

    schedule = CosineNoiseSchedule(config.timesteps, device=args.device)
    # schedule = LinearNoiseSchedule(config.timesteps, 1e-4, 0.02, device=args.device)
    gaussian_diffusion = GausianDiffusion(schedule)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    criterion = torch.nn.MSELoss()
    scaler = torch.amp.GradScaler(args.device)

    DROPOUT = 0.1

    for epoch in range(args.epochs):
        train_loss = 0
        i = 0
        for xt, c in dataloader:

            model.train()

            xt = xt.to(args.device)

            if conditional:
                mask = torch.rand(c.shape)
                c[mask < DROPOUT] = 0
                c = c.to(args.device)
            else:
                c = None

            noised_xt, E, t = gaussian_diffusion.add_noise(xt)
            # with torch.cuda.amp.autocast():

            with torch.amp.autocast(args.device):
                e_pred = model(noised_xt, t, c)
                loss = criterion(e_pred, E)

            optimizer.zero_grad()
            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

            i += 1

        train_loss /= len(dataloader)

        if epoch % (args.epochs // 10) == 0:
            print(f"Epoch = {epoch} {train_loss}")

    model_to_save = model.module if isinstance(model, torch.nn.DataParallel) else model
    torch.save(
        {"model": model_to_save.state_dict(), "config": asdict(config)}, "model.pth"
    )
