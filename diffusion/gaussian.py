import torch
import torch.nn as nn


class NoiseSchedule:
    """
    Base class for noise schedule
    """

    def __init__(self, timesteps: int, device: torch.device):
        self.timesteps = timesteps
        # self.alpha_hat = []
        # self.beta = []
        # self.alpha = []
        self.device = device


class LinearNoiseSchedule(NoiseSchedule):
    """
    Noise variance is increased linearly from a beta_start to beta_end over T.
    """

    def __init__(
        self,
        timesteps: int,
        beta_start: int,
        beta_end: int,
        device: torch.device = "cpu",
    ):

        super().__init__(timesteps, device)
        self.beta = torch.linspace(beta_start, beta_end, steps=timesteps, device=device)
        self.alpha = 1 - self.beta
        self.alpha_hat = self.alpha.cumprod(0)


class CosineNoiseSchedule(NoiseSchedule):
    """
    Cumulative noise follows a cosine curve
    """

    def __init__(self, timesteps: int, s: int = 0.008, device: torch.device = "cpu"):

        super().__init__(timesteps, device)

        steps = timesteps + 2
        x = torch.linspace(0, steps, steps, device=device)
        self.alpha_hat = torch.cos(((x / steps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        self.alpha_hat = self.alpha_hat / self.alpha_hat[0]
        self.alpha_hat = self.alpha_hat[: timesteps + 1]

        self.alpha = self.alpha_hat[1:] / self.alpha_hat[:-1]
        self.alpha = torch.clamp(self.alpha, min=1e-8, max=1.0)  # Safety clamp
        self.beta = 1.0 - self.alpha


class GausianDiffusion:
    """
    Expects a noise schedule to perform adding noise to image and sampling images
    """

    def __init__(self, schedule: NoiseSchedule):

        self.schedule = schedule

    def add_noise(self, x_t: torch.Tensor):
        """
        Adds noise based on type of schedule

        Args:
            x_t : Tensor of shape (batch,channels,img_height,img_width) normalized to [-1,1]

        Returns:
            Tensor : Same shape as input x_t
        """
        t = torch.randint(
            0, self.schedule.timesteps, (x_t.size(0),), device=self.schedule.device
        )
        noise = torch.randn_like(x_t, device=self.schedule.device)
        acap = self.schedule.alpha_hat[t][:, None, None, None]
        return x_t * torch.sqrt(acap) + noise * torch.sqrt(1 - acap), noise, t

    def sample(self, model: nn.Module, batch_size: int, num_classes, w=3):
        """
        Samples images of shape

        Args:
            model: diffusion model object
            batch_size : batch size

        Returns:
            Tensor : Tensor of shape (B,C,H,W) normalized to [-1,1]
        """
        labels = None
        if num_classes > 0:
            labels = torch.randint(
                1, num_classes + 1, (batch_size,), device=self.schedule.device
            )

        print(labels)

        x = torch.normal(0, 1, (batch_size, 1, 48, 48), device=self.schedule.device)

        # Reverse diffusion process
        for t in range(self.schedule.timesteps - 1, 0, -1):
            z = (
                torch.normal(0, 1, (batch_size, 1, 48, 48), device=self.schedule.device)
                if t > 1
                else torch.zeros((batch_size, 1, 48, 48), device=self.schedule.device)
            )
            coeff = (1 - self.schedule.alpha[t]) / torch.sqrt(
                1 - self.schedule.alpha_hat[t]
            )
            predicted_noise_cond = model(
                x, torch.full((batch_size,), t, device=self.schedule.device), labels
            )

            predicted_noise = predicted_noise_cond

            if w > 0:
                predicted_noise_uncond = model(
                    x,
                    torch.full((batch_size,), t, device=self.schedule.device),
                )
                predicted_noise += (1 - w) * predicted_noise_uncond

            x = (1 / torch.sqrt(self.schedule.alpha[t])) * (
                x - coeff * predicted_noise
            ) + z * torch.sqrt(self.schedule.beta[t])

        return x
