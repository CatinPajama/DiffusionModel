"""Configuration dataclasses for diffusion model components."""

from dataclasses import dataclass


@dataclass
class Config:
    """
    Configuration for diffusion model

    Attributes:
        img_size: dimension of image
        in_channels: number of input channels
        num_classes : number of class labels
        patch_size: dimension of each patch
        time_steps: time steps
        noise_schedule : cosine / linear
        embd_dim: dimension of embeddings
        num_dit_blocks: number of DiT blocks
        num_heads: number of heads in each multi-head attention of a single DiT block
        hidden_layer_ratio : dimension multiplier

    """

    img_size: tuple[int, int]
    in_channels: int
    num_classes: int
    patch_size: tuple[int, int]
    out_channels: int
    embed_dim: int = 288  # should be divisible by num_heads
    timesteps: int = 1000
    noise_schedule: str = "cosine"
    num_dit_blocks: int = 8
    num_heads: int = 6
    device: str = "cpu"
    hidden_layer_ratio: int = 4
