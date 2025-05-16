"""Diffusion Models with Transformer"""

import torch
import torch.nn as nn
from diffusion.config import Config


class Patchify2d(nn.Module):
    """
    Patchify flattens image into patches for transformer
    """

    def __init__(self, config: Config):

        super().__init__()

        ih, iw = config.img_size
        ph, pw = config.patch_size
        c = config.in_channels

        assert ih >= ph and ih % ph == 0
        assert iw >= pw and iw % pw == 0

        self.ph = ph
        self.pw = pw
        self.c = c

        self.unfold = nn.Unfold((ph, pw), stride=(ph, pw))
        self.proj = nn.Linear(ph * pw * c, config.embed_dim)

        # nn.init.zeros_(self.proj.weight)
        # nn.init.zeros_(self.proj.bias)

        self.fold = nn.Fold(
            config.img_size, config.patch_size, stride=config.patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to patchify input

        Args:
            x (torch.Tensor): Tensor of shape (batch_size,channels,height,width)

        Returns:
            torch.Tensor:   Patchified output of shape (batch_size,no. of patches,output dimension)
        """
        out = (
            self.unfold(x)
            .transpose(-2, -1)
            .reshape(x.shape[0], -1, self.pw * self.ph * self.c)
        )

        out = self.proj(out)
        return out

    def unpatchify(self, x):
        """
        Turns patches into image

        Args:
            x (Tensor) : Tensor of shape (B,L,C * P * P)
        Returns:
            Tensor : Tensor of shape (B,C,H,W)
        """
        return self.fold(x.transpose(-1, -2))


def positional_embedding2d(config: Config):
    """
    Sinusoidal 2D Positional Embedding
    """

    n_h = config.img_size[0] / config.patch_size[0]
    n_w = config.img_size[1] / config.patch_size[1]

    index_h = torch.arange(0, n_h, dtype=torch.float)
    index_w = torch.arange(0, n_w, dtype=torch.float)
    index_h, index_w = torch.meshgrid(index_h, index_w, indexing="ij")
    index_h = index_h.flatten()[:, None]
    index_w = index_w.flatten()[:, None]

    quarter_dim = config.embed_dim // 4

    log10 = torch.log(torch.tensor(10000.0))
    theta = torch.exp((torch.arange(0, quarter_dim) / quarter_dim) * -log10)[
        None, :
    ]  # (embd_dim,)

    embd_h = torch.concat(
        [torch.sin(index_h * theta), torch.cos(index_h * theta)], dim=-1
    )
    embd_w = torch.concat(
        [torch.sin(index_w * theta), torch.cos(index_w * theta)], dim=-1
    )

    return torch.concat([embd_h, embd_w], dim=-1)


class TimeEmbedding(nn.Module):
    """
    Sinusoidal Time Embedding
    """

    def __init__(self, config: Config):
        super().__init__()

        half_dim = config.embed_dim // 2

        log10 = torch.log(torch.tensor(100000.0))
        theta = torch.exp(
            (torch.arange(0, half_dim) / half_dim) * -log10
        )  # (embd_dim,)

        self.register_buffer("theta", theta)

        self.mlp = MLP(config)

    def forward(self, t):
        """
        Args:
            t (Tensor): Tensor (B,)

        Returns:
            Tensor: A tensor (B, embd_dim)
        """

        theta = t[:, None] * self.theta[None, :]

        embd = torch.concat((theta.sin(), theta.cos()), -1)

        return self.mlp(embd)


class MLP(nn.Module):
    """Multi Layer Perceptron"""

    def __init__(self, config: Config):
        super().__init__()
        hidden_layer_dim = config.hidden_layer_ratio * config.embed_dim
        self.model = nn.Sequential(
            nn.Linear(config.embed_dim, hidden_layer_dim),
            nn.SiLU(),
            nn.Linear(hidden_layer_dim, config.embed_dim),
        )

    def forward(self, x):
        """
        Args:
            x (Tensor) : Tensor of shape (B,...,in_channels)
        Returns:
            Tensor  : Tensor of shape (B,...,out_channels)
        """
        return self.model(x)


class DiTBlock(nn.Module):
    """DiT Block"""

    def __init__(self, config: Config):
        super().__init__()

        self.layer_norm1 = nn.LayerNorm(
            config.embed_dim, elementwise_affine=False, eps=1e-6
        )
        self.layer_norm2 = nn.LayerNorm(
            config.embed_dim, elementwise_affine=False, eps=1e-6
        )
        self.multi_head_attention = nn.MultiheadAttention(
            embed_dim=config.embed_dim, num_heads=config.num_heads, batch_first=True
        )
        self.ada_ln = nn.Sequential(
            nn.SiLU(), nn.Linear(config.embed_dim, 6 * config.embed_dim)
        )

        self.ffn = nn.Sequential(
            nn.Linear(config.embed_dim, config.embed_dim * 4),
            nn.SiLU(),
            nn.Linear(config.embed_dim * 4, config.embed_dim),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor):
        """
        Args:
            x (Tensor) : Tensor of shape (B,N,P*P*c)
            c (Tensor) : Tensor of shape (B,embd_dim)
        """

        alpha1, beta1, gamma1, alpha2, beta2, gamma2 = self.ada_ln(c).chunk(6, dim=1)

        norm_token = self.layer_norm1(x)
        norm_token = gamma1.unsqueeze(1) * norm_token + beta1.unsqueeze(1)

        att, _ = self.multi_head_attention(norm_token, norm_token, norm_token)

        scaled_token = alpha1.unsqueeze(1) * att

        res_scaled_token = x + scaled_token

        norm_token2 = self.layer_norm2(res_scaled_token)
        norm_token2 = gamma2.unsqueeze(1) * norm_token2 + beta2.unsqueeze(1)

        ff_out = self.ffn(norm_token2)

        scaled_ff_out = alpha2.unsqueeze(1) * ff_out

        return scaled_ff_out + res_scaled_token


class DiT(nn.Module):

    def __init__(self, config: Config):
        super().__init__()

        self.patchify = Patchify2d(config)
        self.time_embedder = TimeEmbedding(config)

        if config.num_classes > 0:
            self.label_embedder = nn.Embedding(config.num_classes + 1, config.embed_dim)

        self.register_buffer("pos_embed", positional_embedding2d(config))

        self.dit_blocks = nn.ModuleList(
            [DiTBlock(config) for _ in range(config.num_dit_blocks)]
        )

        self.layer_norm = nn.LayerNorm(
            config.embed_dim, elementwise_affine=False, eps=1e-6
        )

        self.ada_ln = nn.Sequential(
            nn.SiLU(), nn.Linear(config.embed_dim, 2 * config.embed_dim)
        )

        self.proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                config.embed_dim,
                config.patch_size[0] * config.patch_size[1] * config.out_channels,
            ),
        )

    def forward(self, x, t, c=None):
        """
        Args:
            x (Tensor): Tensor of shape (B,C,H,W)
            t (Tensor) : Time tensor of shape (B,)
            c (Tensor) : Class label tensor of shape (B,)
        """

        embed = self.time_embedder(t)

        if c is not None:
            embed += self.label_embedder(c)

        o = self.patchify(x) + self.pos_embed

        for block in self.dit_blocks:
            o = block(o, embed)

        o = self.layer_norm(o)

        beta_final, gamma_final = self.ada_ln(embed).chunk(2, dim=1)

        o = o * beta_final.unsqueeze(1) + gamma_final.unsqueeze(1)

        o = self.proj(o)

        return self.patchify.unpatchify(o)
