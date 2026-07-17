"""From-scratch Vision Transformer and DINO projection head for MNIST.

Only PyTorch tensor/layer primitives are used.  In particular, this module does
not depend on timm, the official DINOv2 package, or another SSL implementation.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class DropPath(nn.Module):
    """Per-sample stochastic depth."""

    def __init__(self, probability: float = 0.0):
        super().__init__()
        self.probability = probability

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.probability == 0.0:
            return x
        keep = 1.0 - self.probability
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep + torch.rand(shape, dtype=x.dtype, device=x.device)
        return x * random_tensor.floor() / keep


class Attention(nn.Module):
    """Multi-head self-attention with an explicitly implemented QKV path."""

    def __init__(self, dim: int, heads: int):
        super().__init__()
        if dim % heads:
            raise ValueError(f"dim={dim} must be divisible by heads={heads}")
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, tokens, dim = x.shape
        qkv = self.qkv(x).reshape(batch, tokens, 3, self.heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        attention = (q @ k.transpose(-2, -1) * self.scale).softmax(dim=-1)
        x = (attention @ v).transpose(1, 2).reshape(batch, tokens, dim)
        return self.proj(x)


class SwiGLUFFN(nn.Module):
    """SwiGLU-style feed-forward block used by the DINOv2 recipe."""

    def __init__(self, dim: int, mlp_ratio: float = 4.0):
        super().__init__()
        hidden = int(dim * mlp_ratio * 2 / 3)
        hidden = math.ceil(hidden / 8) * 8
        self.gate = nn.Linear(dim, hidden)
        self.value = nn.Linear(dim, hidden)
        self.out = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(F.silu(self.gate(x)) * self.value(x))


class TransformerBlock(nn.Module):
    """Pre-normalized ViT block with LayerScale and stochastic depth."""

    def __init__(
        self,
        dim: int,
        heads: int,
        drop_path: float,
        layer_scale: float = 1e-5,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attention = Attention(dim, heads)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.ffn = SwiGLUFFN(dim)
        self.attn_scale = nn.Parameter(torch.full((dim,), layer_scale))
        self.ffn_scale = nn.Parameter(torch.full((dim,), layer_scale))
        self.drop_path = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.attention(self.norm1(x)) * self.attn_scale)
        x = x + self.drop_path(self.ffn(self.norm2(x)) * self.ffn_scale)
        return x


class VisionTransformer(nn.Module):
    """Small DINOv2-style ViT supporting global/local crops and patch masks."""

    def __init__(
        self,
        image_size: int = 56,
        patch_size: int = 7,
        in_channels: int = 1,
        dim: int = 128,
        depth: int = 4,
        heads: int = 4,
        drop_path_rate: float = 0.1,
    ):
        super().__init__()
        if image_size % patch_size:
            raise ValueError("image_size must be divisible by patch_size")
        self.image_size = image_size
        self.patch_size = patch_size
        self.dim = dim
        self.base_grid = image_size // patch_size
        self.patch_embed = nn.Conv2d(
            in_channels, dim, kernel_size=patch_size, stride=patch_size
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, 1 + self.base_grid**2, dim)
        )
        drop_rates = torch.linspace(0, drop_path_rate, depth).tolist()
        self.blocks = nn.ModuleList(
            [TransformerBlock(dim, heads, drop_rates[i]) for i in range(depth)]
        )
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.apply(self._init_weights)
        nn.init.normal_(self.cls_token, std=1e-6)
        nn.init.normal_(self.mask_token, std=1e-6)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def _positions(self, height: int, width: int) -> torch.Tensor:
        grid_h, grid_w = height // self.patch_size, width // self.patch_size
        if grid_h == self.base_grid and grid_w == self.base_grid:
            return self.pos_embed
        cls_pos, patch_pos = self.pos_embed[:, :1], self.pos_embed[:, 1:]
        patch_pos = patch_pos.reshape(1, self.base_grid, self.base_grid, self.dim)
        patch_pos = patch_pos.permute(0, 3, 1, 2)
        patch_pos = F.interpolate(
            patch_pos, size=(grid_h, grid_w), mode="bicubic", align_corners=False
        )
        patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, grid_h * grid_w, self.dim)
        return torch.cat((cls_pos, patch_pos), dim=1)

    def forward_features(
        self, images: torch.Tensor, masks: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, _, height, width = images.shape
        patches = self.patch_embed(images).flatten(2).transpose(1, 2)
        if masks is not None:
            if masks.shape != patches.shape[:2]:
                raise ValueError(
                    f"mask shape {tuple(masks.shape)} does not match patch grid "
                    f"{tuple(patches.shape[:2])}"
                )
            patches = torch.where(
                masks.unsqueeze(-1), self.mask_token.expand(batch, patches.size(1), -1), patches
            )
        cls = self.cls_token.expand(batch, -1, -1)
        x = torch.cat((cls, patches), dim=1) + self._positions(height, width)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return x[:, 0], x[:, 1:]

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.forward_features(images)[0]


class DINOHead(nn.Module):
    """Three-layer projection head followed by normalized prototypes."""

    def __init__(
        self,
        in_dim: int,
        prototypes: int = 1024,
        hidden_dim: int = 512,
        bottleneck_dim: int = 128,
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, bottleneck_dim),
        )
        self.prototype_weight = nn.Parameter(torch.empty(prototypes, bottleneck_dim))
        self.apply(self._init_weights)
        nn.init.trunc_normal_(self.prototype_weight, std=0.02)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.normalize(self.mlp(x), dim=-1, eps=1e-12)
        prototypes = F.normalize(self.prototype_weight, dim=-1, eps=1e-12)
        return F.linear(x, prototypes)


class StudentTeacher(nn.Module):
    """DINOv2 student and EMA teacher with independent CLS and patch heads."""

    def __init__(
        self,
        image_size: int = 56,
        patch_size: int = 7,
        dim: int = 128,
        depth: int = 4,
        heads: int = 4,
        prototypes: int = 1024,
        head_hidden_dim: int = 512,
        bottleneck_dim: int = 128,
        drop_path_rate: float = 0.1,
    ):
        super().__init__()
        backbone_args = dict(
            image_size=image_size,
            patch_size=patch_size,
            dim=dim,
            depth=depth,
            heads=heads,
            drop_path_rate=drop_path_rate,
        )
        head_args = dict(
            in_dim=dim,
            prototypes=prototypes,
            hidden_dim=head_hidden_dim,
            bottleneck_dim=bottleneck_dim,
        )
        self.student_backbone = VisionTransformer(**backbone_args)
        self.student_dino_head = DINOHead(**head_args)
        self.student_ibot_head = DINOHead(**head_args)
        self.teacher_backbone = VisionTransformer(**backbone_args)
        self.teacher_dino_head = DINOHead(**head_args)
        self.teacher_ibot_head = DINOHead(**head_args)
        self.teacher_backbone.load_state_dict(self.student_backbone.state_dict())
        self.teacher_dino_head.load_state_dict(self.student_dino_head.state_dict())
        self.teacher_ibot_head.load_state_dict(self.student_ibot_head.state_dict())
        for parameter in self.teacher_parameters():
            parameter.requires_grad_(False)
        self.teacher_backbone.eval()
        self.teacher_dino_head.eval()
        self.teacher_ibot_head.eval()

    def student_parameters(self):
        yield from self.student_backbone.parameters()
        yield from self.student_dino_head.parameters()
        yield from self.student_ibot_head.parameters()

    def teacher_parameters(self):
        yield from self.teacher_backbone.parameters()
        yield from self.teacher_dino_head.parameters()
        yield from self.teacher_ibot_head.parameters()

    @torch.no_grad()
    def update_teacher(self, momentum: float) -> None:
        for student, teacher in zip(self.student_parameters(), self.teacher_parameters()):
            teacher.mul_(momentum).add_(student.detach(), alpha=1.0 - momentum)

    def train(self, mode: bool = True):
        super().train(mode)
        # The teacher must remain deterministic even while the student uses DropPath.
        self.teacher_backbone.eval()
        self.teacher_dino_head.eval()
        self.teacher_ibot_head.eval()
        return self

    @torch.no_grad()
    def encode(self, images: torch.Tensor, pool: str = "cls") -> torch.Tensor:
        cls, patches = self.teacher_backbone.forward_features(images)
        if pool == "cls":
            return cls
        if pool == "mean":
            return patches.mean(dim=1)
        if pool == "concat":
            return torch.cat((cls, patches.mean(dim=1)), dim=-1)
        raise ValueError(f"unknown pool mode: {pool}")
