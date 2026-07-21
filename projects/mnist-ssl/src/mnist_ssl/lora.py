"""Small, dependency-free LoRA parametrizations for transformer weights."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn as nn
from torch.nn.utils import parametrize


class LoRAWeight(nn.Module):
    """Add a trainable low-rank delta to a frozen two-dimensional weight."""

    def __init__(self, shape: torch.Size, rank: int, alpha: float) -> None:
        super().__init__()
        if len(shape) != 2:
            raise ValueError(f"LoRA requires a matrix, got shape {tuple(shape)}")
        out_features, in_features = shape
        if not 0 < rank <= min(out_features, in_features):
            raise ValueError(
                f"rank must be in [1, {min(out_features, in_features)}], got {rank}"
            )
        if alpha <= 0:
            raise ValueError(f"alpha must be positive, got {alpha}")
        self.rank = rank
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.lora_a = nn.Parameter(torch.empty(rank, in_features))
        self.lora_b = nn.Parameter(torch.zeros(out_features, rank))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    def forward(self, base_weight: torch.Tensor) -> torch.Tensor:
        return base_weight + (self.lora_b @ self.lora_a) * self.scaling


@dataclass(frozen=True)
class LoRAHandle:
    """A named adapter attached to one module parameter."""

    name: str
    adapter: LoRAWeight


def add_lora(
    module: nn.Module,
    parameter_name: str,
    *,
    logical_name: str,
    rank: int,
    alpha: float,
) -> LoRAHandle:
    """Parametrize ``module.<parameter_name>`` without changing its base tensor."""

    parameter = getattr(module, parameter_name, None)
    if not isinstance(parameter, nn.Parameter):
        raise ValueError(f"{logical_name} is not an nn.Parameter")
    parameter.requires_grad_(False)
    adapter = LoRAWeight(parameter.shape, rank, alpha).to(
        device=parameter.device, dtype=parameter.dtype
    )
    parametrize.register_parametrization(module, parameter_name, adapter)
    return LoRAHandle(logical_name, adapter)


def adapter_parameters(handles: Iterable[LoRAHandle]) -> list[nn.Parameter]:
    return [parameter for handle in handles for parameter in handle.adapter.parameters()]


def adapter_state_dict(handles: Iterable[LoRAHandle]) -> dict[str, torch.Tensor]:
    """Return only the portable LoRA tensors, never the frozen base weights."""

    state: dict[str, torch.Tensor] = {}
    for handle in handles:
        for key, value in handle.adapter.state_dict().items():
            state[f"{handle.name}.{key}"] = value.detach().cpu()
    return state


def load_adapter_state_dict(
    handles: Iterable[LoRAHandle], state: dict[str, torch.Tensor]
) -> None:
    handles = list(handles)
    expected = {
        f"{handle.name}.{key}"
        for handle in handles
        for key in handle.adapter.state_dict()
    }
    if set(state) != expected:
        missing = sorted(expected - set(state))
        unexpected = sorted(set(state) - expected)
        raise ValueError(
            f"LoRA state mismatch: missing={missing}, unexpected={unexpected}"
        )
    for handle in handles:
        prefix = f"{handle.name}."
        handle.adapter.load_state_dict(
            {
                key.removeprefix(prefix): value
                for key, value in state.items()
                if key.startswith(prefix)
            }
        )


def capture_base_tensors(module: nn.Module) -> tuple[tuple[str, torch.Tensor], ...]:
    """Capture references to the original state before LoRA is registered."""

    return tuple(sorted(module.state_dict(keep_vars=True).items()))


def tensor_fingerprint(tensors: Iterable[tuple[str, torch.Tensor]]) -> str:
    digest = hashlib.sha256()
    for name, tensor in tensors:
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()
