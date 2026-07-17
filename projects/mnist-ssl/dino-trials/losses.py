"""DINO class-token, iBOT masked-patch, and KoLeo objectives."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CenteredTeacher(nn.Module):
    """EMA centering and temperature sharpening for teacher prototypes."""

    def __init__(self, prototypes: int, momentum: float = 0.9):
        super().__init__()
        self.momentum = momentum
        self.register_buffer("center", torch.zeros(1, prototypes))

    @torch.no_grad()
    def probabilities(self, logits: torch.Tensor, temperature: float) -> torch.Tensor:
        return F.softmax((logits.float() - self.center) / temperature, dim=-1)

    @torch.no_grad()
    def update(self, logits: torch.Tensor) -> None:
        batch_center = logits.detach().float().reshape(-1, logits.size(-1)).mean(dim=0, keepdim=True)
        self.center.mul_(self.momentum).add_(batch_center, alpha=1.0 - self.momentum)


def distribution_cross_entropy(
    student_logits: torch.Tensor,
    teacher_probabilities: torch.Tensor,
    student_temperature: float,
) -> torch.Tensor:
    return -(teacher_probabilities * F.log_softmax(
        student_logits.float() / student_temperature, dim=-1
    )).sum(dim=-1)


def dino_loss(
    student_global: list[torch.Tensor],
    student_local: list[torch.Tensor],
    teacher_global: list[torch.Tensor],
    student_temperature: float,
) -> torch.Tensor:
    """Cross-view CLS loss; matching global views are deliberately excluded."""
    terms = []
    for student_index, student in enumerate(student_global):
        for teacher_index, teacher in enumerate(teacher_global):
            if student_index != teacher_index:
                terms.append(distribution_cross_entropy(student, teacher, student_temperature).mean())
    for student in student_local:
        for teacher in teacher_global:
            terms.append(distribution_cross_entropy(student, teacher, student_temperature).mean())
    if not terms:
        raise ValueError("DINO loss requires at least two global crops")
    return torch.stack(terms).mean()


def ibot_loss(
    student_patch_logits: list[torch.Tensor],
    teacher_patch_probabilities: list[torch.Tensor],
    masks: list[torch.Tensor],
    student_temperature: float,
) -> torch.Tensor:
    """Same-view masked patch-token distillation, normalized per image."""
    terms = []
    for student, teacher, mask in zip(
        student_patch_logits, teacher_patch_probabilities, masks
    ):
        token_loss = distribution_cross_entropy(student, teacher, student_temperature)
        weights = mask.float()
        per_image = (token_loss * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        terms.append(per_image.mean())
    return torch.stack(terms).mean()


def koleo_loss(features: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
    """Kozachenko-Leonenko entropy regularizer on nearest-neighbor distances."""
    if features.size(0) < 2:
        return features.new_zeros(())
    features = F.normalize(features.float(), dim=-1, eps=epsilon)
    similarities = features @ features.T
    similarities.fill_diagonal_(-1.0)
    neighbors = similarities.argmax(dim=1)
    distances = (features - features[neighbors]).norm(dim=-1)
    return -torch.log(distances + epsilon).mean()
