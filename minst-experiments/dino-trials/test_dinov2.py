"""Focused invariants for the from-scratch DINOv2 implementation."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))

from data import make_masks
from losses import CenteredTeacher, dino_loss, ibot_loss, koleo_loss
from model import StudentTeacher, VisionTransformer


def test_vit_supports_global_local_and_masked_views():
    model = VisionTransformer(image_size=56, patch_size=7, dim=48, depth=2, heads=3)
    global_images = torch.randn(2, 1, 56, 56)
    masks = make_masks(2, 8, torch.device("cpu"), sample_probability=1.0)
    cls, patches = model.forward_features(global_images, masks)
    assert cls.shape == (2, 48)
    assert patches.shape == (2, 64, 48)
    local_cls, local_patches = model.forward_features(torch.randn(2, 1, 28, 28))
    assert local_cls.shape == (2, 48)
    assert local_patches.shape == (2, 16, 48)
    assert masks.all(dim=1).logical_not().all()
    assert masks.any(dim=1).all()


def test_losses_are_finite_and_backpropagate_only_student():
    batch, prototypes = 4, 32
    student_global = [torch.randn(batch, prototypes, requires_grad=True) for _ in range(2)]
    student_local = [torch.randn(batch, prototypes, requires_grad=True) for _ in range(2)]
    teacher_logits = [torch.randn(batch, prototypes) for _ in range(2)]
    center = CenteredTeacher(prototypes)
    teacher = [center.probabilities(x, 0.07) for x in teacher_logits]
    global_loss = dino_loss(student_global, student_local, teacher, 0.1)
    masks = [torch.rand(batch, 8) > 0.5 for _ in range(2)]
    student_patches = [torch.randn(batch, 8, prototypes, requires_grad=True) for _ in range(2)]
    teacher_patches = [torch.softmax(torch.randn(batch, 8, prototypes), -1) for _ in range(2)]
    patch_loss = ibot_loss(student_patches, teacher_patches, masks, 0.1)
    entropy_loss = koleo_loss(torch.randn(batch, 16, requires_grad=True))
    loss = global_loss + patch_loss + entropy_loss
    assert torch.isfinite(loss)
    loss.backward()
    assert all(x.grad is not None for x in student_global + student_local + student_patches)


def test_teacher_is_ema_only_and_updates():
    model = StudentTeacher(
        image_size=28,
        patch_size=7,
        dim=48,
        depth=2,
        heads=3,
        prototypes=32,
        head_hidden_dim=64,
        bottleneck_dim=16,
    )
    assert all(not parameter.requires_grad for parameter in model.teacher_parameters())
    before = [parameter.clone() for parameter in model.teacher_parameters()]
    with torch.no_grad():
        next(model.student_parameters()).add_(1.0)
    model.update_teacher(0.5)
    assert not torch.equal(before[0], next(model.teacher_parameters()))
