"""Focused invariants for the from-scratch DINOv2 implementation."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))

from data import EvaluationTransform, MultiCropMNIST, make_masks, upscale_bbox
from eval_knn import weighted_knn_accuracy
from losses import CenteredTeacher, dino_loss, ibot_loss, koleo_loss
from model import StudentTeacher, VisionTransformer
from train import Config, checkpoint_payload, parameter_groups, resume_config_mismatches


def test_default_backbone_matches_custom_ijepa_scale():
    model = VisionTransformer()
    assert model.dim == 128
    assert len(model.blocks) == 4
    assert model.blocks[0].attention.heads == 4
    assert sum(parameter.numel() for parameter in model.parameters()) == 814_144


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


def test_dino_and_ibot_heads_are_independent_ema_pairs():
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
    assert (
        model.student_dino_head.prototype_weight.data_ptr()
        != model.student_ibot_head.prototype_weight.data_ptr()
    )
    assert torch.equal(
        model.student_dino_head.prototype_weight,
        model.teacher_dino_head.prototype_weight,
    )
    assert torch.equal(
        model.student_ibot_head.prototype_weight,
        model.teacher_ibot_head.prototype_weight,
    )
    before_ibot = model.student_ibot_head.prototype_weight.detach().clone()
    with torch.no_grad():
        model.student_dino_head.prototype_weight.add_(1.0)
    assert torch.equal(model.student_ibot_head.prototype_weight, before_ibot)
    model.update_teacher(0.5)
    assert not torch.equal(
        model.student_dino_head.prototype_weight,
        model.teacher_dino_head.prototype_weight,
    )
    assert torch.equal(
        model.student_ibot_head.prototype_weight,
        model.teacher_ibot_head.prototype_weight,
    )


def test_preprocessing_matches_custom_ijepa_upscale_bbox_pipeline():
    image = torch.zeros(1, 28, 28)
    image[:, 8:20, 10:17] = 1.0
    processed = upscale_bbox(image)
    assert processed.shape == (1, 56, 56)
    # The tight rectangular digit is stretched to fill the complete frame.
    assert processed[:, 0].max() > 0
    assert processed[:, -1].max() > 0
    assert processed[:, :, 0].max() > 0
    assert processed[:, :, -1].max() > 0


def test_preprocessing_is_default_for_training_and_eval_transforms():
    pil_image = Image.fromarray(torch.zeros(28, 28, dtype=torch.uint8).numpy())
    training = MultiCropMNIST(local_crops=2)
    views = training(pil_image)
    assert training.preprocess is True
    assert [tuple(view.shape) for view in views["global"]] == [(1, 56, 56)] * 2
    assert [tuple(view.shape) for view in views["local"]] == [(1, 28, 28)] * 2
    evaluation = EvaluationTransform()
    assert evaluation.preprocess is True
    assert evaluation(pil_image).shape == (1, 56, 56)


def test_weighted_knn_evaluation():
    train_features = torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]])
    train_features = torch.nn.functional.normalize(train_features, dim=-1)
    labels = torch.tensor([0, 0, 1, 1])
    accuracy = weighted_knn_accuracy(
        train_features, labels, train_features, labels, k=1, query_batch_size=2
    )
    assert accuracy == 1.0


def test_100_epoch_checkpoint_can_extend_to_300_or_500_only():
    saved = Config(epochs=100).__dict__
    assert resume_config_mismatches(saved, Config(epochs=300).__dict__, 100) == []
    assert resume_config_mismatches(saved, Config(epochs=500).__dict__, 100) == []
    assert resume_config_mismatches(saved, Config(epochs=75).__dict__, 100) == ["epochs"]

    changed = Config(epochs=300, dim=256).__dict__
    assert resume_config_mismatches(saved, changed, 100) == ["dim"]


def test_checkpoint_payload_contains_full_training_state():
    config = Config(dim=48, depth=2, heads=3, prototypes=32)
    model = StudentTeacher(
        dim=config.dim,
        depth=config.depth,
        heads=config.heads,
        prototypes=config.prototypes,
        head_hidden_dim=64,
        bottleneck_dim=16,
    )
    optimizer = torch.optim.AdamW(parameter_groups(model, config.weight_decay))
    class_center = CenteredTeacher(config.prototypes)
    patch_center = CenteredTeacher(config.prototypes)
    payload = checkpoint_payload(
        config, model, optimizer, class_center, patch_center, [], torch.device("cpu"), 0
    )
    required = {
        "teacher_backbone",
        "teacher_dino_head",
        "teacher_ibot_head",
        "student_backbone",
        "student_dino_head",
        "student_ibot_head",
        "class_center",
        "patch_center",
        "optimizer",
        "rng_state",
        "config",
        "completed_epoch",
        "global_step",
    }
    assert required <= payload.keys()
