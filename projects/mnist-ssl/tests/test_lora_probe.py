"""Invariants for supervised LoRA backbone adaptation."""

from __future__ import annotations

import torch
import torch.nn as nn

from mnist_ssl.dinov2.model import VisionTransformer
from mnist_ssl.ijepa.custom_ijepa import build_model
from mnist_ssl.lora import (
    adapter_parameters,
    adapter_state_dict,
    capture_base_tensors,
    load_adapter_state_dict,
    tensor_fingerprint,
)
from mnist_ssl.lora_probe import FeatureExtractor, inject_transformer_lora, parse_positive_ints


def test_dino_lora_starts_exactly_at_the_frozen_backbone_and_updates_only_adapters() -> None:
    backbone = VisionTransformer(dim=48, depth=2, heads=3, drop_path_rate=0.0)
    backbone.eval()
    for parameter in backbone.parameters():
        parameter.requires_grad_(False)
    base_tensors = capture_base_tensors(backbone)
    before_hash = tensor_fingerprint(base_tensors)
    images = torch.randn(2, 1, 56, 56)
    before = backbone(images).detach()

    handles = inject_transformer_lora(backbone, "dinov2", rank=4, alpha=8)
    assert len(handles) == 10
    assert torch.equal(before, backbone(images))
    assert all(not tensor.requires_grad for _, tensor in base_tensors)

    optimizer = torch.optim.SGD(adapter_parameters(handles), lr=0.1)
    loss = backbone(images).square().mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    assert tensor_fingerprint(base_tensors) == before_hash
    assert any(torch.count_nonzero(handle.adapter.lora_b) for handle in handles)


def test_ijepa_lora_covers_qkv_attention_output_and_mlp_in_every_layer() -> None:
    model = build_model(enc_dim=32, n_targets=48)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    handles = inject_transformer_lora(model.target, "ijepa", rank=4, alpha=8)
    assert len(handles) == 16
    names = {handle.name for handle in handles}
    for layer in range(4):
        prefix = f"encoder.blocks.layers.{layer}"
        assert f"{prefix}.self_attn.in_proj_weight" in names
        assert f"{prefix}.self_attn.out_proj.weight" in names
        assert f"{prefix}.linear1.weight" in names
        assert f"{prefix}.linear2.weight" in names


def test_adapter_checkpoint_round_trip_excludes_base_weights() -> None:
    first = VisionTransformer(dim=48, depth=1, heads=3)
    second = VisionTransformer(dim=48, depth=1, heads=3)
    second.load_state_dict(first.state_dict())
    first_handles = inject_transformer_lora(first, "dinov2", rank=4, alpha=8)
    second_handles = inject_transformer_lora(second, "dinov2", rank=4, alpha=8)
    with torch.no_grad():
        for parameter in adapter_parameters(first_handles):
            parameter.add_(torch.randn_like(parameter))
    state = adapter_state_dict(first_handles)
    assert state
    assert all("original" not in key and "base" not in key for key in state)
    load_adapter_state_dict(second_handles, state)
    assert all(
        torch.equal(left, right)
        for left, right in zip(
            adapter_parameters(first_handles), adapter_parameters(second_handles)
        )
    )


def test_feature_extractors_match_requested_readouts() -> None:
    dino = VisionTransformer(dim=48, depth=1, heads=3)
    ijepa = build_model(enc_dim=32, n_targets=48)
    images = torch.randn(2, 1, 56, 56)
    assert FeatureExtractor(dino, "dinov2")(images).shape == (2, 48)
    assert FeatureExtractor(ijepa.target, "ijepa")(images).shape == (2, 64 * 32)


def test_milestones_are_positive_unique_and_sorted() -> None:
    assert parse_positive_ints("150,50,75,100,50") == (50, 75, 100, 150)
