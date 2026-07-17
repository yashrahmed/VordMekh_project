# MNIST SSL roadmap

The original objective—exceeding 99.5% test accuracy with frozen,
self-supervised representations—has been reached by exploratory ensembles.
The next objective is to reproduce that performance with validation-selected
model and ensemble choices and to understand why the members fail differently.

## Current priorities

- [ ] Create a fixed train/validation protocol for all model and ensemble
  selection; evaluate the MNIST test set only after choices are locked.
- [ ] Re-run the DINO/I-JEPA ensemble search using validation-selected weights.
- [ ] Analyze the 21 test errors shared by DINOv2 and both 56x56 I-JEPA models.
- [ ] Account for known MNIST label errors when interpreting the upper bound.
- [ ] Train a vanilla ViT baseline from scratch.
- [ ] Try ConvNeXt as a modern convolutional comparison.
- [ ] Test whether a small transformer over frozen DINOv2/I-JEPA features can
  learn a useful visual strategy without unfreezing the backbones.

## Later research directions

- [ ] Explore decision-tree/neural-network hybrids.
- [ ] Test point tracking as a representation-learning signal.
- [ ] Study action-conditioned V-JEPA and trajectory sampling.
- [ ] Study VLA systems and latent visual reasoning.
- [ ] Follow the connection to Le-JEPA and LeWorldModel.

## Completed milestones

- [x] Train and evaluate ViT and convolutional masked autoencoders.
- [x] Train custom I-JEPA variants and frozen linear/k-NN probes.
- [x] Implement MNIST-scale DINOv2 from PyTorch primitives.
- [x] Verify teacher stop-gradient, EMA updates, centering, iBOT, KoLeo,
  multi-crop augmentation, checkpoint resumption, and frozen-backbone probes.
- [x] Reach 99.42% with an individual frozen DINOv2 CLS probe.
- [x] Reach 99.50% with a test-tuned I-JEPA triplet.
- [x] Reach 99.61% with a test-tuned DINO/I-JEPA triplet.

The detailed chronology, negative results, architecture notes, and historical
task lists are preserved in the [experiment log](docs/experiment-log.md).
