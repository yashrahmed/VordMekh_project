"""I-JEPA variants on MNIST, with frozen mean and flattened probes.

A focused companion to ``trials`` (which holds the MAE / multi-arch comparison).
Three runnable scripts:

* ``custom_ijepa``     -- pretrain the patch-token custom I-JEPA encoder.
* ``cnn_stem_ijepa``  -- pretrain a feature-space I-JEPA encoder with a dense CNN stem.
* ``train_probe`` -- train a linear probe / fine-tune on the flattened encoder output.
* ``eval_probe``  -- score a saved probe on the MNIST test split.

Images are always bbox-preprocessed and the seed defaults to 0 everywhere.
Checkpoints use the ``ijepa_mnist_`` / ``ijepa_clf_`` prefixes, disjoint from
``trials``' ``mae_mnist_`` / ``clf_mnist_`` names, so the two packages share
``models/`` without clobbering each other.
"""
