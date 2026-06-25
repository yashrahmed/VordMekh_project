"""ijepa_trials: a single-2x2-block I-JEPA on MNIST, with a flatten linear probe.

A focused companion to ``trials`` (which holds the MAE / multi-arch comparison).
Three runnable scripts:

* ``ijepa``       -- pretrain the I-JEPA encoder (single random 2x2 target block).
* ``train_probe`` -- train a linear probe / fine-tune on the flattened encoder output.
* ``eval_probe``  -- score a saved probe on the MNIST test split.

Images are always bbox-preprocessed and the seed defaults to 0 everywhere.
Checkpoints use the ``ijepa_mnist_`` / ``ijepa_clf_`` prefixes, disjoint from
``trials``' ``mae_mnist_`` / ``clf_mnist_`` names, so the two packages share
``models/`` without clobbering each other.
"""
