"""Runnable demo.

    python -m grasp_embeddings.demo               # single probe -> out/demo.png
    python -m grasp_embeddings.demo --dataset 2000 # dataset scatter -> out/dataset.png
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from .hand import Hand
from .sampler import generate_dataset, save_dataset
from .shapes import letter_a


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=int, default=0,
                        help="generate a dataset of this many samples")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--outdir", default="out")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    shape = letter_a()
    hand = Hand.fan(n_fingers=5, spread_deg=80, max_length=3.0)
    rng = np.random.default_rng(args.seed)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .visualize import draw_dataset, draw_probe

    if args.dataset > 0:
        data = generate_dataset(shape, hand, n=args.dataset, rng=rng)
        save_dataset(os.path.join(args.outdir, "dataset.npz"), data)
        ax = draw_dataset(shape, data, finger=2)  # middle finger
        ax.set_title(f"{args.dataset} grasp samples (colored by center finger)")
        plt.savefig(os.path.join(args.outdir, "dataset.png"), dpi=120,
                    bbox_inches="tight")
        print(f"wrote {args.outdir}/dataset.npz and dataset.png")
        print("sample[0]:", data["position"][0], data["heading"][0],
              data["lengths"][0])
    else:
        # A few illustrative poses around the 'A'.
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        poses = [((-2.2, 0.0), 0.0), ((0.0, -2.6), np.pi / 2), ((1.6, 0.8), np.pi)]
        for ax, (p, h) in zip(axes, poses):
            draw_probe(shape, hand.at(p, h), ax=ax)
            ax.set_title(f"p={p}, h={h:.2f}\nl={np.round(hand.at(p, h).sense(shape), 2)}")
        plt.savefig(os.path.join(args.outdir, "demo.png"), dpi=120,
                    bbox_inches="tight")
        print(f"wrote {args.outdir}/demo.png")


if __name__ == "__main__":
    main()
