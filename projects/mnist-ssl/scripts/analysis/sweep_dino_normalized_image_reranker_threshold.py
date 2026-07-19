"""Sweep the frozen normalized-image reranker's training-set margin gate."""

from mnist_ssl.dinov2.normalized_image_reranker_sweep import parse_args, run


if __name__ == "__main__":
    run(parse_args())
