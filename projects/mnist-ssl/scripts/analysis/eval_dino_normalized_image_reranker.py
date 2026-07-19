"""Evaluate the frozen epoch-45 normalized-image reranker on MNIST test."""

from mnist_ssl.dinov2.normalized_image_reranker_eval import parse_args, run


if __name__ == "__main__":
    run(parse_args())
