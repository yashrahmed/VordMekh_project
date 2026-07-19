"""Rerank the best nonlinear DINO probe from strict nested OOF pairs."""

from mnist_ssl.dinov2.nonlinear_pairwise_reranker import parse_args, run


if __name__ == "__main__":
    run(parse_args())
