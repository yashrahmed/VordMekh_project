"""Train the leakage-controlled DINO top-two pairwise reranker."""

from mnist_ssl.dinov2.pairwise_reranker import parse_args, run


if __name__ == "__main__":
    run(parse_args())
