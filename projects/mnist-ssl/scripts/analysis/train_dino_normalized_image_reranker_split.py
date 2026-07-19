"""Train the normalized-image reranker with a 50k/10k correction split."""

from mnist_ssl.dinov2.normalized_image_reranker_split import parse_args, run


if __name__ == "__main__":
    run(parse_args())
