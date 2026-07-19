"""Train the independent normalized-image reranker above the DINO linear probe."""

from mnist_ssl.dinov2.normalized_image_reranker import parse_args, run


if __name__ == "__main__":
    run(parse_args())
