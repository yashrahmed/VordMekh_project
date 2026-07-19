"""Evaluate the validation-selected normalized-image reranker on test."""

from mnist_ssl.dinov2.normalized_image_reranker_split_eval import parse_args, run


if __name__ == "__main__":
    run(parse_args())
