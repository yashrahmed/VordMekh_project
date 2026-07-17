"""Evaluate frozen DINOv2 features with weighted k-NN and a linear probe."""

from mnist_ssl.dinov2.eval_frozen import evaluate, parse_args


if __name__ == "__main__":
    evaluate(parse_args())
