"""Train the MNIST-scale DINOv2 implementation."""

from mnist_ssl.dinov2.train import parse_args, train


if __name__ == "__main__":
    train(*parse_args())
