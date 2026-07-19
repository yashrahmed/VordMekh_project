"""Train the frozen-feature DINO nonlinear-probe experiment."""

from mnist_ssl.dinov2.nonlinear_probe import parse_args, run


if __name__ == "__main__":
    run(parse_args())
