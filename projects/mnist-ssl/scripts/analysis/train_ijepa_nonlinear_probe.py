"""Train the matched nonlinear probe on the best frozen I-JEPA member."""

from mnist_ssl.ijepa.nonlinear_probe import parse_args, run


if __name__ == "__main__":
    run(parse_args())
