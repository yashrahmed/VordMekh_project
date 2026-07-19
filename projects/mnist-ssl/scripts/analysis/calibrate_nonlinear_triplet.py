"""Fit training-only temperatures and class-specific diagonal ensemble weights."""

from mnist_ssl.ensembles.temperature_diagonal import parse_args, run


if __name__ == "__main__":
    run(parse_args())
