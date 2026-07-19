"""Grid-search the three best nonlinear probe logits."""

from mnist_ssl.ensembles.nonlinear_probe_triplet import parse_args, run


if __name__ == "__main__":
    run(parse_args())
