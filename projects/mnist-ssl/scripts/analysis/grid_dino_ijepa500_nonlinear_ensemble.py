"""Grid-search the best DINO and I-JEPA-500 nonlinear probe ensemble."""

from mnist_ssl.ensembles.nonlinear_probe_pair import parse_args, run


if __name__ == "__main__":
    run(parse_args())
