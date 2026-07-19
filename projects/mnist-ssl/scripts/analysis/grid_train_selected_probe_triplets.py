"""Select linear and nonlinear triplet weights on train, then evaluate test."""

from mnist_ssl.ensembles.training_selected_triplets import parse_args, run


if __name__ == "__main__":
    run(parse_args())
