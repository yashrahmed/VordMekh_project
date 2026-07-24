"""Tune the top-three LoRA triplet and all pairwise ensembles."""

from mnist_ssl.ensembles.lora_top3_ensembles import parse_args, run


if __name__ == "__main__":
    run(parse_args())
