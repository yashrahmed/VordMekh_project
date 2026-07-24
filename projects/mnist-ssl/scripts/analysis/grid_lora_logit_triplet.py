"""Tune all three final nonlinear LoRA candidates on train, then test."""

from mnist_ssl.ensembles.lora_logit_triplet import parse_args, run


if __name__ == "__main__":
    run(parse_args())
