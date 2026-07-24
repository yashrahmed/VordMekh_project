"""Tune the best cross-family LoRA pair on train, then evaluate test."""

from mnist_ssl.ensembles.lora_logit_pair import parse_args, run


if __name__ == "__main__":
    run(parse_args())
