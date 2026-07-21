"""Run the full I-JEPA/DINOv2 LoRA probe matrix."""

from mnist_ssl.lora_probe import parse_args, run_matrix


if __name__ == "__main__":
    run_matrix(parse_args())
