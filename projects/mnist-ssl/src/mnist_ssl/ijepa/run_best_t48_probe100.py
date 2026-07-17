"""Train 100-epoch probes for the best 56x56 t48 checkpoints."""

from __future__ import annotations

import csv
import re
import subprocess
import sys
from pathlib import Path


from mnist_ssl.paths import MODELS_DIR, OUT_DIR, PROJECT_ROOT as ROOT
N_TARGETS = 48
N_PATCHES = 64
PROBE_EPOCHS = 100
BASE_EPOCHS = (300, 500)
POOLS = ("mean", "flatten")

TRAIN_ACC_RE = re.compile(r"Train accuracy:\s+([0-9.]+)%")
TEST_ACC_RE = re.compile(r"(?:Test accuracy:\s+|acc\s)([0-9.]+)%")


def run(cmd: list[str]) -> str:
    print("$", " ".join(cmd), flush=True)
    proc = subprocess.run(
        cmd,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    print(proc.stdout, end="", flush=True)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)
    return proc.stdout


def parse_pct(pattern: re.Pattern[str], text: str, label: str) -> float:
    match = pattern.search(text)
    if match is None:
        raise RuntimeError(f"Could not parse {label} from command output")
    return float(match.group(1))


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for base_epochs in BASE_EPOCHS:
        for pool in POOLS:
            n_context = N_PATCHES - N_TARGETS
            probe_path = (
                MODELS_DIR
                / (
                    "ijepa_clf_custom_ijepa_upscale_bbox_p7"
                    f"_{pool}_t{N_TARGETS}_base{base_epochs}ep_probe{PROBE_EPOCHS}ep.pt"
                )
            )
            probe_output = run(
                [
                    sys.executable,
                    "-m",
                    "mnist_ssl.ijepa.train_probe",
                    "--encoder",
                    "custom_ijepa",
                    "--ckpt-epochs",
                    str(base_epochs),
                    "--n-targets",
                    str(N_TARGETS),
                    "--epochs",
                    str(PROBE_EPOCHS),
                    "--pool",
                    pool,
                    "--seed",
                    "0",
                    "--out",
                    str(probe_path),
                ]
            )
            eval_output = run(
                [
                    sys.executable,
                    "-m",
                    "mnist_ssl.ijepa.eval_probe",
                    "--model",
                    str(probe_path),
                ]
            )
            rows.append(
                {
                    "pretrain_epochs": base_epochs,
                    "probe_epochs": PROBE_EPOCHS,
                    "n_targets": N_TARGETS,
                    "n_context": n_context,
                    "pool": pool,
                    "train_acc": parse_pct(TRAIN_ACC_RE, probe_output, "train accuracy"),
                    "test_acc": parse_pct(TEST_ACC_RE, eval_output, "test accuracy"),
                    "probe_path": str(probe_path.relative_to(ROOT)),
                }
            )

    out_csv = OUT_DIR / "upscale_bbox_p7_best_t48_probe100_results.csv"
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "pretrain_epochs",
                "probe_epochs",
                "n_targets",
                "n_context",
                "pool",
                "train_acc",
                "test_acc",
                "probe_path",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote results -> {out_csv}", flush=True)


if __name__ == "__main__":
    main()
