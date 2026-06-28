"""Run the 56x56 upscaled-bbox, 7x7-patch custom I-JEPA split sweep.

Protocol:

* pretrain custom I-JEPA for 50 and 75 epochs;
* sweep n_targets in {8, 16, 24, 32, 36, 40, 44, 48};
* train frozen mean and flatten linear probes for 50 epochs for each encoder;
* evaluate on the MNIST test split with the same preprocessing.

Writes:

* ``out/upscale_bbox_p7_split_sweep_results.csv``
* split-specific probe checkpoints under ``models/``
"""

from __future__ import annotations

import csv
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "out"
MODELS_DIR = ROOT / "models"
RESULTS_CSV = OUT_DIR / "upscale_bbox_p7_split_sweep_results.csv"

PRETRAIN_EPOCHS = (50, 75)
N_PATCHES = 64
N_TARGETS = (8, 16, 24, 32, 36, 40, 44, 48)
POOLS = ("mean", "flatten")
PROBE_EPOCHS = 50

TRAIN_ACC_RE = re.compile(r"Train accuracy:\s+([0-9.]+)%")
TEST_ACC_RE = re.compile(r"Test error:\s+[0-9.]+%\s+\(acc\s+([0-9.]+)%\)")


def run(cmd: list[str]) -> str:
    print("\n$ " + " ".join(cmd), flush=True)
    proc = subprocess.Popen(
        cmd,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    chunks: list[str] = []
    for line in proc.stdout:
        print(line, end="", flush=True)
        chunks.append(line)
    code = proc.wait()
    output = "".join(chunks)
    if code != 0:
        raise subprocess.CalledProcessError(code, cmd, output=output)
    return output


def parse_pct(pattern: re.Pattern[str], text: str, label: str) -> float:
    matches = pattern.findall(text)
    if not matches:
        raise RuntimeError(f"Could not parse {label} from command output")
    return float(matches[-1])


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    rows = read_results()
    done = {
        (int(row["pretrain_epochs"]), str(row["pool"]), int(row["n_targets"]))
        for row in rows
    }
    for pretrain_epochs in PRETRAIN_EPOCHS:
        for n_targets in N_TARGETS:
            n_context = N_PATCHES - n_targets
            run(
                [
                    sys.executable,
                    "-m",
                    "ijepa_trials.custom_ijepa",
                    "--epochs",
                    str(pretrain_epochs),
                    "--n-targets",
                    str(n_targets),
                    "--seed",
                    "0",
                ]
            )

            for pool in POOLS:
                key = (pretrain_epochs, pool, n_targets)
                if key in done:
                    print(f"Skipping recorded result: {pretrain_epochs}ep {pool} {n_targets}-{n_context}")
                    continue
                probe_path = (
                    MODELS_DIR
                    / (
                        "ijepa_clf_custom_ijepa_upscale_bbox_p7"
                        f"_{pool}_t{n_targets}_base{pretrain_epochs}ep_probe{PROBE_EPOCHS}ep.pt"
                    )
                )
                probe_output = run(
                    [
                        sys.executable,
                        "-m",
                        "ijepa_trials.train_probe",
                        "--encoder",
                        "custom_ijepa",
                        "--ckpt-epochs",
                        str(pretrain_epochs),
                        "--n-targets",
                        str(n_targets),
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
                train_acc = parse_pct(TRAIN_ACC_RE, probe_output, "train accuracy")

                eval_output = run(
                    [
                        sys.executable,
                        "-m",
                        "ijepa_trials.eval_probe",
                        "--model",
                        str(probe_path),
                    ]
                )
                test_acc = parse_pct(TEST_ACC_RE, eval_output, "test accuracy")

                row = {
                    "pretrain_epochs": pretrain_epochs,
                    "probe_epochs": PROBE_EPOCHS,
                    "split": f"{n_targets}-{n_context}",
                    "n_targets": n_targets,
                    "n_context": n_context,
                    "pool": pool,
                    "train_acc_pct": train_acc,
                    "test_acc_pct": test_acc,
                    "probe_path": str(probe_path.relative_to(ROOT)),
                }
                rows.append(row)
                done.add(key)
                write_results(rows)
                print(f"\nRecorded result: {row}", flush=True)


def read_results() -> list[dict[str, str | int | float]]:
    if not RESULTS_CSV.exists():
        return []
    with RESULTS_CSV.open(newline="") as f:
        return list(csv.DictReader(f))


def write_results(rows: list[dict[str, str | int | float]]) -> None:
    with RESULTS_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "pretrain_epochs",
                "probe_epochs",
                "split",
                "n_targets",
                "n_context",
                "pool",
                "train_acc_pct",
                "test_acc_pct",
                "probe_path",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Results so far -> {RESULTS_CSV}", flush=True)


if __name__ == "__main__":
    main()
