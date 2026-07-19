"""Verify manifest-pinned checkpoint artifacts used by canonical results."""

from __future__ import annotations

import argparse

from mnist_ssl.provenance import artifact_index, sha256_file, verify_artifacts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "artifact_ids",
        nargs="*",
        help="artifact IDs to verify (default: every manifest artifact)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="verify every artifact in the checkpoint manifest",
    )
    args = parser.parse_args()

    index = artifact_index()
    if args.all and args.artifact_ids:
        parser.error("--all cannot be combined with explicit artifact IDs")
    artifact_ids = args.artifact_ids
    if args.all:
        artifact_ids = list(index)
    elif not artifact_ids:
        artifact_ids = list(index)
    paths = verify_artifacts(artifact_ids)
    for artifact_id, path in paths.items():
        print(f"verified {artifact_id}: {sha256_file(path)}  {path}")


if __name__ == "__main__":
    main()
