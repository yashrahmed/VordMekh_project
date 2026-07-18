"""Build a standalone reviewer for the published MNIST label-error audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import webbrowser
from pathlib import Path
from typing import Any, Sequence

from mnist_ssl.paths import DATASET_DIR, OUT_DIR, PROJECT_ROOT


DEFAULT_CANDIDATES = (
    PROJECT_ROOT
    / "configs"
    / "evaluation"
    / "mnist_label_review_candidates.json"
)
DEFAULT_OUTPUT = OUT_DIR / "mnist_label_review.html"
PUBLISHED_CANDIDATE_SET_ID = "northcutt-validated-label-issues"
AUDIT_CATEGORIES = {"correctable", "neither", "non_agreement"}


def project_path(path: Path) -> Path:
    """Resolve a command-line path from the MNIST project root."""

    return path if path.is_absolute() else PROJECT_ROOT / path


def _require_digit(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 9:
        raise ValueError(f"{field} must be an integer digit from 0 through 9")
    return value


def _validate_published_audit(candidate: dict[str, Any]) -> None:
    audit = candidate["published_audit"]
    suggested = _require_digit(
        audit.get("suggested_label"),
        f"candidate {candidate['index']} published_audit.suggested_label",
    )
    if suggested == candidate["original_label"]:
        raise ValueError(
            f"candidate {candidate['index']} has the original label as its suggested label"
        )

    votes = audit.get("votes")
    if not isinstance(votes, dict) or set(votes) != {
        "original",
        "suggested",
        "neither",
    }:
        raise ValueError(
            f"candidate {candidate['index']} published votes must contain "
            "original, suggested, and neither"
        )
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in votes.values()
    ):
        raise ValueError(f"candidate {candidate['index']} has invalid published votes")
    if sum(votes.values()) != 5:
        raise ValueError(f"candidate {candidate['index']} published votes must sum to five")
    if votes["original"] >= 3:
        raise ValueError(
            f"candidate {candidate['index']} does not satisfy the published selection rule"
        )

    category = audit.get("category")
    if category not in AUDIT_CATEGORIES:
        raise ValueError(f"candidate {candidate['index']} has an invalid audit category")
    expected = (
        "correctable"
        if votes["suggested"] >= 3
        else "neither"
        if votes["neither"] >= 3
        else "non_agreement"
    )
    if category != expected:
        raise ValueError(
            f"candidate {candidate['index']} audit category should be {expected!r}"
        )


def load_review_config(path: Path = DEFAULT_CANDIDATES) -> dict[str, Any]:
    """Load and validate the candidate-set provenance and annotations."""

    config = json.loads(path.read_text())
    if config.get("schema_version") != 1:
        raise ValueError(f"unsupported label-review schema in {path}")

    dataset = config.get("dataset")
    if not isinstance(dataset, dict):
        raise ValueError(f"missing dataset metadata in {path}")
    if (
        dataset.get("name") != "MNIST"
        or dataset.get("split") != "test"
        or dataset.get("size") != 10_000
    ):
        raise ValueError(f"{path} must describe the canonical 10,000-example MNIST test set")

    candidate_sets = config.get("candidate_sets")
    if not isinstance(candidate_sets, list) or not candidate_sets:
        raise ValueError(f"missing candidate_sets in {path}")
    set_ids = [item.get("id") for item in candidate_sets]
    if any(not isinstance(set_id, str) or not set_id for set_id in set_ids):
        raise ValueError(f"every candidate set in {path} needs a non-empty id")
    if len(set_ids) != len(set(set_ids)):
        raise ValueError(f"duplicate candidate-set id in {path}")
    if set_ids != [PUBLISHED_CANDIDATE_SET_ID]:
        raise ValueError(
            f"{path} must contain only the published Northcutt et al. MNIST audit"
        )

    candidates = config.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError(f"missing candidates in {path}")
    indexes: list[int] = []
    reason_counts = {set_id: 0 for set_id in set_ids}
    for candidate in candidates:
        if not isinstance(candidate, dict):
            raise ValueError(f"candidate entries in {path} must be objects")
        index = candidate.get("index")
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or not 0 <= index < dataset["size"]
        ):
            raise ValueError(f"candidate index {index!r} is outside the MNIST test split")
        indexes.append(index)
        candidate["original_label"] = _require_digit(
            candidate.get("original_label"), f"candidate {index} original_label"
        )

        reasons = candidate.get("reasons")
        if not isinstance(reasons, list) or not reasons or len(reasons) != len(set(reasons)):
            raise ValueError(f"candidate {index} needs unique candidate-set reasons")
        unknown_reasons = set(reasons) - set(set_ids)
        if unknown_reasons:
            raise ValueError(f"candidate {index} has unknown reasons: {unknown_reasons}")
        for reason in reasons:
            reason_counts[reason] += 1

        if "published_audit" not in candidate:
            raise ValueError(f"candidate {index} is missing its published audit")
        _validate_published_audit(candidate)

    if indexes != sorted(indexes):
        raise ValueError(f"candidates in {path} must be ordered by index")
    if len(indexes) != len(set(indexes)):
        raise ValueError(f"duplicate candidate index in {path}")
    for candidate_set in candidate_sets:
        set_id = candidate_set["id"]
        if candidate_set.get("count") != reason_counts[set_id]:
            raise ValueError(
                f"candidate set {set_id!r} declares {candidate_set.get('count')} "
                f"items but contains {reason_counts[set_id]}"
            )
    return config


def _flatten_pixels(image: Any, index: int) -> list[int]:
    if hasattr(image, "reshape"):
        values = image.reshape(-1)
    else:
        values = [value for row in image for value in row]
    if hasattr(values, "tolist"):
        values = values.tolist()
    pixels = [int(value) for value in values]
    if len(pixels) != 28 * 28:
        raise ValueError(f"MNIST test image {index} does not contain 28x28 pixels")
    if any(not 0 <= value <= 255 for value in pixels):
        raise ValueError(f"MNIST test image {index} contains an invalid pixel value")
    return pixels


def _as_int(value: Any) -> int:
    return int(value.item()) if hasattr(value, "item") else int(value)


def candidate_set_sha256(config: dict[str, Any]) -> str:
    """Return a stable identity for the exact review scope and provenance."""

    canonical = json.dumps(
        {
            "name": config["name"],
            "dataset": config["dataset"],
            "candidate_sets": config["candidate_sets"],
            "candidates": config["candidates"],
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(canonical).hexdigest()


def build_review_payload(
    config: dict[str, Any],
    images: Sequence[Any],
    labels: Sequence[Any],
) -> dict[str, Any]:
    """Attach raw pixels after verifying canonical MNIST labels and ordering."""

    expected_size = config["dataset"]["size"]
    if len(images) != expected_size or len(labels) != expected_size:
        raise ValueError(
            f"expected {expected_size} MNIST test images and labels, "
            f"got {len(images)} images and {len(labels)} labels"
        )

    review_candidates = []
    for candidate in config["candidates"]:
        index = candidate["index"]
        dataset_label = _as_int(labels[index])
        if dataset_label != candidate["original_label"]:
            raise ValueError(
                f"candidate {index} expected original label "
                f"{candidate['original_label']}, found {dataset_label}; "
                "the dataset order does not match the review config"
            )
        review_candidate = dict(candidate)
        review_candidate["pixels"] = _flatten_pixels(images[index], index)
        review_candidates.append(review_candidate)

    return {
        "schema_version": 1,
        "name": config["name"],
        "candidate_set_sha256": candidate_set_sha256(config),
        "dataset": config["dataset"],
        "candidate_sets": config["candidate_sets"],
        "candidates": review_candidates,
    }


def render_review_html(payload: dict[str, Any]) -> str:
    """Render a dependency-free local app with resumable review and JSON export."""

    payload_json = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")
    storage_key = f"mnist-label-review:{payload['candidate_set_sha256']}"
    return (
        _HTML_TEMPLATE.replace("__PAYLOAD__", payload_json)
        .replace("__STORAGE_KEY__", json.dumps(storage_key))
    )


def write_review_html(
    dataset: Any,
    output: Path = DEFAULT_OUTPUT,
    candidates_path: Path = DEFAULT_CANDIDATES,
) -> Path:
    """Build and write the review app from an MNIST-compatible dataset."""

    config = load_review_config(candidates_path)
    payload = build_review_payload(config, dataset.data, dataset.targets)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_review_html(payload))
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--dataset-root", type=Path, default=DATASET_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--download",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="download MNIST when it is not already present",
    )
    parser.add_argument(
        "--open",
        action="store_true",
        help="open the generated reviewer in the default browser",
    )
    return parser.parse_args()


def main() -> None:
    from torchvision import datasets

    args = parse_args()
    candidates_path = project_path(args.candidates)
    dataset_root = project_path(args.dataset_root)
    output = project_path(args.output)
    dataset = datasets.MNIST(
        root=str(dataset_root),
        train=False,
        download=args.download,
    )
    written = write_review_html(dataset, output, candidates_path)
    count = len(load_review_config(candidates_path)["candidates"])
    print(f"wrote {written} with {count} review candidates")
    if args.open:
        webbrowser.open(written.resolve().as_uri())


_HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>MNIST label review</title>
  <style>
    :root {
      color-scheme: light dark;
      --bg: #f4f1e8;
      --surface: #fffdf7;
      --surface-2: #ece7dc;
      --ink: #1c201e;
      --muted: #626a65;
      --border: #c9c2b4;
      --accent: #245d4a;
      --accent-ink: #ffffff;
      --keep: #2d6a4f;
      --relabel: #9a6700;
      --exclude: #9b2c2c;
      --focus: #2979ff;
      --shadow: 0 14px 38px rgba(30, 35, 32, 0.12);
    }
    @media (prefers-color-scheme: dark) {
      :root {
        --bg: #171a18;
        --surface: #202421;
        --surface-2: #2a2f2b;
        --ink: #f3f1e9;
        --muted: #b9c0ba;
        --border: #4a514c;
        --accent: #8bc9ac;
        --accent-ink: #10261c;
        --keep: #86c7a5;
        --relabel: #e4b95f;
        --exclude: #ec8b8b;
        --focus: #8ab4ff;
        --shadow: none;
      }
    }
    * { box-sizing: border-box; }
    .sr-only {
      position: absolute;
      width: 1px;
      height: 1px;
      padding: 0;
      margin: -1px;
      overflow: hidden;
      clip: rect(0, 0, 0, 0);
      white-space: nowrap;
      border: 0;
    }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
      line-height: 1.45;
    }
    button, input, select, textarea { font: inherit; }
    button, select, textarea, .file-label {
      border: 1px solid var(--border);
      border-radius: 8px;
      background: var(--surface);
      color: var(--ink);
    }
    button, .file-label {
      cursor: pointer;
      padding: 0.65rem 0.9rem;
    }
    button:hover, .file-label:hover { background: var(--surface-2); }
    button:focus-visible, input:focus-visible, select:focus-visible, textarea:focus-visible,
    .file-label:focus-within {
      outline: 3px solid var(--focus);
      outline-offset: 2px;
    }
    button.primary {
      border-color: var(--accent);
      background: var(--accent);
      color: var(--accent-ink);
      font-weight: 700;
    }
    button.primary:hover { filter: brightness(1.05); }
    button:disabled { cursor: not-allowed; opacity: 0.5; }
    .shell {
      width: min(1180px, calc(100% - 2rem));
      margin: 0 auto;
      padding: 1.5rem 0 3rem;
    }
    header {
      display: flex;
      flex-wrap: wrap;
      align-items: end;
      justify-content: space-between;
      gap: 1rem;
      margin-bottom: 1rem;
    }
    h1 { margin: 0; font-size: clamp(1.4rem, 4vw, 2.1rem); }
    header p { margin: 0.35rem 0 0; color: var(--muted); }
    .progress { min-width: 15rem; text-align: right; }
    .progress strong { display: block; font-size: 1.2rem; }
    .progress-track {
      height: 8px;
      margin-top: 0.4rem;
      overflow: hidden;
      border-radius: 999px;
      background: var(--surface-2);
    }
    .progress-bar { height: 100%; width: 0; background: var(--accent); }
    .toolbar {
      display: flex;
      flex-wrap: wrap;
      gap: 0.65rem;
      align-items: center;
      margin-bottom: 1rem;
    }
    .toolbar label { color: var(--muted); }
    select { padding: 0.65rem 2.2rem 0.65rem 0.7rem; }
    .file-label input { position: absolute; width: 1px; height: 1px; opacity: 0; }
    .workspace {
      display: grid;
      grid-template-columns: minmax(300px, 1fr) minmax(320px, 0.9fr);
      gap: 1rem;
      align-items: start;
    }
    .panel {
      border: 1px solid var(--border);
      border-radius: 14px;
      background: var(--surface);
      box-shadow: var(--shadow);
    }
    .image-panel { padding: 1rem; }
    .image-heading {
      display: flex;
      flex-wrap: wrap;
      align-items: baseline;
      justify-content: space-between;
      gap: 0.75rem;
      margin-bottom: 0.75rem;
    }
    .image-heading h2 { margin: 0; font-size: 1.15rem; }
    .original-label { color: var(--muted); }
    .digit-stage {
      display: grid;
      min-height: 460px;
      place-items: center;
      border-radius: 10px;
      background: #000;
    }
    #digit-canvas {
      width: min(430px, 100%);
      height: auto;
      aspect-ratio: 1;
      image-rendering: pixelated;
      image-rendering: crisp-edges;
    }
    .review-panel { padding: 1rem; }
    .review-panel h2 { margin: 0 0 0.85rem; font-size: 1.15rem; }
    fieldset { margin: 0; padding: 0; border: 0; }
    .decision-option {
      display: grid;
      grid-template-columns: auto 1fr;
      gap: 0.7rem;
      align-items: start;
      padding: 0.8rem 0;
      border-top: 1px solid var(--border);
    }
    .decision-option:first-of-type { border-top: 0; }
    .decision-option input { margin-top: 0.25rem; }
    .decision-option strong, .decision-option span { display: block; }
    .decision-option span { color: var(--muted); font-size: 0.9rem; }
    .decision-option .relabel-row {
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      gap: 0.5rem;
      margin-top: 0.55rem;
    }
    .decision-option .relabel-row span { display: inline; }
    .field { margin-top: 1rem; }
    .field label { display: block; margin-bottom: 0.35rem; color: var(--muted); }
    textarea { width: 100%; min-height: 5rem; padding: 0.7rem; resize: vertical; }
    .validation { min-height: 1.5rem; margin: 0.5rem 0 0; color: var(--exclude); }
    .nav {
      display: flex;
      justify-content: space-between;
      gap: 0.65rem;
      margin-top: 1rem;
    }
    details {
      margin-top: 1rem;
      border-top: 1px solid var(--border);
      padding-top: 0.85rem;
    }
    summary { cursor: pointer; color: var(--muted); }
    .context-list {
      display: grid;
      grid-template-columns: max-content 1fr;
      gap: 0.35rem 0.9rem;
      margin-bottom: 0;
    }
    .context-list dt { color: var(--muted); }
    .context-list dd { margin: 0; }
    .candidate-section { margin-top: 1rem; }
    .candidate-section h2 { margin: 0 0 0.65rem; font-size: 1.05rem; }
    .candidate-grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(76px, 1fr));
      gap: 0.55rem;
    }
    .candidate-tile {
      display: grid;
      place-items: center;
      gap: 0.35rem;
      padding: 0.55rem 0.3rem;
      border: 2px solid var(--border);
      background: var(--surface);
    }
    .candidate-tile[aria-current="true"] { outline: 3px solid var(--focus); outline-offset: 1px; }
    .candidate-tile[data-status="keep"] { border-color: var(--keep); }
    .candidate-tile[data-status="relabel"] { border-color: var(--relabel); }
    .candidate-tile[data-status="exclude"] { border-color: var(--exclude); }
    .candidate-tile canvas {
      width: 56px;
      height: 56px;
      background: #000;
      image-rendering: pixelated;
    }
    .legend {
      display: flex;
      flex-wrap: wrap;
      gap: 1rem;
      margin-top: 0.7rem;
      color: var(--muted);
      font-size: 0.9rem;
    }
    .legend span::before {
      content: "";
      display: inline-block;
      width: 0.75rem;
      height: 0.75rem;
      margin-right: 0.35rem;
      border: 2px solid var(--border);
      border-radius: 3px;
      vertical-align: -0.1rem;
    }
    .legend .keep::before { border-color: var(--keep); }
    .legend .relabel::before { border-color: var(--relabel); }
    .legend .exclude::before { border-color: var(--exclude); }
    .save-status { margin-left: auto; color: var(--muted); font-size: 0.9rem; }
    @media (max-width: 760px) {
      .workspace { grid-template-columns: 1fr; }
      .digit-stage { min-height: 320px; }
      .progress { width: 100%; text-align: left; }
      .save-status { width: 100%; margin-left: 0; }
    }
  </style>
</head>
<body>
  <main class="shell">
    <header>
      <div>
        <h1>MNIST label review</h1>
        <p>Review the 15 MNIST label issues validated in the published human audit.</p>
      </div>
      <div class="progress" aria-live="polite">
        <strong id="progress-text">0 of 0 reviewed</strong>
        <span id="scope-text"></span>
        <div class="progress-track" aria-hidden="true"><div class="progress-bar" id="progress-bar"></div></div>
      </div>
    </header>

    <div class="toolbar">
      <button type="button" id="export-button" class="primary">Download decisions</button>
      <label class="file-label">Import decisions<input type="file" id="import-input" accept="application/json"></label>
      <button type="button" id="clear-button">Clear current</button>
      <span class="save-status" id="save-status">Draft stored locally</span>
    </div>

    <div class="workspace">
      <section class="panel image-panel" aria-labelledby="image-title">
        <div class="image-heading">
          <h2 id="image-title">Test index</h2>
          <span class="original-label" id="original-label"></span>
        </div>
        <div class="digit-stage">
          <canvas id="digit-canvas" width="28" height="28" role="img" aria-label="MNIST digit under review"></canvas>
        </div>
      </section>

      <section class="panel review-panel" aria-labelledby="decision-title">
        <h2 id="decision-title">Your decision</h2>
        <fieldset>
          <legend class="sr-only">Label decision</legend>
          <label class="decision-option">
            <input type="radio" name="decision" value="keep" id="decision-keep">
            <span><strong>Keep original label</strong><span>The supplied MNIST label is acceptable.</span></span>
          </label>
          <label class="decision-option">
            <input type="radio" name="decision" value="relabel" id="decision-relabel">
            <span>
              <strong>Relabel</strong>
              <span>Assign a different digit.</span>
              <span class="relabel-row">
                <span id="corrected-label-label">Correct digit</span>
                <select id="corrected-label" aria-labelledby="corrected-label-label">
                  <option value="">Select</option>
                  <option value="0">0</option><option value="1">1</option>
                  <option value="2">2</option><option value="3">3</option>
                  <option value="4">4</option><option value="5">5</option>
                  <option value="6">6</option><option value="7">7</option>
                  <option value="8">8</option><option value="9">9</option>
                </select>
              </span>
            </span>
          </label>
          <label class="decision-option">
            <input type="radio" name="decision" value="exclude" id="decision-exclude">
            <span><strong>Exclude as ambiguous</strong><span>No single digit label is defensible.</span></span>
          </label>
        </fieldset>
        <p class="validation" id="validation-message" aria-live="polite"></p>
        <div class="field">
          <label for="review-note">Optional note</label>
          <textarea id="review-note" placeholder="Why did you make this decision?"></textarea>
        </div>
        <div class="nav">
          <button type="button" id="previous-button">Previous</button>
          <button type="button" id="next-button" class="primary">Next</button>
        </div>
        <details>
          <summary>Reveal published-audit context</summary>
          <dl class="context-list" id="context-list"></dl>
        </details>
      </section>
    </div>

    <section class="candidate-section" aria-labelledby="candidate-title">
      <h2 id="candidate-title">Review queue</h2>
      <div class="candidate-grid" id="candidate-grid"></div>
      <div class="legend" aria-label="Decision status legend">
        <span>Unreviewed</span>
        <span class="keep">Keep</span>
        <span class="relabel">Relabel</span>
        <span class="exclude">Exclude</span>
      </div>
    </section>
  </main>

  <script>
    "use strict";
    const payload = __PAYLOAD__;
    const storageKey = __STORAGE_KEY__;
    const byIndex = new Map(payload.candidates.map(candidate => [candidate.index, candidate]));
    let currentIndex = payload.candidates[0].index;
    let decisions = Object.fromEntries(
      payload.candidates.map(candidate => [
        String(candidate.index),
        {action: "unreviewed", corrected_label: null, note: ""}
      ])
    );

    const elements = {
      progressText: document.getElementById("progress-text"),
      progressBar: document.getElementById("progress-bar"),
      scopeText: document.getElementById("scope-text"),
      exportButton: document.getElementById("export-button"),
      importInput: document.getElementById("import-input"),
      clearButton: document.getElementById("clear-button"),
      saveStatus: document.getElementById("save-status"),
      imageTitle: document.getElementById("image-title"),
      originalLabel: document.getElementById("original-label"),
      digitCanvas: document.getElementById("digit-canvas"),
      decisionRadios: [...document.querySelectorAll('input[name="decision"]')],
      correctedLabel: document.getElementById("corrected-label"),
      validationMessage: document.getElementById("validation-message"),
      reviewNote: document.getElementById("review-note"),
      previousButton: document.getElementById("previous-button"),
      nextButton: document.getElementById("next-button"),
      contextList: document.getElementById("context-list"),
      candidateGrid: document.getElementById("candidate-grid")
    };

    function visibleCandidates() {
      return payload.candidates;
    }

    function decisionFor(index) {
      return decisions[String(index)];
    }

    function isComplete(candidate, decision) {
      if (decision.action === "keep" || decision.action === "exclude") return true;
      return decision.action === "relabel"
        && Number.isInteger(decision.corrected_label)
        && decision.corrected_label !== candidate.original_label;
    }

    function decisionStatus(candidate) {
      const decision = decisionFor(candidate.index);
      return isComplete(candidate, decision) ? decision.action : "unreviewed";
    }

    function saveDraft() {
      try {
        localStorage.setItem(storageKey, JSON.stringify(decisions));
        elements.saveStatus.textContent = "Draft stored locally";
      } catch (error) {
        elements.saveStatus.textContent = "Local draft unavailable; download decisions to save";
      }
    }

    function loadDraft() {
      try {
        const saved = localStorage.getItem(storageKey);
        if (!saved) return;
        const parsed = JSON.parse(saved);
        for (const candidate of payload.candidates) {
          const decision = parsed[String(candidate.index)];
          if (decision) decisions[String(candidate.index)] = normalizeDecision(decision);
        }
      } catch (error) {
        elements.saveStatus.textContent = "Stored draft could not be loaded";
      }
    }

    function normalizeDecision(decision) {
      const action = ["keep", "relabel", "exclude"].includes(decision.action)
        ? decision.action
        : "unreviewed";
      const corrected = Number.isInteger(decision.corrected_label)
        && decision.corrected_label >= 0
        && decision.corrected_label <= 9
        ? decision.corrected_label
        : null;
      return {
        action,
        corrected_label: corrected,
        note: typeof decision.note === "string" ? decision.note : ""
      };
    }

    function drawDigit(canvas, pixels) {
      const context = canvas.getContext("2d");
      const image = context.createImageData(28, 28);
      for (let index = 0; index < pixels.length; index += 1) {
        const offset = index * 4;
        image.data[offset] = pixels[index];
        image.data[offset + 1] = pixels[index];
        image.data[offset + 2] = pixels[index];
        image.data[offset + 3] = 255;
      }
      context.putImageData(image, 0, 0);
    }

    function addContextRow(label, value) {
      const term = document.createElement("dt");
      const detail = document.createElement("dd");
      term.textContent = label;
      detail.textContent = value;
      elements.contextList.append(term, detail);
    }

    function renderContext(candidate) {
      elements.contextList.replaceChildren();
      const audit = candidate.published_audit;
      addContextRow("Published suggestion", String(audit.suggested_label));
      addContextRow(
        "Published votes",
        `original ${audit.votes.original}; suggested ${audit.votes.suggested}; `
        + `neither ${audit.votes.neither}`
      );
      addContextRow("Published category", audit.category);
    }

    function validateCurrent() {
      const candidate = byIndex.get(currentIndex);
      const decision = decisionFor(currentIndex);
      let message = "";
      if (decision.action === "relabel" && decision.corrected_label === null) {
        message = "Choose a corrected digit to complete this review.";
      } else if (
        decision.action === "relabel"
        && decision.corrected_label === candidate.original_label
      ) {
        message = "A corrected label must differ from the original; choose Keep instead.";
      }
      elements.validationMessage.textContent = message;
    }

    function renderCurrent() {
      const candidate = byIndex.get(currentIndex);
      const decision = decisionFor(currentIndex);
      elements.imageTitle.textContent = `Test index ${candidate.index}`;
      elements.originalLabel.textContent = `Original label: ${candidate.original_label}`;
      elements.digitCanvas.setAttribute(
        "aria-label",
        `MNIST test example ${candidate.index}, original label ${candidate.original_label}`
      );
      drawDigit(elements.digitCanvas, candidate.pixels);
      for (const radio of elements.decisionRadios) {
        radio.checked = radio.value === decision.action;
      }
      elements.correctedLabel.disabled = decision.action !== "relabel";
      elements.correctedLabel.value = decision.corrected_label === null
        ? ""
        : String(decision.corrected_label);
      elements.reviewNote.value = decision.note;
      renderContext(candidate);
      validateCurrent();

      const visible = visibleCandidates();
      const position = visible.findIndex(item => item.index === currentIndex);
      elements.previousButton.disabled = position <= 0;
      elements.nextButton.disabled = position < 0 || position >= visible.length - 1;
    }

    function renderProgress() {
      const reviewed = payload.candidates.filter(candidate =>
        isComplete(candidate, decisionFor(candidate.index))
      ).length;
      elements.progressText.textContent = `${reviewed} of ${payload.candidates.length} reviewed`;
      elements.progressBar.style.width = `${100 * reviewed / payload.candidates.length}%`;
      elements.scopeText.textContent = `${payload.candidates.length} paper-validated examples`;
    }

    function renderGrid() {
      elements.candidateGrid.replaceChildren();
      for (const candidate of visibleCandidates()) {
        const button = document.createElement("button");
        button.type = "button";
        button.className = "candidate-tile";
        button.dataset.status = decisionStatus(candidate);
        button.setAttribute("aria-current", String(candidate.index === currentIndex));
        button.setAttribute(
          "aria-label",
          `Review test index ${candidate.index}; original label ${candidate.original_label}; `
          + `status ${decisionStatus(candidate)}`
        );
        const canvas = document.createElement("canvas");
        canvas.width = 28;
        canvas.height = 28;
        canvas.setAttribute("aria-hidden", "true");
        drawDigit(canvas, candidate.pixels);
        const identifier = document.createElement("span");
        identifier.textContent = `#${candidate.index}`;
        button.append(canvas, identifier);
        button.addEventListener("click", () => {
          currentIndex = candidate.index;
          render();
        });
        elements.candidateGrid.append(button);
      }
    }

    function render() {
      renderCurrent();
      renderProgress();
      renderGrid();
    }

    function updateDecision(action) {
      const decision = decisionFor(currentIndex);
      decision.action = action;
      if (action !== "relabel") decision.corrected_label = null;
      saveDraft();
      render();
    }

    function move(offset) {
      const visible = visibleCandidates();
      const position = visible.findIndex(candidate => candidate.index === currentIndex);
      const target = visible[position + offset];
      if (target) {
        currentIndex = target.index;
        render();
      }
    }

    function exportDecisions() {
      const rows = payload.candidates.map(candidate => {
        const decision = decisionFor(candidate.index);
        return {
          index: candidate.index,
          original_label: candidate.original_label,
          action: isComplete(candidate, decision) ? decision.action : "unreviewed",
          corrected_label: decision.action === "relabel" ? decision.corrected_label : null,
          note: decision.note
        };
      });
      const completeCount = rows.filter(row => row.action !== "unreviewed").length;
      const exported = {
        schema_version: 1,
        candidate_set_name: payload.name,
        candidate_set_sha256: payload.candidate_set_sha256,
        dataset: payload.dataset,
        exported_at: new Date().toISOString(),
        review_complete: completeCount === rows.length,
        reviewed_count: completeCount,
        candidate_count: rows.length,
        decisions: rows
      };
      const blob = new Blob([JSON.stringify(exported, null, 2) + "\n"], {
        type: "application/json"
      });
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = "mnist-label-review-decisions.json";
      link.click();
      window.setTimeout(() => URL.revokeObjectURL(url), 1000);
    }

    async function importDecisions(file) {
      const imported = JSON.parse(await file.text());
      if (imported.schema_version !== 1) throw new Error("unsupported decision schema");
      if (imported.candidate_set_sha256 !== payload.candidate_set_sha256) {
        throw new Error("the decision file belongs to a different candidate set");
      }
      if (!Array.isArray(imported.decisions)) throw new Error("missing decisions array");
      const importedByIndex = new Map(imported.decisions.map(row => [row.index, row]));
      for (const candidate of payload.candidates) {
        const row = importedByIndex.get(candidate.index);
        if (!row) continue;
        if (row.original_label !== candidate.original_label) {
          throw new Error(`original label mismatch at index ${candidate.index}`);
        }
        decisions[String(candidate.index)] = normalizeDecision({
          action: row.action,
          corrected_label: row.corrected_label,
          note: row.note
        });
      }
      saveDraft();
      render();
    }

    for (const radio of elements.decisionRadios) {
      radio.addEventListener("change", event => updateDecision(event.target.value));
    }
    elements.correctedLabel.addEventListener("change", event => {
      const decision = decisionFor(currentIndex);
      decision.action = "relabel";
      decision.corrected_label = event.target.value === "" ? null : Number(event.target.value);
      saveDraft();
      render();
    });
    elements.reviewNote.addEventListener("input", event => {
      decisionFor(currentIndex).note = event.target.value;
      saveDraft();
    });
    elements.previousButton.addEventListener("click", () => move(-1));
    elements.nextButton.addEventListener("click", () => move(1));
    elements.clearButton.addEventListener("click", () => {
      decisions[String(currentIndex)] = {
        action: "unreviewed",
        corrected_label: null,
        note: ""
      };
      saveDraft();
      render();
    });
    elements.exportButton.addEventListener("click", exportDecisions);
    elements.importInput.addEventListener("change", async event => {
      const [file] = event.target.files;
      if (!file) return;
      try {
        await importDecisions(file);
        elements.saveStatus.textContent = "Decisions imported and stored locally";
      } catch (error) {
        elements.saveStatus.textContent = `Import failed: ${error.message}`;
      } finally {
        event.target.value = "";
      }
    });

    loadDraft();
    render();
  </script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
