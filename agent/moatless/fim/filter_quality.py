"""
Filter multi-hunk dataset: keep only tasks where every hunk has substantive code changes.

Excludes hunks that are:
  - Pure deletion (no + lines)
  - Only blank lines, comments, or syntax characters ({, }, ;, ())

Usage:
  python -m moatless.fim.filter_quality
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

INPUT_DIR = Path(r"D:\Data\2025\CodeCompletion\Dataset\Outputs\re_collected_phase1&2\collected_0418")
OUTPUT_DIR = Path(r"D:\Data\2025\CodeCompletion\Dataset\Outputs\re_collected_phase1&2")


def extract_plus_lines(content: str) -> list[str]:
    return [line[1:] for line in content.splitlines() if line.startswith("+")]


def is_substantive_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if stripped.startswith("#") or stripped.startswith("//"):
        return False
    if stripped in ("{", "}", ";", "()"):
        return False
    return True


def has_substantive_content(content: str) -> bool:
    plus_lines = extract_plus_lines(content)
    if not plus_lines:
        return False
    return any(is_substantive_line(l) for l in plus_lines)


def filter_task(row: dict) -> tuple[bool, list[str]]:
    """Check if a task passes quality filter.

    Returns (passes, reasons) where reasons describe why it was filtered.
    """
    oh = row.get("ordered_hunks", [])
    if not oh:
        return False, ["no_hunks"]

    bad_indices: list[int] = []
    for i, h in enumerate(oh):
        if not has_substantive_content(h.get("content", "")):
            bad_indices.append(i)

    if bad_indices:
        reasons = []
        for idx in bad_indices:
            plus = extract_plus_lines(oh[idx].get("content", ""))
            if not plus:
                reasons.append(f"h{idx}=pure_deletion")
            else:
                reasons.append(f"h{idx}=trivial_only")
        return False, reasons

    return True, []


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_inputs = sorted(INPUT_DIR.glob("*.jsonl"))

    total_input = 0
    total_output = 0
    total_filtered = 0

    for input_path in all_inputs:
        repo = input_path.stem.split("_")[0]
        output_path = OUTPUT_DIR / input_path.name

        line_in = 0
        line_out = 0
        line_fil = 0

        with input_path.open(encoding="utf-8") as fin, output_path.open("w", encoding="utf-8") as fout:
            for raw in fin:
                line = raw.strip()
                if not line:
                    continue
                row = json.loads(line)
                line_in += 1

                ok, reasons = filter_task(row)
                if ok:
                    fout.write(raw if raw.endswith("\n") else raw + "\n")
                    line_out += 1
                else:
                    line_fil += 1

        total_input += line_in
        total_output += line_out
        total_filtered += line_fil
        pct = line_out / line_in * 100 if line_in else 0
        logger.info(f"{repo:20s}  in={line_in:>5}  out={line_out:>5}  filtered={line_fil:>5}  ({pct:.1f}%)")

    logger.info(f"{'TOTAL':20s}  in={total_input:>5}  out={total_output:>5}  filtered={total_filtered:>5}  ({total_output/total_input*100:.1f}%)")


if __name__ == "__main__":
    main()
