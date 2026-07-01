"""
fim/reward.py
Multi-hunk reward computation for agent RL.

Dual-channel design:
  Strict — exact match by (file_path, start_line, end_line) for file application
  Relaxed — Gaussian-decayed location × content similarity for reward signal
"""
from __future__ import annotations

import logging
import math
from typing import Any

from moatless.fim.schema import FillResult, HunkEvalResult
from moatless.fim.evaluator import edit_similarity, exact_match

logger = logging.getLogger(__name__)

# ── Configurable parameters ─────────────────────────────────────────────────
RELAXED_SIGMA = 5.0          # Location decay: distance in lines (σ in Gaussian)
RELAXED_THRESHOLD = 0.05     # Minimum relaxed_es to count as a match
REWARD_LAMBDA = 0.6          # Strict channel weight; relaxed weight = 1 - λ

ALL_EM_BONUS = 0.30          # When strict finds exact match on ALL targets
MISSING_PENALTY_PER = 0.15   # Per missing target (normalised by total targets)
EXTRA_PENALTY_PER = 0.10     # Per extra block (normalised by total predicted)
STEP_PENALTY_PER = 0.005     # Per action step after free steps
STEP_FREE_STEPS = 3          # First N steps incur no penalty


def _build_gt_map(target_hunks: list[dict]) -> dict[int, dict]:
    """Build index -> target hunk lookup."""
    gt_map: dict[int, dict] = {}
    for hunk in target_hunks:
        idx = hunk.get("target_index")
        if idx is not None:
            gt_map[int(idx)] = hunk
    return gt_map


def _build_pred_map(parsed_blocks: list[dict]) -> dict[int, dict]:
    """Build index -> parsed block lookup (last wins on duplicate index)."""
    pred_map: dict[int, dict] = {}
    for block in parsed_blocks:
        if block.get("header_valid") and "index" in block:
            pred_map[int(block["index"])] = block
    return pred_map


def _compute_relaxed_matches(
    target_hunks: list[dict],
    parsed_blocks: list[dict],
    strict_applied_set: set[int],
    sigma: float = RELAXED_SIGMA,
    threshold: float = RELAXED_THRESHOLD,
) -> tuple[dict[int, float], set[int]]:
    """
    For GT targets not in *strict_applied_set*, find the best content +
    location match among remaining parsed blocks.

    Returns
    -------
    relaxed_es_per_target
        ``{target_index: relaxed_es, ...}``
    used_block_indices
        Set of parsed-block indices consumed by relaxed matches.
    """
    # ── which targets still need a match ──
    unmatched_targets = [
        h for h in target_hunks
        if h.get("target_index") is not None
        and int(h["target_index"]) not in strict_applied_set
    ]
    # ── which blocks are free for relaxed matching ──
    available_blocks = [
        b for b in parsed_blocks
        if b.get("header_valid") and b.get("index") is not None
        and int(b["index"]) not in strict_applied_set
    ]

    if not unmatched_targets or not available_blocks:
        return {}, set()

    # ── build all candidate (target, block, score) triples ──
    candidates: list[tuple[int, int, float]] = []

    for target in unmatched_targets:
        gt_idx = int(target["target_index"])
        gt_file = str(target.get("file_path", ""))
        gt_start = int(target.get("start_line", 0))
        gt_end = int(target.get("end_line", 0))
        gt_center = (gt_start + gt_end) / 2.0
        gt_completion = str(target.get("completion", ""))

        for block in available_blocks:
            b_idx = int(block["index"])
            b_file = str(block.get("file_path", ""))
            b_start = int(block.get("start_line", 0))
            b_end = int(block.get("end_line", 0))
            b_body = str(block.get("body", ""))

            # Must be in the same file
            if b_file != gt_file:
                continue

            # Location score: Gaussian decay
            b_center = (b_start + b_end) / 2.0
            line_dist = abs(b_center - gt_center)
            loc_score = math.exp(-(line_dist ** 2) / (2 * sigma ** 2))

            # Content score
            content_score = edit_similarity(b_body, gt_completion) if gt_completion else 0.0

            combined = loc_score * content_score
            if combined > threshold:
                candidates.append((gt_idx, b_idx, combined))

    # ── greedy assignment: highest score first ──
    candidates.sort(key=lambda x: -x[2])
    relaxed_es: dict[int, float] = {}
    used_blocks: set[int] = set()
    matched_targets: set[int] = set()

    for gt_idx, b_idx, combined in candidates:
        if gt_idx in matched_targets or b_idx in used_blocks:
            continue
        relaxed_es[gt_idx] = combined
        matched_targets.add(gt_idx)
        used_blocks.add(b_idx)

    return relaxed_es, used_blocks


def compute_multi_hunk_reward(
    *,
    target_hunks: list[dict],
    parsed_blocks: list[dict],
    applied_indices: list[int],
    missing_indices: list[int],
    extra_predicted_indices: list[int],
    action_steps: int = 0,
) -> tuple[float, list[HunkEvalResult]]:
    """
    Compute per-hunk evaluation results and overall scalar reward.

    Returns (reward, per_hunk_eval).
    """
    # ── 1. Strict path (unchanged) ────────────────────────────────────────
    gt_map = _build_gt_map(target_hunks)
    pred_map = _build_pred_map(parsed_blocks)

    all_indices = sorted({
        int(h["target_index"]) for h in target_hunks
        if h.get("target_index") is not None
    })
    missing_set = set(missing_indices)
    extra_set = set(extra_predicted_indices)
    applied_set = set(applied_indices)

    per_hunk: list[HunkEvalResult] = []
    strict_edit_sims: list[float] = [0.0] * len(all_indices)
    strict_em_flags: list[bool] = [False] * len(all_indices)

    for i, idx in enumerate(all_indices):
        gt = gt_map.get(idx)
        gt_completion = str(gt.get("completion", "")) if gt else ""
        gt_file = str(gt.get("file_path", "")) if gt else ""
        gt_start = int(gt.get("start_line", 0)) if gt else 0
        gt_end = int(gt.get("end_line", 0)) if gt else 0

        is_missing = idx in missing_set

        if is_missing:
            per_hunk.append(HunkEvalResult(
                index=idx,
                file_path=gt_file, start_line=gt_start, end_line=gt_end,
                predicted="", ground_truth=gt_completion,
                exact_match=False, edit_similarity=0.0,
                is_missing=True, is_extra=False, header_compliant=False,
            ))
            strict_em_flags[i] = False
            strict_edit_sims[i] = 0.0
            continue

        block = pred_map.get(idx)
        if block is None:
            per_hunk.append(HunkEvalResult(
                index=idx,
                file_path=gt_file, start_line=gt_start, end_line=gt_end,
                predicted="", ground_truth=gt_completion,
                exact_match=False, edit_similarity=0.0,
                is_missing=True, is_extra=False, header_compliant=False,
            ))
            missing_set.add(idx)
            strict_em_flags[i] = False
            strict_edit_sims[i] = 0.0
            continue

        predicted_code = str(block.get("body") or "")
        pred_file = str(block.get("file_path") or "")
        pred_start = int(block.get("start_line", 0))
        pred_end = int(block.get("end_line", 0))
        header_ok = (pred_file == gt_file and pred_start == gt_start and pred_end == gt_end)

        if idx in applied_set and header_ok:
            em = exact_match(predicted_code, gt_completion) if gt_completion else False
            es = edit_similarity(predicted_code, gt_completion) if gt_completion else 0.0
        else:
            em = False
            es = 0.0

        strict_edit_sims[i] = es
        strict_em_flags[i] = em

        per_hunk.append(HunkEvalResult(
            index=idx, file_path=gt_file, start_line=gt_start, end_line=gt_end,
            predicted=predicted_code, ground_truth=gt_completion,
            exact_match=em, edit_similarity=es,
            is_missing=False, is_extra=False, header_compliant=header_ok,
        ))

    for idx in sorted(extra_set):
        block = pred_map.get(idx)
        body = str(block.get("body") or "") if block else ""
        pred_file = str(block.get("file_path") or "") if block else ""
        pred_start = int(block.get("start_line", 0)) if block else 0
        pred_end = int(block.get("end_line", 0)) if block else 0
        per_hunk.append(HunkEvalResult(
            index=idx, file_path=pred_file, start_line=pred_start, end_line=pred_end,
            predicted=body, ground_truth="",
            exact_match=False, edit_similarity=0.0,
            is_missing=False, is_extra=True, header_compliant=True,
        ))

    strict_es_avg = sum(strict_edit_sims) / len(all_indices) if all_indices else 0.0

    # ── 2. Relaxed path ───────────────────────────────────────────────────
    # Find content+location matches for targets not covered by strict
    relaxed_es_per_target, used_relaxed_blocks = _compute_relaxed_matches(
        target_hunks, parsed_blocks, applied_set,
    )

    relaxed_matched_indices = set(relaxed_es_per_target.keys())
    relaxed_es_list = [relaxed_es_per_target.get(idx, 0.0) for idx in all_indices]
    relaxed_es_avg = sum(relaxed_es_list) / len(all_indices) if all_indices else 0.0

    # ── 3. Coverage-adjusted penalties ────────────────────────────────────
    # A target is "covered" if it was strictly applied OR relaxed-matched
    covered_targets = len(applied_set | relaxed_matched_indices)
    missing_targets = max(0, len(all_indices) - covered_targets)
    missing_penalty = -MISSING_PENALTY_PER * (missing_targets / len(all_indices)) if all_indices else 0.0

    # A predicted block is "useful" if it strict-applied or used in relaxed matching
    # Only the first `covered_targets` predicted blocks are useful
    total_predicted = len(parsed_blocks)
    useful_blocks = covered_targets
    extra_blocks = max(0, total_predicted - useful_blocks)
    extra_penalty = -EXTRA_PENALTY_PER * (extra_blocks / max(1, total_predicted))

    # ── 4. Bonus / penalty components ─────────────────────────────────────
    all_em_bonus = ALL_EM_BONUS if (len(all_indices) > 0 and all(strict_em_flags)) else 0.0

    step_penalty = -STEP_PENALTY_PER * max(0, action_steps - STEP_FREE_STEPS)

    # ── 5. Final reward ───────────────────────────────────────────────────
    reward = (
        REWARD_LAMBDA * strict_es_avg
        + (1 - REWARD_LAMBDA) * relaxed_es_avg
        + all_em_bonus
        + missing_penalty
        + extra_penalty
        + step_penalty
    )

    logger.debug(
        "reward=%.4f  strict_es=%.4f  relaxed_es=%.4f  all_em=%.1f  "
        "missing=%.4f  extra=%.4f  step=%.4f  "
        "covered=%d/%d  extra_blocks=%d/%d",
        reward, strict_es_avg, relaxed_es_avg, all_em_bonus,
        missing_penalty, extra_penalty, step_penalty,
        covered_targets, len(all_indices), extra_blocks, total_predicted,
    )
    return reward, per_hunk


def compute_and_assign_reward(
    result: FillResult,
    target_hunks: list[dict],
    parsed_blocks: list[dict],
    multi_hunk_metadata: dict[str, Any],
) -> None:
    """Convenience wrapper: compute reward and assign to FillResult in place."""
    reward, per_hunk = compute_multi_hunk_reward(
        target_hunks=target_hunks,
        parsed_blocks=parsed_blocks,
        applied_indices=multi_hunk_metadata.get("multi_hunk_applied_indices", []),
        missing_indices=multi_hunk_metadata.get("multi_hunk_missing_indices", []),
        extra_predicted_indices=multi_hunk_metadata.get("multi_hunk_extra_predicted_indices", []),
        action_steps=result.action_steps,
    )
    result.reward = reward
    result.per_hunk_eval = per_hunk
