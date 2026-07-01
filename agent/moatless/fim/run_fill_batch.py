"""
CLI entrypoint: run FIM tasks in batch from a FillRequest JSONL file.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Optional

from moatless.completion.tool_call import ToolCallCompletionModel
from moatless.fim.dataset import load_requests_from_jsonl, save_results_to_jsonl
from moatless.fim.pipeline import run_fill_batch
from moatless.fim.schema import FillRequest

logger = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run FIM batch evaluation from FillRequest JSONL tasks."
    )
    parser.add_argument("--input", required=True, help="Path to FillRequest JSONL")
    parser.add_argument(
        "--input-format",
        default="fill_request",
        choices=["fill_request", "phase3_ordered_hunks"],
        help="Input JSONL format",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSONL path. Default: moatless/results/fim_results_<timestamp>.jsonl",
    )
    parser.add_argument(
        "--trace-root",
        default="moatless/results/fim_traces",
        help="Trace output directory root",
    )
    parser.add_argument(
        "--results-dir",
        default="moatless/results",
        help="Base results directory used when --output is omitted",
    )
    parser.add_argument(
        "--summary-json",
        default=None,
        help="Optional path to save run summary json",
    )

    parser.add_argument(
        "--max-items",
        type=int,
        default=None,
        help="Load at most N tasks from input",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=4,
        help="Max number of concurrent tasks",
    )
    parser.add_argument(
        "--repo-path-override",
        default=None,
        help="Override repo_path for all tasks",
    )
    parser.add_argument(
        "--repo-map",
        action="append",
        default=[],
        help="Repo name to local path mapping, format: repo_name=/abs/path (repeatable)",
    )
    parser.add_argument(
        "--default-repo-path",
        default=None,
        help="Default repo path fallback when phase3 row cannot resolve by repo_map",
    )
    parser.add_argument(
        "--scan-repos-base",
        default=None,
        help="Scan repos_base/{lang}/{repo} directory structure to auto-build --repo-map. "
             "Explicit --repo-map entries override scanned ones.",
    )
    parser.add_argument(
        "--non-strict-loader",
        action="store_true",
        help="Skip invalid input rows instead of failing fast",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip tasks with existing successful results in trace_root",
    )

    parser.add_argument(
        "--model-name",
        default=None,
        help="Logical model name to write into task.model_name",
    )
    parser.add_argument(
        "--override-model-name",
        action="store_true",
        help="Force override model_name for every loaded task",
    )
    parser.add_argument(
        "--litellm-model",
        default=None,
        help="LiteLLM model id (for example openai/deepseek-chat).",
    )
    parser.add_argument(
        "--model-provider",
        default="openai",
        help="Provider prefix used to build --litellm-model when omitted",
    )
    parser.add_argument(
        "--litellm-custom-provider",
        default=None,
        help=(
            "Optional custom_llm_provider passed to LiteLLM "
            "(for OpenAI-compatible base URLs using raw model ids)."
        ),
    )
    parser.add_argument(
        "--model-api-key",
        default=None,
        help="Model API key (default: env MODEL_API_KEY)",
    )
    parser.add_argument(
        "--model-base-url",
        default=None,
        help="Model base URL (default: env MODEL_BASE_URL)",
    )
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--max-tokens", type=int, default=2000, help="Max completion tokens")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log level",
    )

    # ══ Auto-evaluation parameters ════════════════════════════════════════
    parser.add_argument(
        "--eval-output-dir",
        default=None,
        help="""Eval results output directory (optional).
                If specified, automatically runs evaluation after all tasks complete (calls batch_evaluate).
                Default: {trace_root}/../eval_output (parent of trace_root)/eval_output""",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip automatic evaluation after completion (even if --eval-output-dir is specified)",
    )
    parser.add_argument(
        "--eval-task-jsonl",
        default=None,
        help="""Task JSONL path for evaluation (optional).
                Only needed for single-hunk (provides ground_truth),
                multi-hunk is self-contained with target_hunks.""",
    )
    parser.add_argument(
        "--model-pricing-file", default="moatless/fim/my_pricing.json",
        help="""Model pricing JSON file path (optional). Default: moatless/fim/my_pricing.json""",
    )
    parser.add_argument(
        "--model-pricing-json", default=None,
        help="Model pricing JSON string (alternative to --model-pricing-file)",
    )
    return parser


def _normalize_litellm_model(
    litellm_model: Optional[str],
    model_name: Optional[str],
    model_provider: str,
) -> str:
    if litellm_model:
        return litellm_model

    if not model_name:
        raise ValueError("Missing model id: provide --litellm-model or --model-name.")

    if "/" in model_name:
        return model_name

    provider = model_provider.rstrip("/")
    return f"{provider}/{model_name}"


def _resolve_litellm_custom_provider(
    *,
    litellm_model: str,
    model_provider: str,
    model_base_url: Optional[str],
    explicit_custom_provider: Optional[str],
) -> Optional[str]:
    if explicit_custom_provider:
        return explicit_custom_provider.strip() or None

    provider = model_provider.strip().rstrip("/")
    if not provider or not model_base_url:
        return None

    # If model already contains this provider prefix, LiteLLM can infer it directly.
    if litellm_model.startswith(f"{provider}/"):
        return None

    # For OpenAI-compatible gateways using raw model ids (e.g. Pro/moonshotai/Kimi-K2.5),
    # pass provider explicitly so LiteLLM can route the request.
    return provider


def _apply_model_name(tasks: list[FillRequest], model_name: Optional[str], force: bool) -> None:
    if not model_name:
        return

    for task in tasks:
        if force or not task.model_name or task.model_name == "placeholder-model":
            task.model_name = model_name


def _default_result_jsonl(results_dir: str) -> str:
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    return str(Path(results_dir) / f"fim_results_{timestamp}.jsonl")


def _default_summary_json(results_dir: str) -> str:
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    return str(Path(results_dir) / f"fim_summary_{timestamp}.json")


def _scan_repos(repos_base: str) -> dict[str, str]:
    """Scan repos_base/{lang}/{repo} directory structure for git repos.

    Two-level scan:
      repos_base/{repo}           — if entry is a dir with .git
      repos_base/{lang}/{repo}    — if entry is a dir without .git, recurse one level

    Returns {repo_name: abs_path} map.
    """
    scanned: dict[str, str] = {}
    repos_base = str(repos_base)
    if not os.path.isdir(repos_base):
        logger.warning("scan-repos-base not found: %s", repos_base)
        return scanned

    for entry in os.listdir(repos_base):
        sub_path = os.path.join(repos_base, entry)
        if not os.path.isdir(sub_path):
            continue
        if os.path.isdir(os.path.join(sub_path, ".git")):
            scanned[entry] = sub_path
        else:
            for repo_name in os.listdir(sub_path):
                repo_path = os.path.join(sub_path, repo_name)
                if os.path.isdir(os.path.join(repo_path, ".git")):
                    scanned[repo_name] = repo_path

    logger.info("Scanned %d repos in %s", len(scanned), repos_base)
    return scanned


def _parse_repo_map_entries(entries: list[str]) -> dict[str, str]:
    repo_map: dict[str, str] = {}
    for entry in entries:
        if "=" not in entry:
            raise ValueError(
                f"Invalid --repo-map value: {entry!r}. Expected format: repo_name=/abs/path"
            )
        repo_name, repo_path = entry.split("=", 1)
        repo_name = repo_name.strip()
        repo_path = repo_path.strip()
        if not repo_name or not repo_path:
            raise ValueError(
                f"Invalid --repo-map value: {entry!r}. Expected non-empty repo name and path."
            )
        repo_map[repo_name] = repo_path
    return repo_map


def _classify_outcome(result) -> str:
    """
    Collapse run outcomes into 3 categories only:
    - terminal
    - forced_terminal
    - failed
    """
    if not result.success:
        return "failed"
    if (result.finish_reason or "") == "forced_terminal":
        return "forced_terminal"
    return "terminal"


def _build_summary(results, model_name: str) -> dict:
    total = len(results)
    success = sum(1 for r in results if r.success)
    outcome_dist = Counter(_classify_outcome(r) for r in results)
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_reasoning_tokens = 0
    total_cache_read_tokens = 0
    total_cost_usd = 0.0

    for r in results:
        metadata = r.metadata or {}
        total_prompt_tokens += int(metadata.get("total_prompt_tokens", 0) or 0)
        total_completion_tokens += int(metadata.get("total_completion_tokens", 0) or 0)
        total_reasoning_tokens += int(metadata.get("total_reasoning_tokens", 0) or 0)
        total_cache_read_tokens += int(metadata.get("total_cache_read_tokens", 0) or 0)
        total_cost_usd += float(metadata.get("total_cost_usd", 0.0) or 0.0)

    total_tokens = total_prompt_tokens + total_completion_tokens
    # Keep key name for compatibility, but values are now the 3 collapsed classes.
    finish_reason_dist = {
        "terminal": int(outcome_dist.get("terminal", 0)),
        "forced_terminal": int(outcome_dist.get("forced_terminal", 0)),
        "failed": int(outcome_dist.get("failed", 0)),
    }
    return {
        "total": total,
        "success": success,
        "failed": total - success,
        "success_rate": (success / total) if total else 0.0,
        "model_name": model_name,
        "finish_reason_dist": finish_reason_dist,
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "total_tokens": total_tokens,
        "total_reasoning_tokens": total_reasoning_tokens,
        "total_cache_read_tokens": total_cache_read_tokens,
        "avg_tokens_per_task": (total_tokens / total) if total else 0.0,
        "total_cost_usd": total_cost_usd,
    }


async def _run(args: argparse.Namespace) -> tuple[str, str, dict]:
    default_model_name = args.model_name or "placeholder-model"
    repo_map = _parse_repo_map_entries(args.repo_map)

    # ── Auto-scan repos_base directory and merge into repo_map (explicit --repo-map takes priority) ──
    if args.scan_repos_base:
        scanned = _scan_repos(args.scan_repos_base)
        for k, v in scanned.items():
            if k not in repo_map:
                repo_map[k] = v

    tasks = load_requests_from_jsonl(
        jsonl_path=args.input,
        repo_path_override=args.repo_path_override,
        max_items=args.max_items,
        default_model_name=default_model_name,
        strict=not args.non_strict_loader,
        input_format=args.input_format,
        repo_map=repo_map,
        default_repo_path=args.default_repo_path,
    )
    if not tasks:
        raise ValueError(f"No valid task loaded from {args.input}")

    _apply_model_name(tasks=tasks, model_name=args.model_name, force=args.override_model_name)

    primary_model_name = args.model_name or tasks[0].model_name
    litellm_model = _normalize_litellm_model(
        litellm_model=args.litellm_model,
        model_name=primary_model_name,
        model_provider=args.model_provider,
    )
    model_api_key = args.model_api_key or os.getenv("MODEL_API_KEY")
    model_base_url = args.model_base_url or os.getenv("MODEL_BASE_URL")
    custom_provider = _resolve_litellm_custom_provider(
        litellm_model=litellm_model,
        model_provider=args.model_provider,
        model_base_url=model_base_url,
        explicit_custom_provider=args.litellm_custom_provider,
    )

    if not model_api_key:
        logger.warning("MODEL_API_KEY is empty. Requests may fail on providers that require auth.")

    model_params = {}
    if custom_provider:
        model_params["custom_llm_provider"] = custom_provider
        logger.info(
            "Using LiteLLM custom_llm_provider=%s for model=%s",
            custom_provider,
            litellm_model,
        )

    completion_model = ToolCallCompletionModel(
        model=litellm_model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        model_api_key=model_api_key,
        model_base_url=model_base_url,
        params=model_params,
    )

    results = await run_fill_batch(
        tasks=tasks,
        completion_model=completion_model,
        trace_root=args.trace_root,
        max_concurrency=args.max_concurrency,
        resume=args.resume,
    )

    output_jsonl = args.output or _default_result_jsonl(args.results_dir)
    save_results_to_jsonl(results, output_jsonl)

    summary = _build_summary(results=results, model_name=litellm_model)
    summary_json = args.summary_json or _default_summary_json(args.results_dir)
    Path(summary_json).parent.mkdir(parents=True, exist_ok=True)
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # ══ Automatic evaluation (optional) ══════════════════════════════════
    eval_output_dir = args.eval_output_dir
    if not args.skip_eval and eval_output_dir is not None:
        # Default eval_input_dir = trace_root (contains {task_id}/result.json)
        eval_input_dir = args.trace_root
        if not os.path.isabs(eval_input_dir):
            eval_input_dir = str(Path(args.results_dir).resolve() / eval_input_dir)

        eval_output_path = Path(eval_output_dir)
        if not eval_output_path.is_absolute():
            eval_output_path = Path(args.results_dir) / eval_output_dir
        eval_output_path.mkdir(parents=True, exist_ok=True)

        try:
            from moatless.fim.evaluate_results import batch_evaluate

            # ── Parse model pricing (for cost estimation in automatic evaluation) ──
            pricing_override: dict | None = None
            if args.model_pricing_file:
                pricing_path = Path(args.model_pricing_file)
                if pricing_path.exists():
                    with open(pricing_path, "r", encoding="utf-8") as f:
                        pricing_override = json.load(f)
                else:
                    logger.warning(f"Model pricing file not found: {args.model_pricing_file}, using defaults")
            if args.model_pricing_json:
                pricing_override = json.loads(args.model_pricing_json)

            logger.info(f"Starting automatic evaluation: input={eval_input_dir}, output={eval_output_path}")
            eval_results = batch_evaluate(
                results_dir=eval_input_dir,
                output_dir=str(eval_output_path),
                task_jsonl=args.eval_task_jsonl,
                pricing_override=pricing_override,
            )
            eval_summary_path = eval_output_path / "eval_summary.json"
            if eval_summary_path.exists():
                with open(eval_summary_path, "r", encoding="utf-8") as f:
                    eval_summary = json.load(f)
                logger.info(
                    f"Evaluation complete: "
                    f"avg_exact_match={eval_summary.get('avg_exact_match')}, "
                    f"avg_edit_similarity={eval_summary.get('avg_edit_similarity')}, "
                    f"syntax_pass_rate={eval_summary.get('syntax_pass_rate')}, "
                    f"avg_cost_usd={eval_summary.get('efficiency', {}).get('avg_cost_usd')}"
                )
        except ImportError:
            logger.warning("Could not import batch_evaluate, skipping automatic evaluation.")
        except Exception as e:
            logger.error(f"Automatic evaluation failed: {e}", exc_info=True)

    return output_jsonl, summary_json, summary


def main() -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    parser = _build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    output_jsonl, summary_json, summary = asyncio.run(_run(args))
    print(f"Done. output_jsonl={output_jsonl}")
    print(f"Done. summary_json={summary_json}")
    print(
        "Summary: "
        f"total={summary['total']} success={summary['success']} "
        f"failed={summary['failed']} success_rate={summary['success_rate']:.3f}"
    )
    print(
        "Usage: "
        f"prompt_tokens={summary['total_prompt_tokens']} "
        f"completion_tokens={summary['total_completion_tokens']} "
        f"total_tokens={summary['total_tokens']} "
        f"avg_tokens_per_task={summary['avg_tokens_per_task']:.1f} "
        f"total_cost_usd={summary['total_cost_usd']:.6f}"
    )
    print(f"Finish reasons: {summary['finish_reason_dist']}")


if __name__ == "__main__":
    main()
