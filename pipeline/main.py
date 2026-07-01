#!/usr/bin/env python3
"""
MultiPoint Pipeline — Main Entry

Usage examples:
  # Full run (auto mode)
  python main.py \\
      --data_path  data/dataset.jsonl \\
      --repos_base /repos \\
      --output_path results/run_auto.jsonl

  # Oracle mode, run only django, resume from checkpoint
  python main.py \\
      --data_path  data/dataset.jsonl \\
      --repos_base /repos \\
      --output_path results/run_oracle.jsonl \\
      --context_mode oracle \\
      --repo_filter django \\
      --resume

  # Debug: dry-run, only first 5 items, print DEBUG logs
  python main.py \\
      --data_path  data/dataset.jsonl \\
      --repos_base /repos \\
      --output_path results/dry_run.jsonl \\
      --dry_run --limit 5 --log_level DEBUG

  # Verify single hash
  python main.py \\
      --data_path  data/dataset.jsonl \\
      --repos_base /repos \\
      --output_path results/single.jsonl \\
      --hash_filter 1af0271d7c6f,deadbeef1234
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import tempfile
from datetime import datetime

from dotenv import load_dotenv


# ══════════════════════════════════════════════════════════════════════════
# Logging setup (done before importing business modules to ensure all
# modules inherit the configuration)
# ══════════════════════════════════════════════════════════════════════════

def _setup_logging(level_name: str, log_file: str | None = None) -> None:
    """
    Configure root logger:
      - Console handler: specified level, concise format
      - File handler (optional): DEBUG level, full format
    """
    level = getattr(logging, level_name.upper(), logging.INFO)

    console_fmt = logging.Formatter(
        fmt="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    file_fmt = logging.Formatter(
        fmt="%(asctime)s  %(levelname)-8s  [%(name)s:%(lineno)d]  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)  # root set lowest, filtered by each handler

    # Console
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(console_fmt)
    root.addHandler(ch)

    # File (optional)
    if log_file:
        # FIX: dirname may be empty (when only filename is given), handle specially
        log_dir = os.path.dirname(os.path.abspath(log_file))
        os.makedirs(log_dir, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(file_fmt)
        root.addHandler(fh)
        logging.getLogger(__name__).info(f"📄 Log file: {log_file}")

def _resolve_output_path(output_path: str, model: str) -> str:
    ts         = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_slug = model.replace("/", "-").replace(":", "-").replace(" ", "_")

    if output_path.endswith("/") or output_path.endswith("\\") or os.path.isdir(output_path):
        base_dir = output_path.rstrip("/\\")
        return os.path.join(base_dir, f"run__{model_slug}__{ts}.jsonl")

    return output_path

# ══════════════════════════════════════════════════════════════════════════
# Argument Parsing
# ══════════════════════════════════════════════════════════════════════════

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="main.py",
        description="MultiPoint Pipeline — batch code change propagation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Required ──────────────────────────────────────────────────────
    req = p.add_argument_group("required")
    req.add_argument(
        "--data_path", required=True,
        help="Input JSONL dataset path",
    )
    req.add_argument(
        "--repos_base", required=True,
        help="Repository root directory (each repo is a subdirectory under it)",
    )
    req.add_argument(
        "--output_path", required=True,
        help="Output JSONL path (append mode, supports resume)",
    )

    # ── Pipeline Behavior ─────────────────────────────────────────────
    pipe = p.add_argument_group("pipeline")
    pipe.add_argument(
        "--context_mode", choices=["auto", "oracle"], default="oracle",
        help="Context retrieval mode: auto=dependency graph, oracle=use GT directly",
    )
    pipe.add_argument(
        "--dry_run", action="store_true",
        help="Skip LLM calls, only do context retrieval and recall validation",
    )

    # ── Filter & Control ──────────────────────────────────────────────
    ctrl = p.add_argument_group("control")
    ctrl.add_argument(
        "--limit", type=int, default=None, metavar="N",
        help="Only process first N records (debugging)",
    )
    # FIX: resume uses BooleanOptionalAction (Python 3.9+), clearer semantics
    # Equivalent: --resume to enable, --no-resume to disable, default enabled
    ctrl.add_argument(
        "--resume", action=argparse.BooleanOptionalAction, default=True,
        help="Skip already-processed commit hashes in output_path (default: enabled)",
    )
    ctrl.add_argument(
        "--repo_filter", default=None, metavar="REPO",
        help="Only process specified repo (e.g. django)",
    )
    ctrl.add_argument(
        "--hash_filter", default=None, metavar="H1,H2,...",
        help="Only process specified commit hashes (comma-separated, supports prefix matching)",
    )
    ctrl.add_argument(
        "--nproc", type=int, default=1, metavar="N",
        help="Number of parallel processes (default 1, >1 enables multi-process parallel processing)",
    )

    # ── Directories ───────────────────────────────────────────────────
    dirs = p.add_argument_group("directories")
    dirs.add_argument("--cache_dir", default=".cache",  help="Dependency graph cache directory")
    dirs.add_argument("--log_dir",   default="llm_logs", help="LLM interaction log directory")

    # ── Logging ───────────────────────────────────────────────────────
    log_grp = p.add_argument_group("logging")
    log_grp.add_argument(
        "--log_level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Console log level",
    )
    log_grp.add_argument(
        "--log_file", default=None, metavar="PATH",
        help="Log file path (DEBUG level, no file if not specified)",
    )

    # ── LLM ───────────────────────────────────────────────────────────
    llm = p.add_argument_group("llm")
    llm.add_argument(
        "--model", default=None,
        help="Model name (priority: --model > env LLM_MODEL > gpt-4o)",
    )
    llm.add_argument(
        "--base_url", default=None,
        help="Base URL (priority: --base_url > env BASE_URL)",
    )
    llm.add_argument(
        "--api_type", default="chat", choices=["chat", "completions", "responses"],
        help="API endpoint type: chat (/v1/chat/completions), completions (/v1/completions), responses (/v1/responses)",
    )
    llm.add_argument(
        "--temperature", type=float, default=1,
        help="LLM sampling temperature",
    )
    llm.add_argument(
        "--max_tokens", type=int, default=10240,
        help="LLM max output tokens",
    )

    return p


# ══════════════════════════════════════════════════════════════════════════
# LLM Client Construction
# ══════════════════════════════════════════════════════════════════════════

def _build_llm_client(args: argparse.Namespace):
    """
    Build an LLMClient instance from arguments and environment variables.

    Environment variables (API Key):
      API_KEY (environment variable)

    Environment variables (Base URL):
      BASE_URL (environment variable)

    Model priority:
      --model  >  LLM_MODEL  >  "gpt-4o"
    """
    from src.infrastructure.llm.deepseek_client import DeepSeekClient

    # FIX: Remove hardcoded API key, read from environment variable uniformly
    api_key = os.environ.get("API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logging.getLogger(__name__).warning(
            "⚠️  No API key found in API_KEY / OPENAI_API_KEY. "
            "Requests may fail."
        )

    # FIX: base_url actually passed to client
    base_url = args.base_url or os.environ.get("BASE_URL") or os.environ.get("OPENAI_BASE_URL")

    model = (
        args.model
        or os.environ.get("LLM_MODEL")
    )

    # FIX: Pass temperature / max_tokens / base_url to client
    client_kwargs: dict = dict(
        api_key=api_key or "",
        model=model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        base_url = base_url or "",
        api_type = args.api_type,
    )
    if base_url:
        client_kwargs["base_url"] = base_url

    client = DeepSeekClient(**client_kwargs)

    logging.getLogger(__name__).info(
        f"🤖 LLM: model={model} | api_type={args.api_type}"
        f" | temperature={args.temperature} | max_tokens={args.max_tokens}"
        + (f" | base_url={base_url}" if base_url else "")
    )
    return client,model


# ══════════════════════════════════════════════════════════════════════════
# Parallel Execution
# ══════════════════════════════════════════════════════════════════════════

def _run_parallel(
    args: argparse.Namespace,
    resolved_output: str,
    logger: logging.Logger,
) -> int:
    """
    Multi-process parallel execution: split data -> independent subprocess
    runs -> merge outputs.

    Each subprocess runs the full main.py logic (its own Runner, LLM client),
    writes to an independent .part file, and finally merges to resolved_output.

    Returns:
        failed_workers: number of workers with non-zero exit code
    """
    nproc = args.nproc
    out_dir = os.path.dirname(os.path.abspath(resolved_output))

    # ── 1. Read all data lines ──────────────────────────────────────────
    with open(args.data_path, "r", encoding="utf-8") as f:
        lines = [line for line in f if line.strip()]

    logger.info(f"📦 Splitting {len(lines)} items into {nproc} shards...")

    # ── 2. Round-robin sharding ─────────────────────────────────────────
    shards: list[list[str]] = [[] for _ in range(nproc)]
    for i, line in enumerate(lines):
        shards[i % nproc].append(line)

    # ── 3. Write temporary shard files ──────────────────────────────────
    basename = os.path.basename(resolved_output)
    shard_files: list[str] = []
    for i, shard in enumerate(shards):
        shard_path = os.path.join(out_dir, f".shard_{basename}_{i}")
        with open(shard_path, "w", encoding="utf-8") as f:
            for line in shard:
                f.write(line + "\n")
        shard_files.append(shard_path)
        logger.info(f"  Shard {i}: {len(shard)} items → {os.path.basename(shard_path)}")

    # ── 4. Build subprocess commands ────────────────────────────────────
    part_files = [
        resolved_output.replace(".jsonl", f".part{i}.jsonl")
        for i in range(nproc)
    ]

    procs: list[tuple[int, subprocess.Popen]] = []
    for i in range(nproc):
        if not shards[i]:
            logger.info(f"  Worker {i}: empty shard, skipping.")
            continue

        cmd = [
            sys.executable, sys.argv[0],
            f"--data_path={shard_files[i]}",
            f"--repos_base={args.repos_base}",
            f"--output_path={part_files[i]}",
            f"--log_dir={os.path.join(args.log_dir, f'worker{i}')}",
            "--no-resume",
            "--nproc=1",
        ]

        # Forward optional pipeline parameters
        _add_opt(cmd, "--context_mode", args.context_mode, "oracle")
        if args.model:
            cmd.append(f"--model={args.model}")
        if args.base_url:
            cmd.append(f"--base_url={args.base_url}")
        if args.api_type != "chat":
            cmd.append(f"--api_type={args.api_type}")
        if args.temperature is not None:
            cmd.append(f"--temperature={args.temperature}")
        if args.max_tokens is not None:
            cmd.append(f"--max_tokens={args.max_tokens}")
        if args.dry_run:
            cmd.append("--dry_run")
        if args.repo_filter:
            cmd.append(f"--repo_filter={args.repo_filter}")
        if args.limit:
            cmd.append(f"--limit={args.limit}")
        if args.hash_filter:
            cmd.append(f"--hash_filter={args.hash_filter}")
        if args.log_file:
            cmd.append(f"--log_file={args.log_file}_worker{i}")

        logger.info(f"  🚀 Worker {i}: {len(shards[i])} items")
        procs.append((i, subprocess.Popen(cmd)))

    # ── 5. Wait for all workers ─────────────────────────────────────────
    failed_workers = 0
    for i, proc in procs:
        ret = proc.wait()
        if ret != 0:
            logger.warning(f"⚠️  Worker {i} exited with code {ret}")
            failed_workers += 1
        else:
            logger.info(f"  ✅ Worker {i} completed")

    # ── 6. Merge output files ───────────────────────────────────────────
    logger.info(f"📦 Merging {len(part_files)} part files → {resolved_output}")
    total_lines = 0
    for part_file in part_files:
        if os.path.exists(part_file):
            with open(part_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        total_lines += 1
            # Append to final output
            with open(part_file, "r", encoding="utf-8") as src:
                with open(resolved_output, "a", encoding="utf-8") as dst:
                    for line in src:
                        if line.strip():
                            dst.write(line)
            os.remove(part_file)

    # ── 7. Cleanup ─────────────────────────────────────────────────────
    for sf in shard_files:
        if os.path.exists(sf):
            os.remove(sf)

    logger.info(
        f"🏁 Parallel run complete: {total_lines} records → {resolved_output}"
    )
    if failed_workers:
        logger.warning(f"⚠️  {failed_workers} worker(s) failed. Output may be incomplete.")

    return failed_workers


def _add_opt(cmd: list[str], flag: str, value: str | None, default: str | None = None) -> None:
    """Only add command-line argument when value is non-empty and not equal to default."""
    if value and value != default:
        cmd.append(f"{flag}={value}")


# ══════════════════════════════════════════════════════════════════════════
# Entry Point
# ══════════════════════════════════════════════════════════════════════════

def main() -> None:

    load_dotenv()
    parser = _build_parser()
    args   = parser.parse_args()

    # ── Logging initialization (first)────────────────────────────────
    _setup_logging(args.log_level, args.log_file)
    logger = logging.getLogger(__name__)

    # ── Startup banner ───────────────────────────────────────────────
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info("=" * 60)
    logger.info(f"  MultiPoint Pipeline  —  {ts}")
    logger.info("=" * 60)
    logger.info(f"  data_path    : {args.data_path}")
    logger.info(f"  repos_base   : {args.repos_base}")
    logger.info(f"  output_path  : {args.output_path}")
    logger.info(f"  context_mode : {args.context_mode}")
    logger.info(f"  dry_run      : {args.dry_run}")
    logger.info(f"  resume       : {args.resume}")
    if args.limit:
        logger.info(f"  limit        : {args.limit}")
    if args.repo_filter:
        logger.info(f"  repo_filter  : {args.repo_filter}")
    if args.hash_filter:
        logger.info(f"  hash_filter  : {args.hash_filter}")
    if args.nproc and args.nproc > 1:
        logger.info(f"  nproc        : {args.nproc}")
    logger.info("=" * 60)

    # ── Input validation ─────────────────────────────────────────────
    if not os.path.isfile(args.data_path):
        logger.error(f"❌ data_path not found: {args.data_path}")
        sys.exit(1)

    if not os.path.isdir(args.repos_base):
        logger.error(f"❌ repos_base not found: {args.repos_base}")
        sys.exit(1)

    # ── Parallel branch (nproc>1 subprocess mode, no Runner/LLM client)──
    if args.nproc and args.nproc > 1:
        model_name = args.model or os.environ.get("LLM_MODEL") or "unknown"
        resolved_output = _resolve_output_path(args.output_path, model_name)
        logger.info(f"  output_path  : {resolved_output}")
        logger.info("=" * 60)
        failed_workers = _run_parallel(args, resolved_output, logger)
        sys.exit(1 if failed_workers else 0)

    # ── Lazy import business modules ─────────────────────────────────
    from src.core.runner import Runner, RunnerConfig

    # ── hash_filter parsing ──────────────────────────────────────────
    hash_filter_list: list[str] | None = None
    if args.hash_filter:
        hash_filter_list = [
            h.strip() for h in args.hash_filter.split(",") if h.strip()
        ]
        logger.info(f"🔍 hash_filter: {hash_filter_list}")
    # ── Build LLM Client ─────────────────────────────────────────────
    llm_client, model_name = _build_llm_client(args)
    resolved_output = _resolve_output_path(args.output_path, model_name)
    logger.info(f"  output_path  : {resolved_output}")
    llm_logdir = str(os.path.join(args.log_dir, model_name or "unknown", ts.replace(":", "-").replace(" ", "_")))
    # ── Build RunnerConfig ───────────────────────────────────────────
    config = RunnerConfig(
        data_path=args.data_path,
        repos_base=args.repos_base,
        output_path=resolved_output,
        cache_dir=args.cache_dir,
        log_dir=llm_logdir,
        context_mode=args.context_mode,
        dry_run=args.dry_run,
        limit=args.limit,
        resume=args.resume,
        repo_filter=args.repo_filter,
        hash_filter=hash_filter_list,
    )


    # ── Run ──────────────────────────────────────────────────────────
    runner = Runner(config=config, llm_client=llm_client)

    try:
        stats = runner.run_all()
    except KeyboardInterrupt:
        logger.warning("\n⚠️  Interrupted by user (Ctrl+C). Partial results saved.")
        sys.exit(130)
    except Exception as e:
        logger.exception(f"❌ Fatal error: {e}")
        sys.exit(1)

    # ── Final statistics ─────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("  Final Stats")
    logger.info("=" * 60)
    for k, v in stats.summary().items():
        logger.info(f"  {k:<22}: {v}")
    logger.info("=" * 60)

    # FIX: Use getattr for safe access, avoid silent errors when Stats structure changes
    total_failed = sum(
        getattr(stats, field, 0)
        for field in ("failed_checkout", "failed_parse", "failed_pipeline")
    )
    if total_failed > 0:
        logger.warning(f"⚠️  {total_failed} item(s) failed. Check logs for details.")
        sys.exit(2)


if __name__ == "__main__":
    main()
