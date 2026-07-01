#!/usr/bin/env python3
"""
Training Set Builder Entry Point

Usage examples:
# Force skip CoT (quick build of raw dataset)
python train_main.py \
    --data_path  data/dataset.jsonl \
    --repos_base /repos \
    --output_path data/train_set.jsonl \
    --use_cot false

# Force CoT (even if env vars not configured, will show error)
python train_main.py \
    --data_path  data/dataset.jsonl \
    --repos_base /repos \
    --output_path data/train_set.jsonl \
    --use_cot true

  # Full build
  python train_main.py \\
      --data_path  data/dataset.jsonl \\
      --repos_base /repos \\
      --output_path data/train_set.jsonl

  # Build for specific repo only, resume from checkpoint
  python train_main.py \\
      --data_path  data/dataset.jsonl \\
      --repos_base /repos \\
      --output_path data/train_set.jsonl \\
      --repo_filter django \\
      --resume

  # Debug: only first 5 items, print DEBUG logs
  python train_main.py \\
      --data_path  data/dataset.jsonl \\
      --repos_base /repos \\
      --output_path data/train_set.jsonl \\
      --limit 5 --log_level DEBUG

  # Verify single hash
  python train_main.py \\
      --data_path  data/dataset.jsonl \\
      --repos_base /repos \\
      --output_path data/train_set.jsonl \\
      --hash_filter 1af0271d7c6f,deadbeef1234
"""

import argparse
import json
import logging
import os
import subprocess
import sys
from datetime import datetime

from dotenv import load_dotenv


# ══════════════════════════════════════════════════════════════════════════
# Logging initialization
# ══════════════════════════════════════════════════════════════════════════

def _setup_logging(level_name: str, log_file: str | None = None) -> None:
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
    root.setLevel(logging.DEBUG)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(console_fmt)
    root.addHandler(ch)

    if log_file:
        log_dir = os.path.dirname(os.path.abspath(log_file))
        os.makedirs(log_dir, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(file_fmt)
        root.addHandler(fh)
        logging.getLogger(__name__).info("Log file: %s", log_file)


def _resolve_output_path(output_path: str) -> str:
    """
    If output_path is a directory or ends with /, auto-generate a timestamped filename.
    Otherwise use the original path (append mode).
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    if output_path.endswith("/") or output_path.endswith("\\") or os.path.isdir(output_path):
        base_dir = output_path.rstrip("/\\")
        return os.path.join(base_dir, f"train_set__{ts}.jsonl")

    root, ext = os.path.splitext(output_path)
    if not ext:
        ext = ".jsonl"
    return f"{root}{ext}"   # No timestamp for training set, convenient for resume on same file


# ══════════════════════════════════════════════════════════════════════════
# Argument Parsing
# ══════════════════════════════════════════════════════════════════════════

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="train_main.py",
        description="Training Set Builder — Build LLM training samples from code change dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Required ──────────────────────────────────────────────────────
    req = p.add_argument_group("required")
    req.add_argument("--data_path",   required=True, help="Input JSONL dataset path")
    req.add_argument("--repos_base",  required=True, help="Repository root directory (each repo is a subdirectory under it)")
    req.add_argument("--output_path", required=True, help="Output training set JSONL path (append mode, supports resume)")

    # ── Build Control ─────────────────────────────────────────────────
    ctrl = p.add_argument_group("control")
    ctrl.add_argument("--limit",      type=int, default=None, metavar="N",
                      help="Only process first N items (debugging)")
    ctrl.add_argument("--resume",     action=argparse.BooleanOptionalAction, default=True,
                      help="Skip already-processed commit_hash (default: enabled)")
    ctrl.add_argument("--repo_filter", default=None, metavar="REPO",
                      help="Only process specified repo (e.g. django)")
    ctrl.add_argument("--hash_filter", default=None, metavar="H1,H2,...",
                      help="Only process specified commit hashes (comma-separated, supports prefix matching)")
    ctrl.add_argument("--nproc", type=int, default=1, metavar="N",
                      help="Number of parallel processes (default 1, >1 enables multi-process parallel processing)")

    # ── Window Parameters ──────────────────────────────────────────────
    win = p.add_argument_group("window")
    win.add_argument("--context_window", type=int, default=30,
                     help="Oracle context window size (lines, for Stage1 prompt)")
    win.add_argument("--snippet_window", type=int, default=10,
                     help="Stage2 code snippet window size (lines)")
    # ── CoT Control ──────────────────────────────────────────────────
    cot = p.add_argument_group("cot")
    cot.add_argument(
        "--use_cot",
        default=None,
        choices=["true", "false"],
        metavar="true|false",
        help=(
            "CoT enhancement switch. "
            "true=force CoT; false=force skip; "
            "not set=follow default behavior based on whether cot_provider is configured"
        ),
    )
    # ── Directories ──────────────────────────────────────────────────────
    dirs = p.add_argument_group("directories")
    dirs.add_argument("--cache_dir", default=".cache",     help="Sandbox cache directory")
    dirs.add_argument("--log_dir",   default="train_logs", help="Log directory")

    # ── Logging ──────────────────────────────────────────────────────
    log_grp = p.add_argument_group("logging")
    log_grp.add_argument("--log_level", default="INFO",
                         choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                         help="Console log level")
    log_grp.add_argument("--log_file", default=None, metavar="PATH",
                         help="Log file path (DEBUG level, no file if not specified)")

    return p


# ══════════════════════════════════════════════════════════════════════════
# Parallel Execution
# ══════════════════════════════════════════════════════════════════════════

def _add_opt(cmd: list[str], flag: str, value, default=None) -> None:
    if value is not None and value != default:
        cmd.append(f"{flag}={value}")

def _run_parallel(args: argparse.Namespace, resolved_output: str, logger: logging.Logger) -> int:
    nproc = args.nproc
    out_dir = os.path.dirname(os.path.abspath(resolved_output))

    # ── Resume: load already-processed commit_hashes ────────────────────
    done_hashes: set[str] = set()
    if args.resume and os.path.exists(resolved_output):
        try:
            with open(resolved_output, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        h = json.loads(line).get("meta", {}).get("commit_hash", "")
                        if h:
                            done_hashes.add(h)
                    except json.JSONDecodeError:
                        continue
        except OSError as e:
            logger.warning("Could not read output for resume: %s", e)

    with open(args.data_path, "r", encoding="utf-8") as f:
        all_lines = [line for line in f if line.strip()]

    # Filter already-processed hashes
    if done_hashes:
        filtered: list[str] = []
        skipped = 0
        for line in all_lines:
            try:
                raw = json.loads(line)
                if raw.get("hash", "") in done_hashes:
                    skipped += 1
                    continue
            except json.JSONDecodeError:
                pass
            filtered.append(line)
        logger.info("Resume: skipped %d already-processed items (out of %d)", skipped, len(all_lines))
        lines = filtered
    else:
        lines = all_lines

    logger.info("Splitting %d items into %d shards...", len(lines), nproc)

    shards: list[list[str]] = [[] for _ in range(nproc)]
    for i, line in enumerate(lines):
        shards[i % nproc].append(line)

    basename = os.path.basename(resolved_output)
    shard_files: list[str] = []
    for i, shard in enumerate(shards):
        shard_path = os.path.join(out_dir, f".shard_{basename}_{i}")
        with open(shard_path, "w", encoding="utf-8") as f:
            for line in shard:
                f.write(line + "\n")
        shard_files.append(shard_path)
        logger.info("  Shard %d: %d items", i, len(shard))

    part_files = [resolved_output.replace(".jsonl", f".part{i}.jsonl") for i in range(nproc)]
    procs: list[tuple[int, subprocess.Popen]] = []

    for i in range(nproc):
        if not shards[i]:
            logger.info("  Worker %d: empty shard, skipping.", i)
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
        _add_opt(cmd, "--context_window", args.context_window, 30)
        _add_opt(cmd, "--snippet_window", args.snippet_window, 10)
        _add_opt(cmd, "--use_cot", args.use_cot)
        if args.repo_filter:
            cmd.append(f"--repo_filter={args.repo_filter}")
        if args.limit:
            cmd.append(f"--limit={args.limit}")
        if args.hash_filter:
            cmd.append(f"--hash_filter={args.hash_filter}")
        if args.log_file:
            cmd.append(f"--log_file={args.log_file}_worker{i}")

        logger.info("  Worker %d: %d items", i, len(shards[i]))
        procs.append((i, subprocess.Popen(cmd)))

    failed_workers = 0
    for i, proc in procs:
        ret = proc.wait()
        if ret != 0:
            logger.warning("Worker %d exited with code %d", i, ret)
            failed_workers += 1
        else:
            logger.info("Worker %d completed", i)

    # Merge part files
    logger.info("Merging %d part files -> %s", len(part_files), resolved_output)
    total = 0
    for part_file in part_files:
        if os.path.exists(part_file):
            with open(part_file, "r", encoding="utf-8") as src:
                with open(resolved_output, "a", encoding="utf-8") as dst:
                    for line in src:
                        if line.strip():
                            dst.write(line)
                            total += 1
            os.remove(part_file)

    for sf in shard_files:
        if os.path.exists(sf):
            os.remove(sf)

    logger.info("Parallel complete: %d records -> %s", total, resolved_output)
    return failed_workers


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
    logger.info(f"  Training Set Builder  —  {ts}")
    logger.info("=" * 60)
    logger.info(f"  data_path      : {args.data_path}")
    logger.info(f"  repos_base     : {args.repos_base}")
    logger.info(f"  output_path    : {args.output_path}")
    logger.info(f"  resume         : {args.resume}")
    logger.info(f"  context_window : {args.context_window}")
    logger.info(f"  snippet_window : {args.snippet_window}")
    logger.info(f"  use_cot        : {args.use_cot if args.use_cot is not None else '(default)'}")
    if args.limit:
        logger.info(f"  limit          : {args.limit}")
    if args.repo_filter:
        logger.info(f"  repo_filter    : {args.repo_filter}")
    if args.hash_filter:
        logger.info(f"  hash_filter    : {args.hash_filter}")
    logger.info("=" * 60)

    # ── Input validation ─────────────────────────────────────────────
    if not os.path.isfile(args.data_path):
        logger.error(f"data_path not found: {args.data_path}")
        sys.exit(1)
    if not os.path.isdir(args.repos_base):
        logger.error(f"repos_base not found: {args.repos_base}")
        sys.exit(1)

    # ── Lazy import business modules ─────────────────────────────────
    from src.core.train_make_runner import TrainRunner, TrainRunnerConfig

    # ── hash_filter parsing ──────────────────────────────────────────
    hash_filter_list: list[str] | None = None
    if args.hash_filter:
        hash_filter_list = [
            h.strip() for h in args.hash_filter.split(",") if h.strip()
        ]
        logger.info("hash_filter parsed: %s", hash_filter_list)

    # ── Resolve output path ──────────────────────────────────────────
    resolved_output = _resolve_output_path(args.output_path)
    logger.info(f"  resolved output: {resolved_output}")

    # ── Build Config ─────────────────────────────────────────────────
    if args.use_cot == "true":
        use_cot = True
    elif args.use_cot == "false":
        use_cot = False
    else:
        use_cot = None

    config = TrainRunnerConfig(
        data_path=args.data_path,
        repos_base=args.repos_base,
        output_path=resolved_output,
        cache_dir=args.cache_dir,
        log_dir=args.log_dir,
        limit=args.limit,
        resume=args.resume,
        repo_filter=args.repo_filter,
        hash_filter=hash_filter_list,
        context_window=args.context_window,
        snippet_window=args.snippet_window,
        use_cot=use_cot,
    )


    # ── Run ──────────────────────────────────────────────────────────
    if args.nproc and args.nproc > 1:
        failed_workers = _run_parallel(args, resolved_output, logger)
        if failed_workers:
            logger.warning("%d worker(s) exited with error", failed_workers)
        _convert_jsonl_to_alpaca(resolved_output, logger)
        sys.exit(0 if failed_workers == 0 else 1)

    runner = TrainRunner(config=config)

    try:
        stats = runner.run_all()
    except KeyboardInterrupt:
        logger.warning("\nInterrupted by user (Ctrl+C). Partial results saved.")
        sys.exit(130)
    except Exception as e:
        logger.exception(f"Fatal error: {e}")
        sys.exit(1)

    # ── Final statistics ─────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("  Final Stats")
    logger.info("=" * 60)
    for k, v in stats.summary().items():
        logger.info(f"  {k:<22}: {v}")
    logger.info("=" * 60)

    if stats.samples_written == 0:
        logger.warning("No samples were written. Check data / filters.")
        sys.exit(2)

    total_failed = stats.failed_checkout + stats.failed_parse + stats.failed_build
    if total_failed > 0:
        logger.warning(f"{total_failed} item(s) failed. Check logs for details.")
        sys.exit(2)

    # ── Convert to Alpaca format ─────────────────────────────────────
    _convert_jsonl_to_alpaca(resolved_output, logger)


def _convert_jsonl_to_alpaca(jsonl_path: str, logger: logging.Logger) -> None:
    """
    Convert JSONL ({messages, ground_truth, ...}) to Alpaca-format JSON file.
    Output path: same directory as jsonl_path, with extension changed to .json.

    Alpaca format:
      [{"instruction": "", "input": "", "output": "", "system": ""}]
    """
    if not os.path.exists(jsonl_path):
        logger.warning(f"No JSONL found for conversion: {jsonl_path}")
        return

    json_path = jsonl_path.rsplit(".", 1)[0] + ".json"
    records: list[dict] = []

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                sample = json.loads(line)
            except json.JSONDecodeError:
                continue

            messages = sample.get("messages", [])
            if len(messages) < 2:
                continue

            system_prompt = messages[0].get("content", "")
            user_content  = messages[-1].get("content", "")
            ground_truth  = sample.get("ground_truth", "")

            records.append({
                "system":      system_prompt,
                "instruction": user_content,
                "input":       "",
                "output":      ground_truth,
            })

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    logger.info("Converted %d samples -> %s", len(records), json_path)


if __name__ == "__main__":
    main()
