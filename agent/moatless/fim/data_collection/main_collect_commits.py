import argparse
import gzip
import os

from loguru import logger

from filters.benchmark_filters import BenchmarkFilter
from mining.miner import RepoMiner


# from mining.miner import RepoMiner
# from filters.benchmark_filters import BenchmarkFilter

def main():
    parser = argparse.ArgumentParser(description="NEP Benchmark Builder - Phase 1 (Revised)")
    parser.add_argument("--repo", type=str, required=True, help="Local path or URL")
    parser.add_argument("--repo_name", type=str, required=True, help="Local path or URL")
    parser.add_argument("--output", type=str, default="output/benchmark_candidates_single.jsonl")
    parser.add_argument("--limit", type=int, default=10)

    args = parser.parse_args()

    # Only need a comprehensive filter
    filters = [BenchmarkFilter()]

    miner = RepoMiner(args.repo_name,args.repo, filters)
    if os.path.dirname(args.output):
        os.makedirs(os.path.dirname(args.output), exist_ok=True)

    logger.info(f"Starting Benchmark Mining on {args.repo}")
    logger.info("Strategy: Merge Commits | Source (Data) + Test (Verifier)")

    count = 0
    try:
        with open(args.output, 'w', encoding='utf-8') as f:
            for candidate in miner.mine(limit=args.limit):
                # Pydantic v2 uses model_dump_json()
                # For v1 version, use .json()
                f.write(candidate.model_dump_json() + "\n")
                count += 1
                logger.info(
                    f"[{count}] {candidate.hash[:7]}: Source={candidate.source_files_count}, Test={candidate.test_files_count}")
    except KeyboardInterrupt:
        logger.warning("Interrupted.")

    logger.success(f"Done. Collected {count} commits.")


if __name__ == "__main__":
    main()