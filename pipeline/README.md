# CoEdPipeline — Code Change Propagation Pipeline

> An automated code change propagation system: given a **root hunk** (a code change that has already occurred), predict other locations in the repository that need coordinated updates, and generate the modified code.

![Python](https://img.shields.io/badge/python-3.10+-blue)
![License](https://img.shields.io/badge/license-MIT-green)

---

## Table of Contents

- [Overview](#overview)
- [Pipeline](#pipeline)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Data Format](#data-format)
- [Output Format](#output-format)
- [License](#license)

---

## Overview

CoEdPipeline processes commit data to determine what other code locations need to change when a root hunk is applied. The pipeline answers: **If this code changes here, what else in the repository needs to change to stay consistent?**

The system operates in two context modes:

| Mode | Description |
|---|---|
| **Oracle** | Uses ground-truth target hunks directly as context |
| **Auto** | Retrieves related code through dependency graph analysis |

---

## Pipeline

```
For each data item:

Step 1   Apply root_hunk to disk
         (Simulates "change already applied" — subsequent file reads
          reflect the post-root state)

Step 2   Build context
         Oracle:  Use target_hunks line ranges directly
         Auto:    Retrieve related code nodes via dependency graph

Step 3   Stage 1 — LLM Call
         Input:  root hunk diff + related context
         Output: { "impacted_locations": [...], "reasoning": "..." }

Step 4   Stage 2 — LLM Call (one per impacted location, parallel per file)
         Input:  current_version (±10 line context) + Stage 1 reasoning
         Output: { "next_version": "...", "change_summary": "..." }

Output: List[PredictedEdit], serialized to output_path as JSONL
```

---

## Installation

### Prerequisites

- Python 3.10+
- Git

### Setup

```bash
# Clone the repository
git clone <repo-url>
cd CoEdPipeline

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
```

---

## Configuration

Copy `.env` to set the following:

| Variable | Description |
|---|---|
| `API_KEY` | LLM API key |
| `OPENAI_API_KEY` | Alternative API key (fallback) |
| `BASE_URL` | API base URL (for proxy or compatible APIs) |
| `LLM_MODEL` | Default model name (overridable via `--model`) |

### Pipeline Thresholds

Pipeline parameters are configured via CLI arguments (see [Usage](#usage)), including:
- Context window sizes (oracle snippet padding, line context margins)
- Cache and log directories
- Filtering options (repo, hash, limit)

---

## Usage

### Quick Start

```bash
# Minimal run (oracle mode, uses GT context directly)
python main.py \
    --data_path  data/dataset.jsonl \
    --repos_base /repos \
    --output_path results/run_oracle.jsonl \
    --context_mode oracle
```

### Full Arguments

| Argument | Required | Default | Description |
|---|---|---|---|
| `--data_path` | Yes | — | Input JSONL dataset path |
| `--repos_base` | Yes | — | Repository root directory |
| `--output_path` | Yes | — | Output JSONL path (append mode, supports resume) |
| `--context_mode` | | `oracle` | `auto` = dependency graph, `oracle` = GT context directly |
| `--dry_run` | | `false` | Skip LLM calls, context retrieval only |
| `--resume` / `--no-resume` | | `true` | Skip already-processed commit hashes |
| `--limit N` | | — | Only process first N items (debugging) |
| `--repo_filter REPO` | | — | Only process specific repo (e.g., `django`) |
| `--hash_filter H1,H2` | | — | Only process specific commit hashes (comma-separated, prefix match) |
| `--nproc N` | | `1` | Number of parallel processes |
| `--model` | | `LLM_MODEL` or `gpt-4o` | Model name |
| `--temperature` | | `0.1` | LLM sampling temperature |
| `--max_tokens` | | `10240` | LLM max output tokens |
| `--api_type` | | `chat` | API endpoint type: `chat`, `completions`, or `responses` |
| `--base_url` | | `BASE_URL` env | API base URL override |
| `--cache_dir` | | `.cache` | Dependency graph cache directory |
| `--log_dir` | | `llm_logs` | LLM interaction log directory |
| `--log_level` | | `INFO` | Console log level |
| `--log_file PATH` | | — | Log file path (DEBUG level) |

### Common Scenarios

```bash
# Full run in oracle mode
python main.py \
    --data_path data/dataset.jsonl \
    --repos_base /repos \
    --output_path results/run_oracle.jsonl \
    --context_mode oracle

# Resume from checkpoint (default: skips existing hashes)
python main.py \
    --data_path data/dataset.jsonl \
    --repos_base /repos \
    --output_path results/run_oracle.jsonl \
    --context_mode oracle \
    --resume

# Filter by repository
python main.py \
    --data_path data/dataset.jsonl \
    --repos_base /repos \
    --output_path results/run_django.jsonl \
    --context_mode oracle \
    --repo_filter django

# Debug: dry-run with first 5 items, DEBUG logging
python main.py \
    --data_path data/dataset.jsonl \
    --repos_base /repos \
    --output_path results/dry_run.jsonl \
    --dry_run --limit 5 --log_level DEBUG

# Single hash verification
python main.py \
    --data_path data/dataset.jsonl \
    --repos_base /repos \
    --output_path results/single.jsonl \
    --hash_filter 1af0271d7c6f,deadbeef1234

# Parallel processing with 4 workers
python main.py \
    --data_path data/dataset.jsonl \
    --repos_base /repos \
    --output_path results/run_parallel.jsonl \
    --nproc 4
```

### Training Dataset Construction

```bash
# Build training set without CoT enrichment
python train_make_main.py \
    --data_path data/dataset.jsonl \
    --repos_base /repos \
    --output_path data/train_set.jsonl \
    --use_cot false

# Build with CoT enrichment
python train_make_main.py \
    --data_path data/dataset.jsonl \
    --repos_base /repos \
    --output_path data/train_set.jsonl \
    --use_cot true
```


## Data Format

### Input Dataset (`dataset.jsonl`)

Each line is a JSON object:

```json
{
  "commit_hash": "c90c4fb6a7d8e1f2b3c4d5e6f7a8b9c0d1e2f3a4",
  "repo_name":   "fastapi",
  "root_hunk": {
    "id":             "hunk_001",
    "file_path":      "fastapi/openapi/docs.py",
    "old_start_line": 65,
    "old_len":        8,
    "new_len":        10,
    "start_line":     65,
    "end_line":       72,
    "before_code":    "def get_swagger_ui_html(...",
    "after_code":     "def get_swagger_ui_html(..."
  },
  "root_before_code": "def get_swagger_ui_html(...",
  "root_after_code":  "def get_swagger_ui_html(...",
  "target_hunks": [
    {
      "id":         "hunk_002",
      "file_path":  "fastapi/applications.py",
      "start_line": 210,
      "end_line":   225
    }
  ]
}
```

### Pipeline Output (`results/*.jsonl`)

```json
{
  "run_id":      "3165cea2-1af0-271d-7c6f-deadbeef1234",
  "commit_hash": "c90c4fb6a7d8e1f2b3c4d5e6f7a8b9c0d1e2f3a4",
  "repo_name":   "fastapi",
  "predictions": [
    {
      "file_path":       "fastapi/applications.py",
      "start_line":      205,
      "end_line":        230,
      "current_version": "    def setup(self) -> None:\n        ...",
      "next_version":    "    def setup(self) -> None:\n        ...",
      "change_summary":  "Updated swagger_ui_oauth2_redirect_url parameter",
      "predicted_order": 0
    }
  ],
  "root_hunk":    { "...": "..." },
  "target_hunks": [ { "...": "..." } ],
  "timing": {
    "total_s":      45.2,
    "graph_build_s": 3.1,
    "llm_calls": [
      { "stage": "Stage1-identify-lines", "duration_s": 12.3, "input_tokens": 2048, "output_tokens": 256 }
    ]
  },
  "token_usage": {
    "total_tokens": 5888,
    "prompt_tokens": 5120,
    "completion_tokens": 768
  }
}
```


## License

[MIT](LICENSE)
