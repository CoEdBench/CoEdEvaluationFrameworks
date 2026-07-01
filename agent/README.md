# CoEdAgent — Code Change Propagation Agent

> An automated code change propagation system built on [moatless-tools](https://github.com/aorwall/moatless-tools). Given a code change (a hunk), it identifies all other locations in the repository that need coordinated updates and generates the modified code.

![Python](https://img.shields.io/badge/python-3.10+-blue)
![License](https://img.shields.io/badge/license-MIT-green)

---

## Table of Contents

- [Overview](#overview)
- [Acknowledgments](#acknowledgments)
- [Architecture](#architecture)
- [FIM Module](#fim-module)
- [Data Collection Pipeline](#data-collection-pipeline)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [License](#license)

---

## Overview

CoEdAgent adapts the [moatless-tools](https://github.com/aorwall/moatless-tools) framework—an agentic loop system for code editing—into a **Fill-in-Middle (FIM) completion pipeline** for change propagation. The system answers: **If this code changes here, what else in the repository needs to change to stay consistent?**

The core workflow:

```
Input: root_hunk (change A) + repository state
  │
  ├─ 1. Apply root_hunk to simulate "change already applied"
  ├─ 2. Inject FIM mask at target location (or known input hunk for multi-hunk mode)
  ├─ 3. Agentic exploration: read files, grep, search → identify impacted locations
  ├─ 4. Generate modified code → SubmitCompletion
  └─ Output: predicted edit with change summary
```

Two task modes are supported:

| Mode | Description |
|---|---|
| **Single-hunk** | Predict the completion for one target code span given a known root hunk |
| **Multi-hunk one-shot** | Predict completions for multiple target locations in a single agent run |

---

## Acknowledgments

This project is based on **[moatless-tools](https://github.com/aorwall/moatless-tools)** by Albert Örwall ([@aorwall](https://github.com/aorwall)). The original framework provides:

- **AgenticLoop**: A tree-based execution loop for autonomous code editing
- **ActionAgent**: Tool-augmented agent with file read/write/search capabilities
- **FileRepository**: Repository-level file abstraction with search support
- **Workspace**: Composable environment (file system + bash)

Our custom modifications extend moatless with a **FIM-specific data collection pipeline**. All custom code resides in the `moatless/fim/` directory.

---

## Architecture

```
moatless/
├── fim/                          ← Custom FIM pipeline (our contributions)
│   ├── pipeline.py               Core execution: run_fill_task / run_fill_batch
│   ├── schema.py                 Data structures: FillRequest, FillResult, HunkEvalResult
│   ├── prompt.py                 System/user prompt builders for FIM tasks
│   ├── dataset.py                Dataset loading and serialization
│   ├── utils.py                  Utilities (line replacement, trace persistence)
│   ├── reward.py                 Reward computation
│   ├── rl_types.py               RL data types
│   ├── rl_export.py              RL data export
│   ├── rl_to_sft.py              RL to SFT format conversion
│   ├── filter_quality.py         Quality filtering
│   ├── run_fill_batch.py         CLI entry for batch execution
│   └── data_collection/          Data mining and filtering pipeline
│       ├── config/settings.py        Mining thresholds and LLM configuration
│       ├── core/types.py             Core data types for commit processing
│       ├── filters/                  Commit filtering strategies
│       │   ├── base.py               Abstract filter interface
│       │   ├── basic_filters.py      Basic commit statistics filters
│       │   └── benchmark_filters.py  Benchmark-specific filters
│       ├── mining/miner.py           Commit mining implementation
│       ├── main_collect_commits.py   Entry: collect and filter commits
│       ├── phase1_stats.py           Phase 1 statistics computation
│       ├── run_main_PP.py            Post-processing runner
│       └── convert_*.py              Data format converters
└── ...                           ← Original moatless code (unchanged)
```

---

## FIM Module

The FIM module (`moatless/fim/`) is our primary contribution. It implements:

### Fill-in-Middle Task

A **FillRequest** specifies:
- Repository path, file path, and line range to fill
- Language and model configuration
- Optional ground truth for validation
- Optional known-input hunk for multi-hunk mode

The pipeline:
1. **Pre-flight**: Checkout the target commit in an isolated temp clone; inject FIM mask at the target location
2. **Agentic exploration**: The agent searches the codebase, reads related files, and identifies what code should fill the masked span
3. **Finish & collect**: Extract the completion, compute git patch, persist trace directory

### Key Features

- **Isolated repo checkout**: Each task clones into a temp directory at the target commit
- **Forced finish fallback**: If the agent hits `max_iterations`, a forced finish round ensures a completion is captured
- **Multi-hunk one-shot**: Parse structured completion blocks with `<TARGET_HUNK>` markers for multiple file edits
- **Trace persistence**: Full trajectory logs saved per-task for debugging and analysis
- **Resume support**: `run_fill_batch` with `resume=True` skips tasks with existing traces

---

## Data Collection Pipeline

The `data_collection/` sub-module mines commits from git repositories and filters them for FIM training:

1. **Collect commits**: Scan repositories for commits with code changes
2. **Filter by stats**: Apply thresholds (line count, file count, hunk count)
3. **Dependency analysis**: (Optional) Check for def-use dependency chains between hunks
4. **Benchmark filters**: Prepare benchmark-specific filtered datasets

Output is used to build `FillRequest` instances for the pipeline.

---

## Installation

### Prerequisites

- Python 3.10+
- Git
- Docker (optional)

### Setup

```bash
# Clone the repository
git clone <repo-url>
cd CoEdAgent

# Create virtual environment (recommended with uv)
uv venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Install dependencies
uv sync

# Copy and configure environment
cp .env.example .env
```

### Using pip (alternative)

```bash
pip install -e .
```

---

## Configuration

Copy `.env.example` to `.env` and set the following:

| Variable | Description |
|---|---|
| `OPENAI_API_KEY` | OpenAI API key |
| `ANTHROPIC_API_KEY` | Anthropic API key |
| `VOYAGE_API_KEY` | VoyageAI API key (for embeddings) |
| `OPENROUTER_API_KEY` | OpenRouter API key (alternative provider) |
| `MOATLESS_DIR` | Storage directory for moatless state |

### Mining Thresholds

Configure commit mining thresholds in `moatless/fim/data_collection/config/settings.py`:

| Parameter | Description |
|---|---|
| `MIN_SOURCE_LOC` / `MAX_SOURCE_LOC` | Source code line change range |
| `MIN_SOURCE_FILES` / `MAX_SOURCE_FILES` | Source files modified range |
| `MIN_SOURCE_HUNKS` / `MAX_SOURCE_HUNKS` | Hunks per commit range |
| `REQUIRE_TEST_CHANGE` | Require test file changes in commit |
| `SOURCE_EXTENSIONS` | Supported source file extensions |

---

## Usage

### Quick Start

```python
import asyncio
from moatless.fim import FillRequest, run_fill_task
from moatless.completion.tool_call import ToolCallCompletionModel

model = ToolCallCompletionModel(
    model="openai/gpt-4o",
    temperature=0.0,
)

task = FillRequest(
    task_id="example_001",
    model_name="gpt-4o",
    repo_path="/path/to/repo",
    file_path="src/module.py",
    start_line=100,
    end_line=120,
    max_iterations=30,
)

result = asyncio.run(run_fill_task(task, model))
print(f"Success: {result.success}, Completion: {result.completion[:100]}")
```

### Batch Execution

```bash
python -m moatless.fim.run_fill_batch \
    --input data/tasks.jsonl \
    --output results/output.jsonl \
    --max_concurrency 4 \
    --resume
```

### Data Collection

```bash
# Collect and filter commits
python -m moatless.fim.data_collection.main_collect_commits \
    --repos_dir /repos \
    --output data/commits.jsonl

# Run post-processing
python -m moatless.fim.data_collection.run_main_PP \
    --input data/commits.jsonl \
    --output data/processed.jsonl
```


## Project Structure

```
CoEdAgent/
├── moatless/
│   ├── fim/                     ← Custom FIM pipeline (our work)
│   │   ├── data_collection/     Commit mining and filtering
│   │   ├── pipeline.py          Core execution
│   │   ├── schema.py            Data types
│   │   ├── prompt.py            Prompt engineering
│   │   ├── dataset.py           Data I/O
│   │   └── ...
│   └── ...                      Original moatless code
├── tests/
├── notebooks/
├── docker/
├── docker-compose.yml
├── pyproject.toml
└── README.md
```

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

The original [moatless-tools](https://github.com/aorwall/moatless-tools) framework is also MIT-licensed.
