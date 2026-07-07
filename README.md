# CoEdEvaluationFrameworks

> Frameworks for code change propagation — tools for mining, filtering, predicting, and propagating coordinated code changes across repositories.

![Python](https://img.shields.io/badge/python-3.10+-blue)
![License](https://img.shields.io/badge/license-MIT-green)

---

## Overview

Code change propagation is the task of determining what other locations in a codebase need to be updated when a given code change (a *root hunk*) is applied. This repository provides two approaches for this task:

| Module | Description |
|---|---|
| **[Pipeline](./pipeline/)** | Two-stage LLM pipeline: identifies impacted locations, then generates coordinated edits. Supports oracle context (ground-truth targets) and auto mode (dependency graph). |
| **[Agent](./agent/)** | Agent-based FIM (Fill-in-Middle) system built on [moatless-tools](https://github.com/aorwall/moatless-tools) that explores the repository and generates edits autonomously. |

Both approaches share the same dataset format and can be evaluated with the same benchmarks.

---

## Repository Structure

```
CoEdEvaluationFrameworks/
├── pipeline/              Two-stage LLM change propagation prediction pipeline
│   ├── main.py           Pipeline entry point
│   ├── train_make_main.py  Training dataset construction
│   ├── src/              Core pipeline logic
│   └── config/           Configuration files
│
├── agent/                FIM completion agent (based on moatless-tools)
│   ├── moatless/fim/     Custom FIM data collection pipeline
│   └── docker/           Docker configuration for sandboxed execution
│
└── README.md             This file
```

---

## Getting Started

### Pipeline

```bash
cd pipeline
cp .env.example .env   # configure API keys
python main.py \
    --data_path data/dataset.jsonl \
    --repos_base /repos \
    --output_path results/run.jsonl
```

See [Pipeline README](./pipeline/README.md) for full documentation.

### Agent

```bash
cd agent
cp .env.example .env   # configure API keys
uv sync                 # install dependencies
python -m moatless.fim.run_fill_batch \
    --input data/tasks.jsonl \
    --output results/output.jsonl
```

See [Agent README](./agent/README.md) for full documentation.

---

## License

[MIT](LICENSE)

The Agent module is based on [moatless-tools](https://github.com/aorwall/moatless-tools) (MIT-licensed).
