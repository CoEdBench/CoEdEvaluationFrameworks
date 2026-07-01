# CoEdEvaluationFrameworks

> Frameworks for code change propagation — tools for mining, filtering, predicting, and propagating coordinated code changes across repositories.

![Python](https://img.shields.io/badge/python-3.10+-blue)
![License](https://img.shields.io/badge/license-MIT-green)

---

## Overview

Code change propagation is the task of determining what other locations in a codebase need to be updated when a given code change (a *root hunk*) is applied. This repository provides two complementary frameworks for this task:

| Module | Description |
|---|---|
| **[Pipeline](./pipeline/)** | Data pipeline for mining commits, filtering relevant changes, and constructing benchmarks |
| **[Agent](./agent/)** | Agent-based FIM (Fill-in-Middle) completion system built on [moatless-tools](https://github.com/aorwall/moatless-tools) that explores a repository and generates coordinated edits |

### How They Fit Together

The **Pipeline** processes commits and constructs datasets. The **Agent** consumes those datasets to perform the actual change propagation in an automated, agent-driven manner — exploring the codebase, identifying impacted locations, and generating the necessary edits.

```
Commits → [Pipeline] → Dataset → [Agent] → Predicted edits
```

---

## Repository Structure

```
CoEdEvaluationFrameworks/
├── pipeline/              Data mining, filtering, and benchmark construction
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
