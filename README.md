# research-peft

A research benchmark comparing Parameter-Efficient Fine-Tuning (PEFT) methods under identical hardware constraints.

**Base model:** [Qwen2.5-1.5B](https://huggingface.co/Qwen/Qwen2.5-1.5B) · **Dataset:** AG News · **GPU:** NVIDIA A10 (24 GB)

## PEFT Methods

| Method       | File                          |
|--------------|-------------------------------|
| LoRA         | `src/peft/lora.py`            |
| AdaLoRA      | `src/peft/adalora.py`         |
| IA³          | `src/peft/1a3.py`             |
| DoRA         | `src/peft/dora.py`            |
| Prefix Tuning| `src/peft/prefix_tuning.py`   |

## Setup

```sh
uv sync                    # install dependencies
pre-commit install         # install git hooks
```

Python 3.13 only. Deps in `requirements.txt`.

## Development

```sh
ruff check src/ tests/ scripts/   # lint
ruff format src/ tests/ scripts/  # format
pre-commit run --all-files         # full pre-commit suite
```

## Experiment Design

See [`docs/experiment_design.md`](docs/experiment_design.md) for full specification: research questions, hypotheses, controlled variables, metrics, and naming conventions.

## Output Layout

```
outputs/<experiment_name>/
├── config.yaml
├── metrics.json
├── training.csv
├── checkpoint/
└── plots/
```
