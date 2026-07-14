# research-peft

Compare PEFT methods (LoRA, AdaLoRA, IA³, DoRA, Prefix Tuning) under identical hardware constraints. Base model: **Qwen2.5-1.5B**.

## Status

Scaffold — all `src/`, `scripts/`, `tests/`, `datasets/` `.py` files are empty stubs. **No implementation exists.**

## Key Specs

| Item            | Value                     |
|-----------------|---------------------------|
| PyTorch         | 2.7.0                     |
| Transformers    | 5.7.0                     |
| PEFT            | 0.18.0                    |
| CUDA            | 11.8                      |
| GPU             | NVIDIA A10 (24 GB GDDR6)  |
| Precision       | BF16 / FP16               |

## Commands

```sh
uv sync                          # install deps (uv, not pip)
ruff check src/ tests/ scripts/  # lint
ruff format src/ tests/ scripts/ # format
pre-commit run --all-files       # full pre-commit suite
```

Deps in `requirements.txt` (`dependencies = []` in pyproject.toml). Python 3.13 only. No test framework or CI configured yet.

## What to build

Full spec in `docs/experiment_design.md`. Key facts:

- **Model:** Qwen2.5-1.5B, **Dataset:** AG News, **GPU:** NVIDIA A10 24 GB
- **Metrics:** accuracy, F1, val loss, peak GPU memory, training time, trainable params, checkpoint size
- **Composite:** PTP (accuracy / trainable params), Memory Efficiency (accuracy / peak GPU memory), Training Efficiency (accuracy / training time)
- **Experiment naming:** `<model>_<dataset>_<peft>_<seed>` → `outputs/qwen15b_agnews_lora_seed1/`
- **Fixed constraints:** BF16/FP16, gradient checkpointing, max seq len 512

## Conventions

- Ruff: line-length 88, double quotes, target py313
- Modules mirror `docs/experiment_design.md` structure — keep them aligned
- `config/` YAML files are stubs to hydrate as implementations land
- W&B logger exists (`src/logging/wand_b_logger.py`) but not configured
- DeepSpeed 0.19.2 pinned for potential memory profiling use

## Output structure

```
outputs/<experiment_name>/
├── config.yaml          # experiment config snapshot
├── metrics.json         # all metrics
├── training.csv         # per-epoch/step log
├── checkpoint/          # model weights
└── plots/               # figures
```
