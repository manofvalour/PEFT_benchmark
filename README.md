# research-peft

A research benchmark comparing Parameter-Efficient Fine-Tuning (PEFT) methods under identical training and hardware constraints.

**Base model:** [Qwen2.5-1.5B](https://huggingface.co/Qwen/Qwen2.5-1.5B)
**Dataset:** [`yahma/alpaca-cleaned`](https://huggingface.co/datasets/yahma/alpaca-cleaned)
**Hardware:** 4× NVIDIA A10G (24 GB each) via Modal

The current experiment report is in [`docs/results.md`](docs/results.md). The analysis notebook is [`notebooks/experiment_analysis.ipynb`](notebooks/experiment_analysis.ipynb).

## PEFT Methods

| Method | Implementation | Config section |
|---|---|---|
| LoRA | `peft.LoraConfig` | `lora` |
| DoRA | `peft.LoraConfig(use_dora=True)` | `dora` |
| AdaLoRA | `peft.AdaLoraConfig` | `adalora` |
| IA3 | `peft.IA3Config` | `ia3` |

All methods are selected in [`scripts/trainer.py`](scripts/trainer.py). Prefix Tuning is not implemented.

## Setup

```sh
uv pip install -r requirements.txt   # install dependencies
pre-commit install                   # install git hooks
```

Dependencies live in [`requirements.txt`](requirements.txt); `pyproject.toml` intentionally has `dependencies = []`. Do **not** run bare `uv sync` in this repository: it would resolve the empty project and remove the installed packages from `.venv`. Python 3.13 is used locally, while Modal images use Python 3.12.

`liger-kernel` is installed in the Modal images but is **not** in `requirements.txt`. For local runs, either install it or disable it:

```sh
uv run main.py --config scripts/config_file/config.json --finetune_type lora \
  --overrides training.use_liger_kernel=false
```

## Quick Start

Train one PEFT method:

```sh
uv run main.py --config scripts/config_file/config.json --finetune_type lora
```

Supported values for `--finetune_type` are `lora`, `dora`, `adalora`, and `ia3`.

Override any config value with dot notation:

```sh
uv run main.py --config scripts/config_file/config.json --finetune_type lora \
  --overrides training.learning_rate=1e-4 lora.r=32
```

Dataset splits are cached under `dataset/<dataset_name>/` after the first run. The cache is reused automatically; delete that directory to regenerate splits.

## Evaluation

Run the generation benchmark as a script (the module uses bare imports):

```sh
python scripts/benchmark.py \
  --model_name Qwen/Qwen2.5-1.5B \
  --config_path scripts/config_file/config.json \
  --adapter_path outputs/checkpoints/qwen2.5-1.5b-lora_finetune \
  --output_dir outputs/reports/lora.json
```

> **Known issue:** `scripts/benchmark.py` currently calls `load_model_with_adapter(..., adapter_path=None)`, so the adapter is not loaded. Generation metrics in `outputs/reports/*.json` should be treated as provisional until this is fixed.

## Hyperparameter Sweeps

```sh
uv run scripts/sweep.py --method lora --num_trials 20 --backend modal
uv run scripts/sweep.py --method all --search grid --backend local
```

Sweep configuration and per-method search spaces live in [`scripts/config_file/sweep_config.yaml`](scripts/config_file/sweep_config.yaml). Trial configs are written to `scripts/config_file/sweep_trial_<method>_<id>.json`, and summaries to `outputs/sweep_results/<method>_results.json`.

For the Modal backend, deploy the training app first:

```sh
modal deploy modal_train.py
```

## Modal

```sh
modal run modal_train.py          # single training job
modal deploy modal_train.py       # required before sweep.py --backend modal
modal run modal_eval_run.py --adapter-path ... --model-name ... --report-dir ...
```

Modal jobs require `modal` authentication and the `wandb-secret` secret.

## Development

```sh
ruff check scripts/ main.py      # lint
ruff format scripts/ main.py     # format
pre-commit run --all-files       # full pre-commit suite
```

There is no test suite; verify changes with lint plus a short training or evaluation run.

The repository is not currently ruff-clean (`scripts/sweep.py`, `template.py`, and notebooks already have violations). Avoid repo-wide `ruff format` or `ruff check --fix`; target only the files you change.

## Project Structure

There is no `src/` directory. Code lives in `scripts/` and the repository root.

| Path | Purpose |
|---|---|
| `main.py` | CLI entry point: dataset cache/load, config overrides, calls `scripts.trainer.train()` |
| `scripts/trainer.py` | Model/tokenizer loading, PEFT selection, `SFTTrainer` run, metrics |
| `scripts/evaluate.py` | Metrics and W&B efficiency callbacks |
| `scripts/dataset_utils.py` | Prompt-masking collator and train/val/test split |
| `scripts/benchmark.py` | Generation benchmark and `eval_main` CLI |
| `scripts/inference.py` | Load base + adapter, generate, merge and save |
| `scripts/sweep.py` | Hyperparameter sweep controller |
| `scripts/config.py` | `ModelConfig` and `DataConfig` dataclasses |
| `scripts/config_file/config.json` | Training and PEFT hyperparameters |
| `scripts/config_file/sweep_config.yaml` | Sweep search spaces |
| `scripts/config_file/ds_config.json` | DeepSpeed ZeRO stage 0 config |
| `scripts/config_file/accelerate_cfg.yaml` | Accelerate launcher config |
| `modal_train.py` | Modal training app and sweep function |
| `modal_eval_run.py` | Modal evaluation job |
| `modal_sweep_run.py` | Standalone duplicate sweep app |

## Output Layout

```
outputs/
├── checkpoints/qwen2.5-1.5b-<peft>_finetune/   # normal run adapter + tokenizer
├── checkpoints/sweep_<peft>_trial<id>/         # sweep run adapters
├── sweep_results/<method>_results.json         # sweep summaries
├── reports/                                    # generation evaluation JSON
├── figures/                                    # analysis plots
└── wnb_report/                                 # exported W&B metrics
```

## Experiment Design

See [`docs/experiment_design.md`](docs/experiment_design.md) for the research questions, hypotheses, controlled variables, and metric definitions. Verified constraints:

- Dataset: `yahma/alpaca-cleaned`
- Max sequence length: 1,024
- Precision: bf16, gradient checkpointing enabled
- DeepSpeed: ZeRO stage 0 (no offloading)
- Metrics: loss/perplexity, ROUGE/BLEU/BERTScore, trainable parameters, GPU memory, training time/throughput, checkpoint size
- Composite metrics: PTP, Memory Efficiency, Training Efficiency

## Notes

- `use_flash_attn: true` selects SDPA (`attn_implementation="sdpa"`), not FlashAttention.
- DoRA is configured through `LoraConfig(use_dora=True)` under the `dora` config section.
- W&B logging is enabled via `report_to: "wandb"`; log in or change `report_to` for offline runs.
- The experiment naming convention in the design doc (`<model>_<dataset>_<peft>_<seed>`) is aspirational; actual output paths use the fixed names above.
