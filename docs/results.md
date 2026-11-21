# Experiment Report: Parameter-Efficient Fine-Tuning of Qwen2.5-1.5B

**Report date:** 2026-11-21
**Experiment:** LoRA vs DoRA vs AdaLoRA vs IA3 under identical training conditions
**Base model:** `Qwen/Qwen2.5-1.5B`
**Dataset:** `yahma/alpaca-cleaned`
**Hardware:** 4× NVIDIA A10G (24 GB each) on Modal
**Analysis notebook:** [`notebooks/experiment_analysis.ipynb`](../notebooks/experiment_analysis.ipynb)

---

## 1. Executive Summary

This experiment compares four parameter-efficient fine-tuning (PEFT) methods on Qwen2.5-1.5B using the Alpaca-cleaned instruction dataset. All methods were trained with the same base model, optimizer, schedule, precision, sequence length, and hardware configuration.

The main findings are:

1. **IA3 is the most parameter-efficient method by a wide margin.** It trains 0.85M parameters (0.055% of the model), 21.7× fewer than LoRA, and produces a 1.7 MB adapter. Its generation quality is competitive with the other methods in the stored evaluation reports.
2. **AdaLoRA produces the best generation-quality scores in the stored reports** (BERTScore F1 0.8822, ROUGE-1 0.4310, BLEU 11.81), but uses 2× the trainable parameters of LoRA and a 2× larger adapter.
3. **LoRA is the fastest method** (62.0 min, 8,101 tokens/s) and remains the strongest practical baseline for rapid iteration.
4. **DoRA did not provide a quality advantage over LoRA in this run.** It used 1.75 GB more peak GPU memory, was 1.72× slower, and produced nearly identical quality scores. However, DoRA used a different dropout value (0.1 vs 0.05), so this is not a clean DoRA-vs-LoRA ablation.
5. **No single method dominates all objectives.** The Pareto trade-off hypothesis is supported: quality, parameter count, memory, and speed favor different methods.

> **Important validity caveat:** The training and efficiency metrics come from W&B logs and are reliable. The generation-quality metrics (ROUGE, BERTScore, BLEU) are taken from `outputs/reports/*.json`, but the current `scripts/benchmark.py` calls `load_model_with_adapter(..., adapter_path=None)`, which loads only the base model. Unless the stored reports were produced by an earlier or different evaluation harness, they may describe base-model generations rather than adapter generations. Treat the generation-quality rankings as **provisional** until the benchmark is re-run with the adapter actually loaded. See Section 9.

---

## 2. Research Question and Hypotheses

### Research question

How do different PEFT methods trade off adaptation quality, trainable parameters, GPU memory, and training cost under identical hardware and training constraints?

### Hypotheses

| ID | Hypothesis | Verdict |
|---|---|---|
| H1 | No single PEFT method optimizes adaptation quality, parameter count, memory, and speed simultaneously. | **Supported** |
| H2 | AdaLoRA achieves quality similar to LoRA while using fewer trainable parameters. | **Rejected** |
| H3 | IA3 uses the least GPU memory but may show reduced adaptation quality. | **Rejected** |

---

## 3. Experimental Setup

### 3.1 Model and data

| Item | Value |
|---|---|
| Base model | `Qwen/Qwen2.5-1.5B` |
| Dataset | `yahma/alpaca-cleaned` |
| Train examples used | 35,197 |
| Validation examples used | 4,399 |
| Test examples used | 4,400 |
| Max sequence length | 1,024 |
| Precision | bf16, TF32 enabled |
| Attention implementation | SDPA (`use_flash_attn: true` maps to `attn_implementation="sdpa"`) |
| Gradient checkpointing | Enabled |
| DeepSpeed | ZeRO stage 0, no offloading |

The experiments consumed the cached split in `dataset/yahma/alpaca-cleaned/`. The cache is reused automatically whenever that directory exists, so the exact split is the cached one rather than a fresh split from the current `test_split`/`val_split` settings. Delete the cache to regenerate splits.

### 3.2 Training configuration

| Hyperparameter | Value |
|---|---|
| Epochs | 4 |
| Per-device train batch size | 16 |
| Gradient accumulation steps | 8 |
| Effective global batch size | 512 (16 × 8 × 4 GPUs) |
| Learning rate | 2e-4 |
| Optimizer | AdamW (`adamw_torch`) |
| LR schedule | Cosine |
| Warmup steps | 30 |
| Weight decay | 0.01 |
| Max gradient norm | 1.0 |
| Evaluation interval | Every 20 steps |
| Save interval | Every 40 steps |
| Early stopping | Patience 3 |
| Logging | Weights & Biases (`report_to: "wandb"`) |

### 3.3 PEFT configurations

| Method | Configuration |
|---|---|
| **LoRA** | r=16, alpha=32, dropout=0.05, bias=none, targets: q/k/v/o/gate/up/down projections |
| **DoRA** | r=16, alpha=32, dropout=0.1, `use_dora=true`, same target modules as LoRA |
| **AdaLoRA** | init_r=32, target_r=16, tinit=20, tfinal=220, deltaT=10, total_step=276, same target modules; saved adapter reports final r=8 |
| **IA3** | target modules: q/k/v/o/gate/up/down projections; `feedforward_modules=["down_proj"]` |

LoRA and DoRA share the same rank, alpha, and target modules, but their dropout differs (0.05 vs 0.1). That difference is a confound for the DoRA-vs-LoRA comparison.

### 3.4 Evaluation protocol

- **Optimization quality:** final training loss, validation loss, and perplexity derived as `exp(eval_loss)`.
- **Generation quality:** ROUGE-1/2/L, BERTScore F1 (`roberta-large`), and corpus BLEU on the test split.
- **Generation settings:** prompt template `"{instruction} {input}\n\n### Response:\n"`, `max_new_tokens=256`, `batch_size=70`, `temperature=0.1`, `do_sample=True`, max input length 1,024.
- **Parameter efficiency:** trainable parameters, trainable percentage, total parameters, adapter file size.
- **Memory efficiency:** peak, allocated, average, and reserved GPU memory.
- **Training efficiency:** wall-clock runtime, epoch time, tokens/s, samples/s, and global steps.
- **Composite metrics:** PTP, memory efficiency, and training efficiency, using BERTScore F1 as the quality proxy and min-max normalization across methods.

---

## 4. Main Results

### 4.1 Summary table

| Metric | LoRA | DoRA | AdaLoRA | IA3 |
|---|---:|---:|---:|---:|
| Trainable parameters | 18,464,768 | 19,109,888 | 36,935,808 | **852,992** |
| Trainable % | 1.1820% | 1.2228% | 2.3367% | **0.0552%** |
| Total parameters | 1,562,179,072 | 1,562,824,192 | 1,580,650,308 | 1,544,567,296 |
| Adapter file size (MB) | 37.0 | 38.3 | 73.9 | **1.7** |
| Train loss | **1.0312** | 1.0313 | 1.0704 | 1.0657 |
| Eval loss | 1.0490 | **1.0489** | 1.0577 | 1.0598 |
| Perplexity (`exp(eval_loss)`) | 2.8549 | **2.8546** | 2.8798 | 2.8857 |
| BERTScore F1 | 0.8816 | 0.8820 | **0.8822** | 0.8821 |
| ROUGE-1 | 0.4278 | 0.4296 | **0.4310** | 0.4290 |
| ROUGE-2 | 0.1977 | 0.1997 | **0.2005** | 0.1985 |
| ROUGE-L | 0.3019 | 0.3044 | **0.3047** | 0.3026 |
| BLEU | 11.6675 | 11.7105 | **11.8094** | 11.7402 |
| Peak GPU memory (GB) | 7.039 | 8.785 | **7.017** | 7.325 |
| Allocated GPU memory (GB) | 3.366 | 3.374 | 3.623 | **3.118** |
| Average GPU memory (GB) | 3.366 | 3.374 | 3.624 | **3.118** |
| Reserved GPU memory (GB) | **13.455** | 20.091 | 14.097 | 15.712 |
| Training time (min) | **62.0** | 106.8 | 87.5 | 76.4 |
| Epoch time (min) | 21.1 | 36.4 | 22.3 | **19.4** |
| Throughput (tokens/s) | **8,101** | 4,536 | 6,123 | 7,088 |
| Throughput (samples/s) | **3.96** | 2.22 | 2.99 | 3.46 |
| Global steps | 200 | 200 | 276 | 276 |

Adapter file sizes are decimal MB (10⁶ bytes). The notebook reports MiB instead (35.3, 36.5, 70.5, 1.6 MiB). The W&B checkpoint-size metric was logged only for AdaLoRA (1.07 GB) and IA3 (0.49 GB); adapter file size is the comparable deployment metric for all four methods.

### 4.2 Parameter efficiency

![Trainable parameters and percentage](../outputs/figures/param_efficiency.png)

IA3 is the clear parameter-efficiency winner:

- 0.85M trainable parameters, or 0.0552% of the model.
- 21.7× fewer parameters than LoRA.
- 43.3× fewer parameters than AdaLoRA.
- 1.7 MB adapter versus 37.0 MB for LoRA and 73.9 MB for AdaLoRA.

LoRA and DoRA are close to each other (18.46M vs 19.11M), while AdaLoRA has the largest trainable set because it starts with `init_r=32` and allocates rank during training. The saved AdaLoRA adapter reports `r=8`, reflecting the post-allocation rank.

### 4.3 Adaptation quality

![Loss and training time](../outputs/figures/training_performance.png)

**Optimization metrics:**

- LoRA and DoRA achieve the lowest training loss (1.0312 and 1.0313).
- DoRA and LoRA also achieve the lowest validation loss (1.0489 and 1.0490).
- AdaLoRA and IA3 have higher training and validation losses (1.0704/1.0577 and 1.0657/1.0598).

**Generation metrics (provisional; see caveat):**

- AdaLoRA has the highest BERTScore F1 (0.8822), ROUGE-1 (0.4310), ROUGE-L (0.3047), and BLEU (11.8094).
- IA3 is second on BERTScore F1 (0.8821) and BLEU (11.7402), but third on ROUGE-1/ROUGE-L.
- DoRA and LoRA are slightly lower on BERTScore and ROUGE, with LoRA lowest on most generation metrics.
- The absolute spread is small: BERTScore F1 spans only 0.0006 and ROUGE-1 spans 0.0032 across methods.

The optimization and generation metrics disagree: LoRA/DoRA have the best validation loss, while AdaLoRA has the best generation scores. This is a reminder that token-level likelihood and sequence-level generation quality measure different things.

### 4.4 Memory efficiency

![GPU memory usage](../outputs/figures/memory_efficiency.png)

All methods fit comfortably within the 24 GB per-GPU limit:

- Peak memory ranges from 7.017 GB (AdaLoRA) to 8.785 GB (DoRA).
- DoRA uses 1.746 GB more peak memory than LoRA despite having only 3.5% more trainable parameters.
- Reserved memory is much higher than allocated memory: DoRA reserves 20.09 GB, versus 13.46–15.71 GB for the others.
- IA3 has the lowest allocated/average memory (3.118 GB), but its peak memory is slightly higher than LoRA and AdaLoRA.

The DoRA memory overhead is consistent with its magnitude-direction decomposition, which adds computation and activation memory relative to plain LoRA.

### 4.5 Training efficiency

- LoRA is fastest: 62.0 min, 8,101 tokens/s, 3.96 samples/s.
- IA3 is second: 76.4 min, 7,088 tokens/s.
- AdaLoRA is third: 87.5 min, 6,123 tokens/s.
- DoRA is slowest: 106.8 min, 4,536 tokens/s.

LoRA is 1.72× faster than DoRA and 1.41× faster than AdaLoRA. However, LoRA and DoRA logged 200 global steps while AdaLoRA and IA3 logged 276, so wall-clock and throughput comparisons are not perfectly controlled across methods.

### 4.6 Adapter size

![Adapter size](../outputs/figures/adapter_size.png)

| Method | Adapter file size (MB) | Relative to LoRA |
|---|---:|---:|
| IA3 | 1.7 | 0.047× |
| LoRA | 37.0 | 1.00× |
| DoRA | 38.3 | 1.04× |
| AdaLoRA | 73.9 | 2.00× |

IA3 is roughly 21× smaller than LoRA and 43× smaller than AdaLoRA, making it the most attractive option for storage- or bandwidth-constrained deployment.

### 4.7 Generation quality (provisional)

![Generation quality](../outputs/figures/generation_quality.png)

| Method | ROUGE-1 | ROUGE-2 | ROUGE-L | BERTScore F1 | BLEU |
|---|---:|---:|---:|---:|---:|
| LoRA | 0.4278 | 0.1977 | 0.3019 | 0.8816 | 11.6675 |
| DoRA | 0.4296 | 0.1997 | 0.3044 | 0.8820 | 11.7105 |
| AdaLoRA | **0.4310** | **0.2005** | **0.3047** | **0.8822** | **11.8094** |
| IA3 | 0.4290 | 0.1985 | 0.3026 | 0.8821 | 11.7402 |

These numbers should not be treated as definitive adapter-quality measurements until the benchmark harness is corrected and the evaluation is repeated. See Section 9.

---

## 5. Composite Metrics

Composite metrics use BERTScore F1 as the quality proxy:

- **PTP** = BERTScore F1 / trainable parameters
- **Memory Efficiency** = BERTScore F1 / peak GPU memory
- **Training Efficiency** = BERTScore F1 / training time

### 5.1 Raw values

| Method | PTP | Memory Efficiency | Training Efficiency |
|---|---:|---:|---:|
| LoRA | 4.7745e-08 | 0.125245 | 0.000237 |
| DoRA | 4.6154e-08 | 0.100401 | 0.000138 |
| AdaLoRA | 2.3885e-08 | 0.125714 | 0.000168 |
| IA3 | **1.0341e-06** | 0.120427 | 0.000192 |

### 5.2 Min-max normalized values

Each column is normalized independently to [0, 1], where 1 is best and 0 is worst.

| Method | PTP | Memory Efficiency | Training Efficiency |
|---|---:|---:|---:|
| LoRA | 0.0236 | 0.9814 | **1.0000** |
| DoRA | 0.0220 | 0.0000 | 0.0000 |
| AdaLoRA | 0.0000 | **1.0000** | 0.3049 |
| IA3 | **1.0000** | 0.7911 | 0.5514 |

![Normalized composite metrics](../outputs/figures/composite_metrics.png)

IA3 dominates PTP because of its extremely low parameter count. AdaLoRA has the best memory efficiency, driven by its slightly lower peak memory and high quality score. LoRA has the best training efficiency because it is the fastest method.

---

## 6. Pareto Analysis

![Pareto trade-offs](../outputs/figures/pareto_analysis.png)

No single method dominates across all axes:

- **AdaLoRA** has the best generation quality and the lowest peak memory, but the most parameters and the largest adapter.
- **IA3** has by far the fewest parameters and a tiny adapter, with quality close to the best methods, but it is not the lowest-memory or fastest method.
- **LoRA** is the fastest and has the lowest training loss, but it has the lowest generation scores in the stored reports.
- **DoRA** is the slowest and highest-memory method, with quality nearly identical to LoRA.

The Pareto frontier therefore depends on the objective: AdaLoRA for quality, IA3 for parameter/storage efficiency, and LoRA for speed.

---

## 7. Hypothesis Evaluation

### H1: No single PEFT method optimizes all objectives simultaneously

**Verdict: Supported.**

Each method occupies a different position in the trade-off space:

- AdaLoRA: best quality and lowest peak memory, but most parameters and largest adapter.
- IA3: best parameter efficiency and smallest adapter, with competitive quality.
- LoRA: fastest training and lowest train loss, but lowest generation scores.
- DoRA: no clear advantage in this run; slower and higher memory than LoRA for similar quality.

### H2: AdaLoRA achieves quality similar to LoRA with fewer parameters

**Verdict: Rejected.**

AdaLoRA achieves slightly better generation scores than LoRA (BERTScore F1 0.8822 vs 0.8816; ROUGE-1 0.4310 vs 0.4278), but it uses **more** parameters, not fewer: 36.94M vs 18.46M (2.0×). Its adapter is also 2.0× larger. AdaLoRA's advantage is quality per step, not quality per parameter.

### H3: IA3 uses the least GPU memory but may show reduced adaptation quality

**Verdict: Rejected.**

IA3 is the most parameter-efficient method, but:

- It does **not** have the lowest peak GPU memory. AdaLoRA is lowest (7.017 GB), followed by LoRA (7.039 GB), IA3 (7.325 GB), and DoRA (8.785 GB).
- It does **not** have the lowest generation quality. LoRA has the lowest BERTScore F1 in the stored reports (0.8816); IA3 is second-highest (0.8821).

IA3's real advantage is parameter and storage efficiency, not memory or quality superiority.

---

## 8. Discussion

### 8.1 IA3 is the efficiency specialist

IA3 trains only 0.85M parameters and produces a 1.7 MB adapter, roughly 21× smaller than LoRA and 43× smaller than AdaLoRA. Its generation scores are close to the best methods in the stored reports. For deployment scenarios where adapter size, download size, or parameter count dominate, IA3 is the strongest candidate.

### 8.2 AdaLoRA buys quality with parameters and adapter size

AdaLoRA produces the highest BERTScore, ROUGE, and BLEU scores in the stored reports. It also has the lowest peak GPU memory, which is counterintuitive given its larger parameter count. The cost is a 2× larger trainable set and a 2× larger adapter than LoRA. If the quality difference is real and reproducible, AdaLoRA is the quality-oriented choice; if it is within noise, the extra parameters are hard to justify.

### 8.3 DoRA is not worth it under this configuration

DoRA uses 1.75 GB more peak memory and is 1.72× slower than LoRA, while its quality scores are essentially identical (BERTScore F1 0.8820 vs 0.8816; ROUGE-1 0.4296 vs 0.4278). The comparison is confounded by the different dropout values (0.1 for DoRA, 0.05 for LoRA), so a controlled DoRA-vs-LoRA run with identical dropout is needed before drawing a firm conclusion. Under the current configuration, DoRA has the worst efficiency profile.

### 8.4 LoRA remains the practical baseline

LoRA is the fastest method, has the lowest training loss, and uses moderate memory. Its generation scores are the lowest in the stored reports, but the differences are small. For rapid iteration and hyperparameter exploration, LoRA is the best default.

### 8.5 Optimization loss and generation quality disagree

LoRA/DoRA have the best validation loss, while AdaLoRA has the best generation metrics. This is not necessarily contradictory: validation loss measures next-token likelihood on the reference text, while ROUGE/BERTScore/BLEU measure overlap or semantic similarity of generated text. A method can model the reference distribution well without producing the most similar generations.

---

## 9. Limitations and Validity Caveats

1. **Generation evaluation may not use the adapters.** The current `scripts/benchmark.py` calls `load_model_with_adapter(model_name, adapter_path=None)` inside `benchmark_result()`, so the documented evaluation path loads only the base model. The stored reports predate the last edit to `benchmark.py`, so their provenance is uncertain. The generation-quality rankings and composite metrics that depend on them should be treated as provisional until the evaluation is re-run with the adapter path actually passed through.
2. **Single seed, no confidence intervals.** Each method was trained once. The quality differences are very small (BERTScore F1 spread 0.0006; ROUGE-1 spread 0.0032), so run-to-run variance could change the ranking.
3. **One model and one dataset.** Results are specific to Qwen2.5-1.5B and `yahma/alpaca-cleaned`. They should not be generalized to other model sizes, architectures, or tasks without further experiments.
4. **Different step counts.** LoRA/DoRA logged 200 global steps; AdaLoRA/IA3 logged 276. Runtime and throughput comparisons therefore include a difference in the amount of work performed. The early-stopping configuration (patience 3) is a plausible contributor, but the exact cause is not logged.
5. **DoRA vs LoRA is confounded by dropout.** DoRA uses dropout 0.1 and LoRA uses 0.05. A controlled comparison should hold dropout fixed.
6. **Cached dataset split.** The runs used the cached split in `dataset/yahma/alpaca-cleaned/` (35,197 train / 4,399 validation / 4,400 test), which is not necessarily reproducible from the current `test_split=0.2` and `val_split=0.5` settings on the full 51,760-example dataset. Delete the cache to regenerate splits.
7. **Missing W&B fields.** Perplexity was logged only for AdaLoRA and IA3; the table derives perplexity as `exp(eval_loss)` for all methods. Checkpoint-size metrics were also logged only for AdaLoRA and IA3, so adapter file size is used for the cross-method comparison.
8. **Adapter size units.** The report uses decimal MB (10⁶ bytes); the notebook uses MiB. The numbers differ slightly but the ranking is unchanged.
9. **No full fine-tuning baseline in the analysis.** A full fine-tuning run exists in the W&B export, but the experiment design excludes full fine-tuning from the PEFT comparison. No conclusions about PEFT-vs-full-FT quality are drawn here.
10. **AdaLoRA final rank differs from the configured target.** The saved adapter reports `r=8`, while the training configuration specifies `init_r=32` and `target_r=16`. Parameter counts describe the saved adapter, not the initial rank.

---

## 10. Recommendations

| Goal | Recommended method | Reason |
|---|---|---|
| Fastest iteration / baseline | **LoRA** | 62.0 min, 8,101 tokens/s, lowest train loss |
| Best generation quality (provisional) | **AdaLoRA** | Highest BERTScore, ROUGE, and BLEU in stored reports |
| Smallest adapter / fewest parameters | **IA3** | 0.85M params, 1.7 MB adapter, competitive quality |
| Lowest peak memory | **AdaLoRA** | 7.017 GB peak, slightly below LoRA and IA3 |
| Avoid unless quality gain is proven | **DoRA** | 1.75 GB more peak memory and 1.72× slower than LoRA for similar quality |

Before finalizing the quality ranking, fix the benchmark adapter-loading issue and re-run generation evaluation for all four adapters. Also consider repeating the experiment with multiple seeds and a controlled DoRA/LoRA dropout comparison.

---

## 11. Reproducibility

### Training

```sh
uv sync

uv run main.py --config scripts/config_file/config.json --finetune_type lora
uv run main.py --config scripts/config_file/config.json --finetune_type dora
uv run main.py --config scripts/config_file/config.json --finetune_type adalora
uv run main.py --config scripts/config_file/config.json --finetune_type ia3
```

### Evaluation (intended flow)

```sh
python scripts/benchmark.py \
  --model_name Qwen/Qwen2.5-1.5B \
  --config_path scripts/config_file/config.json \
  --adapter_path outputs/checkpoints/qwen2.5-1.5b-lora_finetune \
  --output_dir outputs/reports/lora.json
```

Repeat with the corresponding adapter directory and output file for DoRA, AdaLoRA, and IA3. **The current `benchmark_result()` ignores `adapter_path`; fix that before relying on the evaluation outputs.**

### Analysis

Open and run `notebooks/experiment_analysis.ipynb`. It reads:

- `outputs/wnb_report/wnb_report.csv` for training/efficiency metrics.
- `outputs/reports/{lora,dora,adalora,ia3}.json` for generation metrics.
- `outputs/checkpoints/*/adapter_config.json` and `adapter_model.safetensors` for adapter configuration and size.

The notebook regenerates the figures in `outputs/figures/`.

### W&B runs

| Method | W&B run name |
|---|---|
| LoRA | `qwen2.5-1_5b-alpaca-cleaned-lora` |
| DoRA | `qwen2.5-1_5b-alpaca-cleaned-dora` |
| AdaLoRA | `qwen2.5-1_5b-alpaca-cleaned-adalora` |
| IA3 | `qwen2.5-1_5b-alpaca-cleaned-ia3` |

---

## 12. Figures

All figures are generated by `notebooks/experiment_analysis.ipynb` and saved under `outputs/figures/`:

| Figure | File |
|---|---|
| Trainable parameters and percentage | [`param_efficiency.png`](../outputs/figures/param_efficiency.png) |
| Loss and training time | [`training_performance.png`](../outputs/figures/training_performance.png) |
| GPU memory usage | [`memory_efficiency.png`](../outputs/figures/memory_efficiency.png) |
| Adapter size | [`adapter_size.png`](../outputs/figures/adapter_size.png) |
| Generation quality | [`generation_quality.png`](../outputs/figures/generation_quality.png) |
| Normalized composite metrics | [`composite_metrics.png`](../outputs/figures/composite_metrics.png) |
| Pareto trade-offs | [`pareto_analysis.png`](../outputs/figures/pareto_analysis.png) |

---

## 13. Conclusion

Under identical training conditions, the four PEFT methods occupy distinct points on the quality–parameter–memory–speed trade-off surface. IA3 is the parameter and storage efficiency winner, AdaLoRA has the best generation-quality scores in the stored reports, and LoRA is the fastest and most practical baseline. DoRA shows no quality advantage over LoRA in this run and has the worst memory/speed profile.

The primary hypothesis is supported: no single method dominates all objectives. The secondary hypotheses are rejected: AdaLoRA does not achieve its quality with fewer parameters, and IA3 is neither the lowest-memory nor the lowest-quality method.

The most important next step is to fix the generation benchmark so it actually loads each adapter, then re-run the evaluation and, ideally, repeat the experiment across multiple seeds before treating the small quality differences as reliable.
