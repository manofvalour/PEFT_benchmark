## Research Question
- How do Different PEFT Methods Trade off Adaptation Quality, Trainable Parameter, and GPU Memory Under the Same Hardware Constraints?

## Hypothesis
- **Primary Hypothesis**: Under identical hardware and training conditions, no single PEFT method will optimize adaptation quality, trainable parameters, and GPU memory simultaneously. Instead, each method will exhibit distinct trade-offs, forming a Pareto frontier where improvements in one objective come at the expense of another.

- **H2**: AdaLoRA will achieve similar adaptation quality to LoRA while using fewer trainable parameters.

- **H3**: IA³ will use the least GPU memory but may show reduced adaptation quality on more complex tasks.

## Variable Definition
The different independent, controlled, and dependent variables
### Independent Variables
- LORA
- AdaLORA
- IA**3
- DORA
- Prefix Tuning

### Controlled Variables
parameters that remain constant during experimentation:
- Model
- Dataset
- Learning rate
- Batch size
- Epochs
- Optimizer
- Sequence length
- Random seed
- GPU
- CUDA version

### Dependent Variables
what we'll measure:
- Accuracy
- F1
- Validation loss
- Peak GPU memory
- Training time
- Trainable parameters
- Checkpoint size

## Scope of the Benchmark
### Objective of the Benchmark Experiment:
Evaluating how different Parameter-Efficient Fine-Tuning (PEFT) methods trade off adaptation quality, trainable parameters, and GPU memory under identical training conditions.

### Included in the experiment
- Comparison of multiple PEFT methods
- Same base model
- Same dataset(s)
- Same optimizer
- Same training configuration
- Same hardware
- Same evaluation protocol

### Excluded from the experiment
- full finetuning

## Base model
- Qwen2.5-1.5B

## dataset
- AGnews
-yahma/alpaca-cleaned

## PEFT methods to compare
These are the different PEFT methods to compare:
- LORA
- AdaLORA
- IA**3
- DORA
- Prefix Tuning

## Evaluation Metrics
These are the evaluation metrics:
### A. For Adaptation Quality
- Training Loss
- Validation Loss

**Evaluation metrics for classification problems:**
- Accuracy
- Precision
- Recall
- F1

**Evaluation metrics for instruction tuning:**
- ROUGE
- BLEU

### B. Parameter Efficiency
- Trainable parameters
- Total parameters
- Percentage trainable ((trainable parameter)/(Total parameters)) * 100

### c. Memory Efficiency
- Peak GPU memory
- Average GPU memory
- Reserved memory
- Allocated memory

### D. Computational Efficiency
- Training time
- Epoch time
- Tokens/sec
- Samples/sec
- Checkpoint size

### E. Composite Metrics
- Performance per trainable parameters(PTP) ==> Accuracy/trainable parameters
- Memory Efficiency ==> Accuracy/Peak GPU memory
- Training Efficiency ==> Accuracy/Training Time

## Hardware Constraints
All experiments will be executed on the same hardware using identical software versions. GPU memory, precision, batch size, sequence length, optimizer, learning rate, and random seeds will remain fixed across all experiments.

| Component       | Specification           |
|-----------------|-------------------------|
| GPU             | Nvidia A10 24GB GDDR6   |
| CPU             |                         |
| RAM             | 4GB                     |
| CUDA            | V11.8.0                 |
| PyTorch         | V2.7.0                  |
| Transformers    | V5.7.0                  |
| PEFT            | V0.18.0                 |

### Fixed Constraints
- Maximum GPU memory: 24 GB
- Precision: BF16 (or FP16)
- Gradient checkpointing: Enabled
- Maximum sequence length: 1024
- Batch size: Fixed
- Gradient accumulation: Fixed

## Experiment naming convention
Experiment naming follows the format below:
    <model>_<dataset>_<peft>_<seed>

## output directory convention

```
outputs/
├── qwen15b_agnews_lora_seed1/
│   ├── config.yaml
│   ├── metrics.json
│   ├── training.csv
│   ├── checkpoint/
│   └── plots/
├── qwen15b_agnews_lora_seed2/
└── qwen15b_agnews_ia3_seed1/
```

## Cofiguration naming
```
configs/
models/
    qwen15b.yaml
datasets/
    agnews.yaml
peft/
    lora.yaml
trainer/
    default.yaml
```
