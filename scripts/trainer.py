"""
Fine-tuning Trainer
"""

import json
import logging
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    EarlyStoppingCallback,
)
from trl import SFTConfig
from trl.trainer.sft_trainer import SFTTrainer

from scripts.config import DataConfig, ModelConfig

# Main Training Entry Point
from scripts.dataset_utils import custom_data_collator, train_val_test_split
from scripts.evaluate import EfficiencyMetricsCallback, PerplexityCallback, compute_trainable_params

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Model Loading
def load_base_model(model_cfg: ModelConfig):
    """Load base model"""

    model_kwargs: dict[str, Any] = {
        "pretrained_model_name_or_path": model_cfg.model_name_or_path,
        "trust_remote_code": model_cfg.trust_remote_code,
        "torch_dtype": torch.bfloat16,
    }

    # if model_cfg.cache_dir:
    #   model_kwargs["cache_dir"] = model_cfg.cache_dir
    if model_cfg.use_flash_attn:
        model_kwargs["attn_implementation"] = "sdpa"

    return AutoModelForCausalLM.from_pretrained(**model_kwargs)


# Tokenizer
def load_tokenizer(model_cfg: ModelConfig):
    name = model_cfg.tokenizer_name or model_cfg.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(
        name,
        trust_remote_code=model_cfg.trust_remote_code,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_and_prepare_data(data_cfg: DataConfig) -> tuple:
    """Load and tokenize dataset."""

    ## loading the dataset directly using the Dataset Library
    if data_cfg.dataset_name:
        raw = load_dataset(
            data_cfg.dataset_name,
            data_cfg.dataset_config,
            num_proc=data_cfg.num_proc,
        )

        split = train_val_test_split(
            raw["train"], val_ratio=data_cfg.val_split, test_ratio=data_cfg.test_split
        )

        train_ds, test_ds, val_ds = split["train"], split["test"], split["validation"]

        print(f"Train: {len(train_ds):,} | Val: {len(val_ds):,} | Test: {len(test_ds):,}")
        return train_ds, val_ds, test_ds

    raise ValueError("Provide dataset_name or train_file.")


# Dataset Preparation
def format_instruction(sample: dict, data_cfg: DataConfig) -> str:
    """Format dataset into instruction-following format."""

    instruction = sample.get(data_cfg.prompt_column, "")
    context = sample.get("input", "")
    response = sample.get(data_cfg.response_column, "")

    if context:
        return (
            f"{data_cfg.instruction_template}{instruction}\n\n"
            f"### Context:\n{context}\n\n"
            f"{data_cfg.response_template}{response}"
        )
    return f"{data_cfg.instruction_template}{instruction}\n\n{data_cfg.response_template}{response}"


def train(config_path: str, train_ds, val_ds, finetune_type: str) -> dict:
    """Run fine-tuning and return a metrics dict.

    When ``config_path`` points to a sweep trial config (written by sweep.py),
    the output directory and run name include a trial ID to avoid collisions.
    """
    from peft import get_peft_model

    with Path.open(config_path) as f:
        cfg = json.load(f)

    # loading the configurations
    model_cfg = ModelConfig(**cfg.get("model", {}))
    data_cfg = DataConfig(**cfg.get("data", {}))
    train_args_dict = cfg.get("training", {})

    # Detect sweep trial (config lives in sweep_trial_*.json)
    config_stem = Path(config_path).stem
    is_sweep = config_stem.startswith("sweep_trial_")
    trial_tag = config_stem.replace("sweep_trial_", "") if is_sweep else ""

    # Load base model components
    tokenizer = load_tokenizer(model_cfg)
    base_model = load_base_model(model_cfg)

    if finetune_type == "lora":
        from peft import LoraConfig

        lora_train_args_dict = cfg.get("lora", {})
        config = LoraConfig(**lora_train_args_dict)

    elif finetune_type == "dora":
        from peft import LoraConfig

        dora_train_args_dict = cfg.get("dora", {})
        config = LoraConfig(**dora_train_args_dict)

    elif finetune_type == "adalora":
        from peft import AdaLoraConfig

        adalora_train_args_dict = cfg.get("adalora", {})
        config = AdaLoraConfig(**adalora_train_args_dict)

    elif finetune_type == "ia3":
        from peft import IA3Config

        ia3_train_args_dict = cfg.get("ia3", {})
        config = IA3Config(**ia3_train_args_dict)

    else:
        raise ValueError("Value must be lora, adalora, ia3, dora")

    model = get_peft_model(base_model, config)

    # Format dataset
    def formatting_func(sample):
        return format_instruction(sample, data_cfg)

    # Output directory and run name
    if is_sweep:
        output_dir = f"./outputs/checkpoints/sweep_{finetune_type}_trial{trial_tag}"
        run_name = f"sweep_{finetune_type}_trial{trial_tag}"
    else:
        output_dir = f"./outputs/checkpoints/qwen2.5-1.5b-{finetune_type}_finetune"
        run_name = f"qwen2.5-1_5b-alpaca-cleaned-{finetune_type}"

    # Training arguments
    training_args = SFTConfig(
        **train_args_dict,
        output_dir=output_dir,
        run_name=run_name,
        deepspeed="scripts/config_file/ds_config.json",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        loss_type="nll",
    )

    # Completion-only collator (trains only on responses, not prompts)
    collator = custom_data_collator(
        tokenizer=tokenizer,
        response_template=data_cfg.response_template,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        formatting_func=formatting_func,
        data_collator=collator,
        args=training_args,
        compute_metrics=None,
        callbacks=[
            EfficiencyMetricsCallback(),
            PerplexityCallback(),
            EarlyStoppingCallback(early_stopping_patience=3),
        ],
    )

    param_metrics = compute_trainable_params(model, trainer=trainer)

    logger.info(">>>> Starting fine-tuning...")
    trainer.train()

    # Save finetuned model
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info(f">>>> Fine tuned model saved to {output_dir}")

    # Collect final metrics
    eval_metrics = trainer.evaluate()
    result = {
        "train_loss": trainer.state.log_history[-1].get("loss"),
        "eval_loss": eval_metrics.get("eval_loss"),
        "trainable_params": param_metrics.get("trainable_params"),
        "trainable_pct": param_metrics.get("trainable_pct"),
        "total_params": param_metrics.get("total_params"),
        "output_dir": output_dir,
        "global_step": trainer.state.global_step,
    }

    # Extract efficiency metrics from the last log entry
    for entry in reversed(trainer.state.log_history):
        if "train_runtime" in entry:
            result["train_runtime_s"] = entry["train_runtime"]
            result["train_samples_per_second"] = entry.get("train_samples_per_second")
            break

    # Extract memory metrics from callback logs
    for entry in reversed(trainer.state.log_history):
        if "memory/peak_gb_across_gpus" in entry:
            result["peak_gpu_gb"] = entry["memory/peak_gb_across_gpus"]
            break

    logger.info(f">>>> Metrics: {result}")
    return result


def load_checkpoint(model_path: str, tokenizer_path: str):
    """Load finetuned model and tokenizer from checkpoint."""
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    return model, tokenizer
