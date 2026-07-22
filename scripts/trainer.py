"""
Fine-tuning Trainer
"""

import json
import logging
from pathlib import Path
from typing import Any

import torch
from config import DataConfig, ModelConfig

# Main Training Entry Point
from dataset_utils import custom_data_collator, train_val_test_split
from datasets import load_dataset
from evaluate import EfficiencyMetricsCallback, PerplexityCallback, compute_trainable_params
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    EarlyStoppingCallback,
)
from trl import SFTConfig
from trl.trainer.sft_trainer import SFTTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Model Loading
def load_base_model(model_cfg: ModelConfig):
    """Load base model"""

    model_kwargs: dict[str, Any] = {
        "pretrained_model_name_or_path": model_cfg.model_name_or_path,
        "trust_remote_code": model_cfg.trust_remote_code,
        "torch_dtype": torch.bfloat16,
        #  "device_map": "auto",
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

    else:
        raise ValueError("Provide dataset_name or train_file.")

    print(f"Train: {len(train_ds):,} | Val: {len(val_ds):,} | Test: {len(test_ds):,}")
    return train_ds, val_ds, test_ds


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


def train(config_path: str, train_ds, val_ds):
    with Path.open(config_path) as f:
        cfg = json.load(f)

    model_cfg = ModelConfig(**cfg.get("model", {}))
    data_cfg = DataConfig(**cfg.get("data", {}))
    train_args_dict = cfg.get("training", {})

    # Load components
    tokenizer = load_tokenizer(model_cfg)
    model = load_base_model(model_cfg)

    # Format dataset
    def formatting_func(sample):
        return format_instruction(sample, data_cfg)

    # Training arguments
    training_args = SFTConfig(
        **train_args_dict,
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

    compute_trainable_params(model, trainer=trainer)

    logger.info(">>>> Starting fine-tuning...")
    trainer.train()

    # Save finetuned
    output_dir = training_args.output_dir
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info(f">>>> Fine tuned model saved to {output_dir}")


def load_checkpoint(model_path: str, tokenizer_path: str):
    """Load finetuned model and tokenizer from checkpoint."""
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    return model, tokenizer
