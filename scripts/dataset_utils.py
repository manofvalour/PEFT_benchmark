"""
Dataset Preparation Utilities
Handles:
mask loss on prompt token, and train_test_split
"""

import logging

import torch
from datasets import Dataset, DatasetDict

logger = logging.getLogger(__name__)


def custom_data_collator(tokenizer, response_template: str):
    """
    Returns a data collator that masks loss on prompt tokens.
    Only computes loss on the response portion.
    """
    response_token_ids = tokenizer.encode(response_template, add_special_tokens=False)

    def collate_fn(examples):
        input_ids = [torch.tensor(e["input_ids"]) for e in examples]
        labels = [torch.tensor(e["input_ids"]).clone() for e in examples]

        for label in labels:
            # Find where the response starts and mask everything before it
            for i in range(len(label) - len(response_token_ids) + 1):
                if label[i : i + len(response_token_ids)].tolist() == response_token_ids:
                    label[:i] = -100  # -100 = ignore in CrossEntropyLoss
                    break

        # Pad sequences
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": input_ids.ne(tokenizer.pad_token_id),
        }

    return collate_fn


def train_val_test_split(
    dataset: Dataset,
    val_ratio: float = 0.5,
    test_ratio: float = 0.05,
    seed: int = 42,
) -> DatasetDict:
    """Split a dataset into train/val/test."""
    total = len(dataset)
    n_test = int(total * test_ratio)
    n_val = int(n_test * val_ratio)

    tmp = dataset.train_test_split(test_size=n_test, seed=seed)
    train_val = tmp["test"].train_test_split(test_size=n_val, seed=seed)

    return DatasetDict(
        {
            "train": tmp["train"],
            "validation": train_val["test"],
            "test": train_val["train"],
        }
    )
