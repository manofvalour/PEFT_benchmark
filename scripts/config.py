from dataclasses import dataclass


# ─────────────────────────────────────────────
# Config Dataclasses
# ─────────────────────────────────────────────
@dataclass
class ModelConfig:
    model_name_or_path: str = "Qwen/Qwen2.5-1.5B"
    tokenizer_name: str = None
    trust_remote_code: bool = True
    use_flash_attn: bool = True


@dataclass
class DataConfig:
    dataset_name: str | None = None
    dataset_config: str | None = None
    train_file: str | None = None
    val_file: str | None = None
    text_column: str = "text"
    prompt_column: str | None = "instruction"
    response_column: str | None = "output"
    max_seq_length: int = 2048
    instruction_template: str = "### Instruction:\n"
    response_template: str = "### Response:\n"
    val_split: float = 0.05
    num_proc: int = 4
    packing: bool = True
