from dataclasses import dataclass

# ─────────────────────────────────────────────
# Config Dataclasses
# ─────────────────────────────────────────────


@dataclass
class ModelConfig:
    model_name_or_path: str = "qwen/qwen2.5-1.5B"
    tokenizer_name: str | None = None
    use_flash_attention: bool = False
    trust_remote_code: bool = False
    cache_dir: str | None = None


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
