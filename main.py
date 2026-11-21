import argparse
import json
from pathlib import Path

from datasets import load_from_disk

from scripts.config import DataConfig
from scripts.trainer import load_and_prepare_data, train


def apply_overrides(cfg: dict, overrides: list[str]) -> dict:
    """Apply key=value overrides to a nested config dict.

    Supports dot notation for nested keys: ``training.learning_rate=1e-4``.
    Values are auto-cast to int/float/bool/None when possible.
    """
    for item in overrides:
        key, _, value = item.partition("=")
        if not value:
            raise ValueError(f"Override must be key=value, got: {item}")

        # auto-cast value
        if value.lower() == "true":
            casted = True
        elif value.lower() == "false":
            casted = False
        elif value.lower() == "none" or value.lower() == "null":
            casted = None
        else:
            try:
                casted = int(value)
            except ValueError:
                try:
                    casted = float(value)
                except ValueError:
                    casted = value

        # set nested key
        parts = key.split(".")
        d = cfg
        for part in parts[:-1]:
            d = d.setdefault(part, {})
        d[parts[-1]] = casted

    return cfg


def main(config_path: str, finetune_type: str, overrides: list[str] | None = None):
    with Path.open(config_path) as f:
        cfg = json.load(f)

    if overrides:
        cfg = apply_overrides(cfg, overrides)

    data_cfg = DataConfig(**cfg.get("data", {}))

    if Path(f"dataset/{data_cfg.dataset_name}").is_dir():
        data_cfg.test_file = f"dataset/{data_cfg.dataset_name}/test_file"
        data_cfg.train_file = f"dataset/{data_cfg.dataset_name}/train_file"
        data_cfg.val_file = f"dataset/{data_cfg.dataset_name}/val_file"

        train_ds = load_from_disk(Path(data_cfg.train_file))
        val_ds = load_from_disk(Path(data_cfg.val_file))
        test_ds = load_from_disk(Path(data_cfg.test_file))

    else:
        train_ds, val_ds, test_ds = load_and_prepare_data(data_cfg)

        data_cfg.test_file = f"dataset/{data_cfg.dataset_name}/test_file"
        data_cfg.train_file = f"dataset/{data_cfg.dataset_name}/train_file"
        data_cfg.val_file = f"dataset/{data_cfg.dataset_name}/val_file"

        test_ds.save_to_disk(data_cfg.test_file)
        train_ds.save_to_disk(data_cfg.train_file)
        val_ds.save_to_disk(data_cfg.val_file)

    # If overrides were applied, write a temp config so trainer reads the modified values
    if overrides:
        tmp_cfg = Path(config_path).parent / "sweep_trial_config.json"
        with tmp_cfg.open("w") as f:
            json.dump(cfg, f, indent=2)
        config_path = str(tmp_cfg)

    return train(config_path, train_ds, val_ds, finetune_type)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to JSON config file")

    parser.add_argument(
        "--finetune_type",
        required=True,
        type=str,
        help="type of finetuning techniques (e.g lora, adalora, ia3, dora)",
    )

    parser.add_argument(
        "--overrides",
        nargs="*",
        default=None,
        help="Key=value config overrides (e.g learning_rate=1e-4 lora.r=32)",
    )

    args = parser.parse_args()
    main(args.config, args.finetune_type, args.overrides)
