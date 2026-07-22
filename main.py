import argparse
import json
from pathlib import Path

from datasets import load_from_disk

from scripts.config import DataConfig
from scripts.trainer import load_and_prepare_data, train

data_cfg = DataConfig()


def main(config_path: str):
    from scripts.config import DataConfig

    with Path.open(config_path) as f:
        cfg = json.load(f)

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

        ## saving test set to disk for later evaluation
        data_cfg.test_file = f"dataset/{data_cfg.dataset_name}/test_file"
        data_cfg.train_file = f"dataset/{data_cfg.dataset_name}/train_file"
        data_cfg.val_file = f"dataset/{data_cfg.dataset_name}/val_file"

        test_ds.save_to_disk(data_cfg.test_file)
        train_ds.save_to_disk(data_cfg.train_file)
        val_ds.save_to_disk(data_cfg.val_file)

    train(config_path, train_ds, val_ds)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to JSON config file")
    args = parser.parse_args(["--config", "scripts/config_file/config.json"])
    main(args.config)
