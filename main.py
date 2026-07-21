import argparse

from scripts.trainer import train

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to JSON config file")
    args = parser.parse_args(["--config", "scripts/config_file/config.json"])
    train(args.config)
