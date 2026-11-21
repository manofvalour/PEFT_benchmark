"""
PEFT Hyperparameter Sweep Controller

Generates trial configs from the search space defined in sweep_config.yaml,
dispatches each trial to Modal GPUs, and collects results.

Usage:
    # Sweep a single method (20 random trials on Modal)
    uv run scripts/sweep.py --method lora --num_trials 20 --backend modal

    # Sweep a single method locally (no Modal)
    uv run scripts/sweep.py --method lora --num_trials 5 --backend local

    # Sweep all methods
    uv run scripts/sweep.py --method all --num_trials 20 --backend modal

    # Grid search (all combinations)
    uv run scripts/sweep.py --method lora --search grid --backend local
"""

import argparse
import itertools
import json
import logging
import random
import subprocess
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import yaml

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
SWEEP_CONFIG = ROOT / "scripts" / "config_file" / "sweep_config.yaml"
BASE_CONFIG = ROOT / "scripts" / "config_file" / "config.json"
RESULTS_DIR = ROOT / "outputs" / "sweep_results"


def load_sweep_config() -> dict:
    with Path.open(SWEEP_CONFIG) as f:
        return yaml.safe_load(f)


def load_base_config() -> dict:
    with Path.open(BASE_CONFIG) as f:
        return json.load(f)


def build_fixed_overrides(sweep_cfg: dict) -> list[str]:
    """Convert fixed params from sweep_config.yaml into override strings."""
    overrides = []
    fixed = sweep_cfg.get("fixed", {})

    # training overrides
    for key, val in fixed.get("training", {}).items():
        overrides.append(f"training.{key}={val}")

    # model overrides
    for key, val in fixed.get("model", {}).items():
        overrides.append(f"model.{key}={val}")

    # data overrides
    for key, val in fixed.get("data", {}).items():
        overrides.append(f"data.{key}={val}")

    return overrides


def generate_grid_combinations(search_space: dict) -> list[dict]:
    """Generate all combinations from the search space (cartesian product)."""
    keys = list(search_space.keys())
    values = list(search_space.values())
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def generate_random_samples(search_space: dict, n: int) -> list[dict]:
    """Sample n random combinations from the search space."""
    samples = []
    for _ in range(n):
        sample = {k: random.choice(v) for k, v in search_space.items()}
        samples.append(sample)
    return samples


def build_trial_overrides(trial_params: dict, method: str) -> list[str]:
    """Convert trial parameters into override strings.

    PEFT-specific params go to the method section (e.g. ``lora.r=16``).
    Training params like learning_rate go to ``training.learning_rate``.
    """
    overrides = []
    peft_keys = {
        "lora": {"r", "lora_alpha", "lora_dropout", "bias", "task_type", "target_modules"},
        "dora": {
            "r",
            "lora_alpha",
            "lora_dropout",
            "bias",
            "task_type",
            "target_modules",
            "use_dora",
        },
        "adalora": {
            "init_r",
            "target_r",
            "tinit",
            "tfinal",
            "deltaT",
            "total_step",
            "target_modules",
            "modules_to_save",
            "lora_alpha",
        },
        "ia3": {"task_type", "target_modules", "feedforward_modules"},
    }

    training_keys = {
        "learning_rate",
        "num_train_epochs",
        "per_device_train_batch_size",
        "gradient_accumulation_steps",
        "weight_decay",
        "lr_scheduler_type",
        "warmup_steps",
        "max_grad_norm",
    }

    for key, val in trial_params.items():
        if key in training_keys:
            overrides.append(f"training.{key}={val}")
        elif method in peft_keys and key in peft_keys[method]:
            overrides.append(f"{method}.{key}={val}")
        else:
            # default: put in the method section
            overrides.append(f"{method}.{key}={val}")

    return overrides


def write_trial_config(
    base_config: dict,
    fixed_overrides: list[str],
    trial_overrides: list[str],
    trial_id: int,
    method: str,
) -> Path:
    """Build and write a trial config JSON file."""
    import copy

    cfg = copy.deepcopy(base_config)

    # Apply fixed overrides
    for item in fixed_overrides:
        key, _, value = item.partition("=")
        parts = key.split(".")
        d = cfg
        for part in parts[:-1]:
            d = d.setdefault(part, {})
        d[parts[-1]] = _cast_value(value)

    # Apply trial overrides
    for item in trial_overrides:
        key, _, value = item.partition("=")
        parts = key.split(".")
        d = cfg
        for part in parts[:-1]:
            d = d.setdefault(part, {})
        d[parts[-1]] = _cast_value(value)

    # Write config
    config_dir = ROOT / "scripts" / "config_file"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / f"sweep_trial_{method}_{trial_id}.json"
    with Path.open(config_path, "w") as f:
        json.dump(cfg, f, indent=2)

    return config_path


def _cast_value(value: str):
    """Auto-cast a string value to int/float/bool/None."""
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    if value.lower() in ("none", "null"):
        return None
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def _parse_metrics_from_output(stdout: str) -> dict:
    """Parse the '>>>> Metrics: {...}' line from trainer output."""
    for line in reversed(stdout.splitlines()):
        if ">>>> Metrics:" in line:
            metrics_str = line.split(">>>> Metrics:", 1)[1].strip()
            try:
                return json.loads(metrics_str)
            except json.JSONDecodeError:
                return {"raw_metrics": metrics_str}
    return {}


def run_trial_local(config_path: Path, method: str, trial_id: int) -> dict:
    """Run a trial locally via main.py."""
    logger.info(f"  [trial {trial_id}] Running locally: {method}")
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "main.py"),
            "--config",
            str(config_path),
            "--finetune_type",
            method,
        ],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )

    if result.returncode != 0:
        logger.error(f"  [trial {trial_id}] FAILED: {result.stderr[-500:]}")
        return {"status": "failed", "trial_id": trial_id, "error": result.stderr[-1000:]}

    # Parse metrics from stdout
    metrics = _parse_metrics_from_output(result.stdout)
    metrics["status"] = "success"
    metrics["trial_id"] = trial_id
    return metrics


def run_trial_modal(config_path: Path, method: str, trial_id: int) -> dict:
    """Run a trial on Modal GPUs via modal_train.py::sweep_trial."""
    logger.info(f"  [trial {trial_id}] Dispatching to Modal: {method}")

    # We use the Modal Python SDK directly for cleaner integration
    import modal_train

    fn = modal_train.sweep_trial
    remote_result = fn.remote(str(config_path), method)
    result = json.loads(remote_result)

    if result.get("status") == "failed":
        logger.error(f"  [trial {trial_id}] Modal trial failed: {result.get('error', '')[:300]}")
        return {"status": "failed", "trial_id": trial_id, "error": result.get("error", "")}

    metrics = result.get("metrics", {})
    metrics["status"] = "success"
    metrics["trial_id"] = trial_id
    return metrics


def run_sweep_modal_parallel(
    method: str,
    trials: list[dict],
    sweep_cfg: dict,
    base_config: dict,
    fixed_overrides: list[str],
) -> list[dict]:
    """Dispatch all trials for one method to Modal in parallel via .map()."""
    import modal

    try:
        fn = modal.Function.from_name("qwen-sft-training", "sweep_trial")
    except AttributeError:
        fn = modal.Function.lookup("qwen-sft-training", "sweep_trial")

    # Build every trial config up front (this is the part that made
    # sequential dispatch pointless to keep — configs don't depend on
    # earlier trials' results, so there's no reason to interleave
    # config-writing with remote calls).
    config_paths = []
    for trial_id, params in enumerate(trials):
        trial_overrides = build_trial_overrides(params, method)
        config_path = write_trial_config(
            base_config, fixed_overrides, trial_overrides, trial_id, method
        )
        config_paths.append(str(config_path))

    methods_arg = [method] * len(config_paths)

    logger.info(f"Dispatching {len(config_paths)} trials for {method} in parallel...")
    sweep_start = time.time()

    # return_exceptions=True: one trial raising shouldn't kill the whole
    # sweep — you want partial results, not zero results.
    raw_results = list(fn.map(config_paths, methods_arg, return_exceptions=True))

    sweep_elapsed = time.time() - sweep_start
    logger.info(f"All {method} trials returned in {sweep_elapsed / 60:.1f}min (wall clock)")

    all_results = []
    for trial_id, (params, raw) in enumerate(zip(trials, raw_results)):
        if isinstance(raw, Exception):
            result = {"status": "failed", "trial_id": trial_id, "error": str(raw)}
        else:
            parsed = json.loads(raw) if isinstance(raw, str) else raw
            if parsed.get("status") == "failed":
                result = {
                    "status": "failed",
                    "trial_id": trial_id,
                    "error": parsed.get("error", ""),
                }
            else:
                result = parsed.get("metrics", {})
                result["status"] = "success"
                result["trial_id"] = trial_id

        result["params"] = params
        result["method"] = method
        all_results.append(result)

    return all_results


def run_sweep(method, num_trials, search, backend, sweep_cfg):
    base_config = load_base_config()
    fixed_overrides = build_fixed_overrides(sweep_cfg)
    search_space = sweep_cfg["search_space"][method]

    if search == "grid":
        trials = generate_grid_combinations(search_space)
    else:
        trials = generate_random_samples(search_space, num_trials)
        if len(trials) > num_trials:
            trials = trials[:num_trials]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if backend == "modal":
        all_results = run_sweep_modal_parallel(
            method, trials, sweep_cfg, base_config, fixed_overrides
        )
    else:
        all_results = []
        for trial_id, params in enumerate(trials):
            trial_overrides = build_trial_overrides(params, method)
            config_path = write_trial_config(
                base_config, fixed_overrides, trial_overrides, trial_id, method
            )
            start_time = time.time()
            result = run_trial_local(config_path, method, trial_id)
            elapsed = time.time() - start_time
            result["params"] = params
            result["method"] = method
            result["elapsed_s"] = round(elapsed, 1)
            all_results.append(result)

    results_path = RESULTS_DIR / f"{method}_results.json"
    with Path.open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    _print_summary(method, all_results)
    return all_results


def _print_summary(method: str, results: list[dict]):
    """Print a summary table of sweep results."""
    successful = [r for r in results if r.get("status") == "success"]
    if not successful:
        logger.info(f"\nNo successful trials for {method}")
        return

    logger.info(f"\n{'=' * 80}")
    logger.info(f"SWEEP SUMMARY: {method.upper()}")
    logger.info(f"{'=' * 80}")
    logger.info(f"Total trials: {len(results)}, Successful: {len(successful)}")

    # Sort by eval_loss
    successful.sort(key=lambda x: x.get("eval_loss") or float("inf"))

    logger.info(
        f"\n{'Trial':<8} {'Eval Loss':<12} {'Train Loss':<12} {'Params':<12} {'Time':<10}{'Config'}"
    )
    logger.info("-" * 90)
    for r in successful[:10]:  # top 10
        trial_id = r.get("trial_id", "?")
        eval_loss = f"{r.get('eval_loss', 0):.4f}" if r.get("eval_loss") else "N/A"
        train_loss = f"{r.get('train_loss', 0):.4f}" if r.get("train_loss") else "N/A"
        params = (
            f"{r.get('trainable_params', 0) / 1e6:.2f}M" if r.get("trainable_params") else "N/A"
        )
        elapsed = f"{r.get('elapsed_s', 0) / 60:.1f}m"
        config_str = json.dumps(r.get("params", {}), default=str)[:40]
        logger.info(
            f"{trial_id:<8} {eval_loss:<12} {train_loss:<12} {params:<12} {elapsed:<10}{config_str}"
        )

    best = successful[0]
    logger.info(f"\nBest config: {json.dumps(best.get('params', {}), default=str)}")
    logger.info(f"Best eval_loss: {best.get('eval_loss', 'N/A')}")


def main():
    parser = argparse.ArgumentParser(description="PEFT Hyperparameter Sweep")
    parser.add_argument(
        "--method",
        required=True,
        choices=["lora", "dora", "adalora", "ia3", "all"],
        help="PEFT method to sweep",
    )
    parser.add_argument(
        "--num_trials",
        type=int,
        default=20,
        help="Number of random trials per method (default: 20)",
    )
    parser.add_argument(
        "--search",
        choices=["random", "grid"],
        default="random",
        help="Search strategy: random or grid (default: random)",
    )
    parser.add_argument(
        "--backend",
        choices=["local", "modal"],
        default="local",
        help="Execution backend: local GPU or Modal cloud GPUs (default: local)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    sweep_cfg = load_sweep_config()

    methods = ["lora", "dora", "adalora", "ia3"] if args.method == "all" else [args.method]

    for method in methods:
        if method not in sweep_cfg.get("search_space", {}):
            logger.warning(f"No search space defined for {method}, skipping")
            continue

        logger.info(f"\n{'#' * 80}")
        logger.info(f"# SWEEPING: {method.upper()}")
        logger.info(f"{'#' * 80}")

        run_sweep(
            method=method,
            num_trials=args.num_trials,
            search=args.search,
            backend=args.backend,
            sweep_cfg=sweep_cfg,
        )

    logger.info("\n\nSweep complete! Results in outputs/sweep_results/")


if __name__ == "__main__":
    main()
