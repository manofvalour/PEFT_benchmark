import logging
import os
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="[%(asctime)s]: %(levelname)s: %(message)s")

list_of_dir = [
    "config/experiments/qwen_agnews.yaml",
    "config/models/qwen_1_5b.yaml",
    "config/models/gemma_2b.yaml",
    "config/models/smollm.yaml",
    "config/peft/lora.yaml",
    "config/peft/prefix_tuning.yaml",
    "config/peft/adalora.yaml",
    "config/peft/dora.yaml",
    "config/peft/1a3.yaml",
    "config/trainer/default_trainer.yaml",
    "config/trainer/deepspeed_trainer.yaml",
    "datasets/download.py",
    "datasets/preprocess.py",
    "datasets/cache.py",
    "datasets/__init__.py",
    "src/__init__.py",
    "src/benchmark/__init__.py",
    "src/configs/__init__.py",
    "src/models/loader.py",
    "src/models/__init__.py",
    "src/models/tokenizer.py",
    "src/peft/__init__.py",
    "src/peft/builder.py",
    "src/peft/lora.py",
    "src/peft/prefix_tuning.py",
    "src/peft/adalora.py",
    "src/peft/dora.py",
    "src/peft/1a3.py",
    "src/training/__init__.py",
    "src/training/callback.py",
    "src/training/optimizer.py",
    "src/training/trainer.py",
    "src/evaluation/metrics.py",
    "src/evaluation/__init__.py",
    "src/evaluation/inference.py",
    "src/evaluation/evaluator.py",
    "src/profiling/gpu_profiler.py",
    "src/profiling/__init__.py",
    "src/profiling/throughput.py",
    "src/profiling/timer.py",
    "src/profiling/memory.py",
    "src/logging/__init__.py",
    "src/logging/logger.py",
    "src/logging/wand_b_logger.py",
    "src/logging/csv_logger.py",
    "src/visualization/__init__.py",
    "src/visualization/plot.py",
    "src/visualization/pareto.py",
    "src/visualization/dashboard.py",
    "src/utils/__init__.py",
    "src/utils/io.py",
    "src/utils/helpers.py",
    "scripts/__init__.py",
    "scripts/trainer.py",
    "scripts/evaluate.py",
    "scripts/benchmark.py",
    "scripts/profile.py",
    "scripts/reports.py",
    "outputs/__init__.py",
    "outputs/checkpoints/",
    "outputs/logs/",
    "outputs/metrics/",
    "outputs/figures/",
    "outputs/reports/",
    "notebooks/analysis.ipynb",
    "notebooks/visualizations.ipynb",
    "tests/test_peft.py",
    "tests/test_metrics.py",
    "tests/test_config.py",
    "docs/methodology.md",
    "docs/results.md",
]


for dir in list_of_dir:
    file_path = Path(dir)
    file_dir, filename = os.path.split(file_path)

    if file_dir != "":
        Path(file_dir).mkdir(parents=True, exist_ok=True)
        logging.info(f"directory created successfully {file_dir}")

    if not (Path(file_path).exists()) or (Path((file_path).stat().st_size == 0)):
        with Path.open(file_path, "w") as file:
            pass

    else:
        logging.info("file exists already")
