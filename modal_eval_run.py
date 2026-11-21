import logging

import modal

logger = logging.getLogger(__name__)

app = modal.App("qwen-sft-evaluation")

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.12")
    .pip_install("torch==2.12.1")
    .pip_install(
        "transformers",
        "trl",
        "accelerate",
        "deepspeed",
        "datasets",
        "wandb",
        "evaluate",
        "tqdm",
        "rouge_score",
        "sacrebleu",
        "peft",
        "bert_score",
        "lm-eval",
    )
    .add_local_dir(
        ".",
        remote_path="/root/app",
        ignore=[".venv", ".git", "__pycache__", "*.pyc", "outputs", "*.log", ".ipynb_checkpoints"],
    )
)

eval_volume = modal.Volume.from_name("evaluation-outputs", create_if_missing=True)


@app.function(
    image=image,
    gpu="A10G:1",
    timeout=60 * 60 * 2,
    volumes={
        #    "/training-outputs": training_volume,  # mount the volume that actually has the model
        "/evaluation-outputs": eval_volume,
    },
    secrets=[modal.Secret.from_name("wandb-secret")],
)
def run_eval_suite(adapter_path: str, model_name: str, report_path: str, config_path: str):
    import subprocess

    logging.basicConfig(level=logging.INFO, force=True)
    logger.info("Starting eval...")

    try:
        subprocess.run(
            [
                "python",
                "scripts/benchmark.py",
                "--model_name",
                model_name,
                "--config_path",
                config_path,
                "--adapter_path",
                adapter_path,
                "--output_dir",
                f"/evaluation-outputs/reports/{report_path}",
            ],
            check=True,
            cwd="/root/app",
        )
    finally:
        eval_volume.commit()


@app.local_entrypoint()
def main(adapter_path, model_name, report_dir, config_path="scripts/config_file/config.json"):

    run_eval_suite.remote(
        adapter_path=adapter_path,
        model_name=model_name,
        report_path=report_dir,
        config_path=config_path,
    )
