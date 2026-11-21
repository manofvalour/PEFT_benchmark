import modal

app = modal.App("qwen-sft-training")
image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.12")
    .pip_install("torch==2.12.1")
    .pip_install(
        "transformers",
        "peft",
        "trl",
        "accelerate",
        "deepspeed",
        "datasets",
        "wandb",
        "liger-kernel",
    )
    .add_local_dir(
        ".",
        remote_path="/root/app",
        ignore=[".venv", ".git", "__pycache__", "*.pyc", "outputs", "*.log", ".ipynb_checkpoints"],
    )
)

volume = modal.Volume.from_name("training-outputs", create_if_missing=True)


@app.function(
    image=image,
    gpu="A10G:4",
    timeout=60 * 60 * 2,
    volumes={"/root/app/outputs": volume},
    secrets=[modal.Secret.from_name("wandb-secret")],
)
def train_job(config_path: str = "scripts/config_file/config.json", finetune_type: str = "lora"):
    """Run a single training job on Modal GPUs."""
    import subprocess

    subprocess.run(
        [
            "accelerate",
            "launch",
            "--config_file",
            "scripts/config_file/accelerate_cfg.yaml",
            "main.py",
            "--config",
            config_path,
            "--finetune_type",
            finetune_type,
        ],
        check=True,
        cwd="/root/app/",
    )
    volume.commit()


@app.function(
    image=image,
    gpu="A10G:4",
    timeout=60 * 60 * 2,
    volumes={"/root/app/outputs": volume},
    secrets=[modal.Secret.from_name("wandb-secret")],
)
def sweep_trial(config_path: str, finetune_type: str) -> str:
    """Run a sweep trial and return metrics as JSON."""
    import json
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "main.py",
            "--config",
            config_path,
            "--finetune_type",
            finetune_type,
        ],
        capture_output=True,
        text=True,
        cwd="/root/app/",
    )

    volume.commit()

    if result.returncode != 0:
        return json.dumps(
            {
                "status": "failed",
                "error": result.stderr[-2000:] if result.stderr else "unknown error",
            }
        )

    # Parse the metrics line from stdout (trainer prints ">>>> Metrics: {...}")
    for line in reversed(result.stdout.splitlines()):
        if ">>>> Metrics:" in line:
            metrics_str = line.split(">>>> Metrics:", 1)[1].strip()
            return json.dumps({"status": "success", "metrics": json.loads(metrics_str)})

    return json.dumps({"status": "completed", "metrics": {}})


@app.local_entrypoint()
def main(
    config_path: str = "scripts/config_file/config.json",
    finetune_type: str = "lora",
):
    train_job.remote(config_path=config_path, finetune_type=finetune_type)
