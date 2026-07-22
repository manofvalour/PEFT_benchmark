import modal

app = modal.App("qwen-sft-training")
image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.12")
    .pip_install("torch==2.12.1")
    # .pip_install("wheel", "setuptools", "packaging", "ninja")
    .pip_install("transformers", "trl", "accelerate", "deepspeed", "datasets", "wandb")
    # .pip_install("flash-attn", extra_options="--no-build-isolation")
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
def train_job():
    import subprocess

    subprocess.run(
        [
            "accelerate",
            "launch",
            "--config_file",
            "scripts/config_file/accelerate_cfg.yaml",
            "main.py",
            "--config",
            "scripts/config_file/config.json",
        ],
        check=True,
        cwd="/root/app/",
    )
    volume.commit()


@app.local_entrypoint()
def main():
    train_job.remote()
