import modal

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

training_volume = modal.Volume.from_name("training-outputs", create_if_missing=False)
eval_volume = modal.Volume.from_name("evaluation-outputs", create_if_missing=True)


@app.function(
    image=image,
    gpu="A10G:1",
    timeout=60 * 60 * 2,
    volumes={
        "/training-outputs": training_volume,  # mount the volume that actually has the model
        "/eval-outputs": eval_volume,
    },
    secrets=[modal.Secret.from_name("wandb-secret")],
)
def run_eval_suite(model_path: str = "/training-outputs/checkpoints/qwen2.5-1.5b-finetune"):
    import subprocess

    subprocess.run(
        [
            "python",
            "scripts/benchmark.py",
            "--model_path",
            model_path,
            "--config_path",
            "scripts/config_file/config.json",
            "--eval_texts",
            "dataset/yahma/alpaca-cleaned/test_file",
            "--output_json",
            "/eval-outputs/output/report.json",
        ],
        check=True,
        cwd="/root/app",
    )
    eval_volume.commit()


@app.local_entrypoint()
def main():
    run_eval_suite.remote()
