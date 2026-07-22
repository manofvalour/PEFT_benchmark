import modal

app = modal.App("gpu-test")


@app.function(gpu="A10G")
def check_gpu():
    import subprocess

    subprocess.run(["nvidia-smi"])


@app.local_entrypoint()
def main():
    check_gpu.remote()
