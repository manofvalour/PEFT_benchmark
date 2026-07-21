import modal

app = modal.App("test-app")


@app.function()
def hello():
    print("Hello from Modal!")


@app.local_entrypoint()
def main():
    hello.remote()
