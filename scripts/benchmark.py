import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import torch
from config import DataConfig
from datasets import Dataset, load_from_disk
from evaluate import EvaluationMetrics
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


# Instruction-Following Benchmark
@dataclass
class BenchmarkResult:
    model_name: str
    task: str
    metrics: dict[str, float] = field(default_factory=dict)
    num_samples: int = 0
    generation_config: dict = field(default_factory=dict)


def run_generation_benchmark(
    model,
    tokenizer,
    dataset: Dataset,
    prompt_template: str,
    reference_column: str = "output",
    max_new_tokens: int = 256,
    batch_size: int = 70,
    temperature: float = 0.1,
) -> BenchmarkResult:
    """Run full generation benchmark on a dataset split."""
    predictions, references = [], []

    for i in tqdm(range(0, len(dataset), batch_size), desc="Evaluating"):
        batch = dataset.select(range(i, min(i + batch_size, len(dataset))))
        prompts = [prompt_template.format(**row) for row in batch]
        refs = [row[reference_column] for row in batch]

        ## encode input
        inputs = tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True, max_length=512
        ).to(model.device)

        # run model generation
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=tokenizer.eos_token_id,
            )

        # decode the input
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        # Strip prompt from output
        for prompt, dec, ref in zip(prompts, decoded, refs):
            predictions.append(dec[len(prompt) :].strip())
            references.append(ref.strip())

    eval = EvaluationMetrics(predictions, references)
    rouge = eval.compute_rouge()
    bert = eval.compute_bertscore()
    bleu = eval.compute_bleu()

    return BenchmarkResult(
        model_name=getattr(model.config, "_name_or_path", "unknown"),
        task="generation",
        metrics={**rouge, **bert, **bleu},
        num_samples=len(dataset),
        generation_config={"max_new_tokens": max_new_tokens, "temperature": temperature},
    )


def load_model_with_adapter(base_model_name: str, adapter_path: str = None, device: str = "auto"):
    """Load a base model + LoRA adapter and run inference."""

    kwargs = {"device_map": device, "torch_dtype": torch.bfloat16}
    base = AutoModelForCausalLM.from_pretrained(base_model_name, **kwargs)

    if adapter_path:
        tokenizer = AutoTokenizer.from_pretrained(adapter_path)
        model = PeftModel.from_pretrained(base, adapter_path)
        logger.info(f">>> Loaded {base_model_name} + adapter from {adapter_path}")
    else:
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        model = base

    return model, tokenizer


# benchmark result
def benchmark_result(
    model_path: str,
    adapter_path: str,
    eval_texts: Dataset,
    prompt_template: str,
    output_json: str = "eval_comparison.json",
):
    """evaluation result for fine-tuned model."""

    logger.info("Evaluating finetuned base model...")

    model, tokenizer = load_model_with_adapter(model_path, adapter_path=None)

    # ft_ppl = compute_perplexity(model, tokenizer, eval_texts)
    benchmark_result = run_generation_benchmark(model, tokenizer, eval_texts, prompt_template)
    ft_result = benchmark_result.metrics

    results = {
        "finetuned_model": {"path": adapter_path, **ft_result}  # ppl, **ft_result},
    }

    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {output_json}")
    return results


def eval_main(
    model_path: str, config_path: str, prompt_template: str, adapter_path: str, output_dir: str
):

    with Path.open(config_path) as f:
        cfg = json.load(f)

    data_cfg = DataConfig(**cfg.get("data", {}))

    if Path(f"dataset/{data_cfg.dataset_name}").is_dir():
        data_cfg.test_file = f"dataset/{data_cfg.dataset_name}/test_file"

        test_ds = load_from_disk(Path(data_cfg.test_file))

    benchmark_result(
        model_path=model_path,
        adapter_path=adapter_path,
        prompt_template=prompt_template,
        eval_texts=test_ds,
        output_json=output_dir,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, help="Path to base model")
    parser.add_argument("--config_path", required=True, help="Path to config file")
    parser.add_argument("--adapter_path", required=False, help="Path to LoRA adapter")
    parser.add_argument("--eval_texts", required=True, help="Path to evaluation texts (JSONL)")
    parser.add_argument(
        "--prompt_template",
        default="{instruction} {input}\n\n### Response:\n",
        help="Prompt template for generation",
    )
    parser.add_argument("--output_json", default="eval.json", help="Output JSON file for results")
    args = parser.parse_args()

    eval_main(
        args.model_path, args.config_path, args.eval_texts, args.prompt_template, args.output_json
    )
