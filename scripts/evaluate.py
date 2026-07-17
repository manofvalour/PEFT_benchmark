"""
Evaluation Suite for Fine-tuned Models
Includes: perplexity, ROUGE, BERTScore, task-specific metricsThese are the evaluation metrics:

Evaluation metrics for classification problems:
    Accuracy, Precision, Recall, F1

Evaluation metrics for instruction tuning:
    BLEU, Exact Match

Parameter Efficiency:
    Trainable parameters, Total parameters,
    Percentage trainable ((trainable parameter)/(Total parameters)) * 100

Memory Efficiency:
    Peak GPU memory, Average GPU memory, Reserved memory, Allocated memory

Computational Efficiency:
    Training time, Epoch time, Tokens/sec, Samples/sec, Checkpoint size

Composite Metrics
    Performance per trainable parameters(PTP) ==> Accuracy/trainable parameters
    Memory Efficiency ==> Accuracy/Peak GPU memory
    Training Efficiency ==> Accuracy/Training Time

"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


# Perplexity
@torch.inference_mode()
def compute_perplexity(
    model,
    tokenizer,
    texts: list[str],
    max_length: int = 1024,
    stride: int = 512,
) -> dict[str, float]:
    """
    Sliding-window perplexity (handles texts longer than context window).
    Returns mean PPL, median PPL, and standard deviation across the corpus.
    """
    model.eval()
    ppls = []

    for text in tqdm(texts, desc="Computing perplexity"):
        encodings = tokenizer(text, return_tensors="pt", truncation=False)
        input_ids = encodings.input_ids.to(model.device)
        seq_len = input_ids.size(1)

        nlls = []
        prev_end = 0

        for begin in range(0, seq_len, stride):
            end = min(begin + max_length, seq_len)
            target_len = end - prev_end
            input_chunk = input_ids[:, begin:end]
            target_chunk = input_chunk.clone()
            target_chunk[:, :-target_len] = -100  # mask previously seen tokens

            with torch.no_grad():
                outputs = model(input_chunk, labels=target_chunk)
                neg_ll = outputs.loss * target_len
            nlls.append(neg_ll)
            prev_end = end
            if end == seq_len:
                break

        ppl = torch.exp(torch.stack(nlls).sum() / prev_end).item()
        ppls.append(ppl)

    return {
        "perplexity_mean": round(float(np.mean(ppls)), 4),
        "perplexity_std": round(float(np.std(ppls)), 4),
        "perplexity_median": round(float(np.median(ppls)), 4),
    }


# Generation Quality Metrics
def compute_rouge(predictions: list[str], references: list[str]) -> dict[str, float]:
    """ROUGE-1, ROUGE-2, ROUGE-L scores."""
    try:
        from rouge_score import rouge_scorer

        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        scores = {"rouge1": [], "rouge2": [], "rougeL": []}
        for pred, ref in zip(predictions, references):
            s = scorer.score(ref, pred)
            for k in scores:
                scores[k].append(s[k].fmeasure)
        return {k: round(float(np.mean(v)), 4) for k, v in scores.items()}

    except ImportError:
        logger.warning("Install rouge-score: pip install rouge-score")
        return {}


def compute_bertscore(
    predictions: list[str],
    references: list[str],
    lang: str = "en",
    model_type: str = "microsoft/deberta-xlarge-mnli",
) -> dict[str, float]:
    """BERTScore F1 (semantic similarity)."""
    try:
        from bert_score import score

        P, R, F1 = score(predictions, references, lang=lang, model_type=model_type, verbose=False)
        return {
            "bertscore_precision": round(P.mean().item(), 4),
            "bertscore_recall": round(R.mean().item(), 4),
            "bertscore_f1": round(F1.mean().item(), 4),
        }
    except ImportError:
        logger.warning("Install bert-score: pip install bert-score")
        return {}


def compute_bleu(prediction: list[str], reference: list[str]) -> dict[str, float]:

    import sacrebleu

    try:
        # Compute corpus BLEU
        res = []
        for pred, ref in zip(prediction, reference):
            result = sacrebleu.corpus_bleu(pred, ref)
            res.append(result.score)

        return {"Bleu Score": np.mean(res)}

    except ImportError:
        logger.warning("insall sacrebleu: pip install sacrebleu")
        return {}


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
    batch_size: int = 8,
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

    rouge = compute_rouge(predictions, references)
    bert = compute_bertscore(predictions, references)
    bleu = compute_bleu(predictions, references)

    return BenchmarkResult(
        model_name=getattr(model.config, "_name_or_path", "unknown"),
        task="generation",
        metrics={**rouge, **bert, **bleu},
        num_samples=len(dataset),
        generation_config={"max_new_tokens": max_new_tokens, "temperature": temperature},
    )


# ─────────────────────────────────────────────
# Comparison: Finetuned models
# ─────────────────────────────────────────────


def compare_models(
    base_model_path: str,
    adapter_path: str,
    eval_texts: list[str],
    prompt_template: str,
    output_json: str = "eval_comparison.json",
):
    """Side-by-side perplexity comparison of base vs fine-tuned model."""
    import LoraInference

    logger.info("Evaluating base model...")
    base_tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path, torch_dtype=torch.bfloat16, device_map="auto"
    )
    base_ppl = compute_perplexity(base_model, base_tokenizer, eval_texts)
    base_benchmark_result = run_generation_benchmark(
        base_model, base_tokenizer, eval_texts, prompt_template
    )
    base_result = base_benchmark_result.metrics

    logger.info("Evaluating fine-tuned model...")
    ft = LoraInference(base_model_path, adapter_path)
    ft_ppl = compute_perplexity(ft.model, ft.tokenizer, eval_texts)
    benchmark_result = run_generation_benchmark(ft.model, ft.tokenizer, eval_texts, prompt_template)
    ft_result = benchmark_result.metrics

    results = {
        "base_model": {"path": base_model_path, **base_ppl, **base_result},
        "finetuned_model": {"path": adapter_path, **ft_ppl, **ft_result},
        "delta": {k: round(base_ppl[k] - ft_ppl[k], 4) for k in base_ppl},
    }

    with Path.open(output_json, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {output_json}")
    return results
