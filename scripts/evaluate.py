"""
Evaluation Suite for Fine-tuned Models
Includes: perplexity, ROUGE, BERTScore, task-specific metricsThese are the evaluation metrics:

Evaluation metrics for classification problems:
    Accuracy, Precision, Recall, F1

Evaluation metrics for instruction tuning:
    BLEU, ROUGE, BERTScore
"""

import logging
import math
import time
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from tqdm import tqdm
from transformers import TrainerCallback

logger = logging.getLogger(__name__)


@torch.inference_mode()
def compute_perplexity(
    model,
    tokenizer,
    texts: Dataset,
    max_length: int = 1024,
    stride: int = 512,
    batch_size: int = 4,
) -> dict[str, float]:
    """
    Sliding-window perplexity (handles texts longer than context window).
    Returns mean PPL and standard deviation across the corpus.
    """
    from trainer import format_instruction

    model.eval()
    ppls = []

    for text in tqdm(texts, desc="Computing perplexity"):
        data = format_instruction(texts)
        encodings = tokenizer(data, return_tensors="pt", truncation=False)
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


# Perplexity
class PerplexityCallback(TrainerCallback):
    @torch.inference_mode()
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None or not state.is_world_process_zero:
            return
        eval_loss = metrics.get("eval_loss")
        if eval_loss is not None:
            perplexity = math.exp(eval_loss) if eval_loss < 20 else float("inf")  # guard overflow
            import wandb

            if wandb.run is not None:
                wandb.log({"benchmark/perplexity": perplexity}, step=state.global_step)


# Generation Quality Metrics
class EvaluationMetrics:
    def __init__(self, predictions: list[str], references: list[str]):
        self.predictions = predictions
        self.references = references

    def compute_rouge(self) -> dict[str, float]:
        """ROUGE-1, ROUGE-2, ROUGE-L scores."""
        try:
            from rouge_score import rouge_scorer

            scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
            scores = {"rouge1": [], "rouge2": [], "rougeL": []}
            for pred, ref in zip(self.predictions, self.references):
                s = scorer.score(ref, pred)
                for k in scores:
                    scores[k].append(s[k].fmeasure)
            return {k: round(float(np.mean(v)), 4) for k, v in scores.items()}

        except ImportError:
            logger.warning("Install rouge-score: pip install rouge-score")
            return {}

    def compute_bertscore(
        self,
        lang: str = "en",
        model_type: str = "roberta-large",
    ) -> dict[str, float]:
        """BERTScore F1 (semantic similarity)."""
        try:
            from bert_score import score

            P, R, F1 = score(
                self.predictions,
                self.references,
                lang=lang,
                model_type=model_type,
                verbose=False,  # baseline_path=None,
                #  idf=False
            )
            return {
                "bertscore_precision": round(P.mean().item(), 4),
                "bertscore_recall": round(R.mean().item(), 4),
                "bertscore_f1": round(F1.mean().item(), 4),
            }
        except ImportError:
            logger.warning("Install bert-score: pip install bert-score")
            return {}

    def compute_bleu(self) -> dict[str, float]:
        try:
            import sacrebleu
        except ImportError:
            logger.warning("install sacrebleu: pip install sacrebleu")
            return {}

        result = sacrebleu.corpus_bleu(self.predictions, [self.references])
        return {"Bleu Score": result.score}


def compute_trainable_params(model, trainer=None) -> dict:
    """Report trainable vs total parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    metrics = {
        "total_params": total,
        "trainable_params": trainable,
        "trainable_pct": round(100 * trainable / total, 4),
        "total_M": round(total / 1e6, 2),
        "trainable_M": round(trainable / 1e6, 2),
    }

    print(f"Trainable params: {trainable:,} / {total:,} ({metrics['trainable_pct']:.4f}%)")

    if trainer is not None:
        import wandb

        trainer.log(metrics)
        if wandb.run is not None:
            wandb.config.update(metrics)
    return metrics


def compute_computational_efficiency_metrics(training_time, epoch_time, num_tokens, num_samples):
    """
    Compute computational efficiency metrics.
    """
    tokens_per_sec = num_tokens / training_time
    samples_per_sec = num_samples / training_time

    return {
        "training_time": training_time,
        "epoch_time": epoch_time,
        "tokens_per_sec": tokens_per_sec,
        "samples_per_sec": samples_per_sec,
    }


class EfficiencyMetricsCallback(TrainerCallback):
    def __init__(self):
        self.train_start_time = None
        self.epoch_start_time = None
        self.step_start_time = None
        self.mem_samples = []  # for "average" GPU memory across steps
        self.tokens_seen_this_log = 0
        self.samples_seen_this_log = 0
        self.last_log_time = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.train_start_time = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def on_epoch_begin(self, args, state, control, **kwargs):
        self.epoch_start_time = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def on_epoch_end(self, args, state, control, **kwargs):
        epoch_time = time.time() - self.epoch_start_time
        if state.is_world_process_zero:
            self._log({"time/epoch_seconds": epoch_time}, state)

    def on_step_begin(self, args, state, control, **kwargs):
        self.step_start_time = time.time()
        if self.last_log_time is None:
            self.last_log_time = self.step_start_time

    def on_step_end(self, args, state, control, **kwargs):
        # sample memory every step so the "average" is meaningful
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e9
            self.mem_samples.append(allocated)

        # accumulate tokens/samples between logging_steps intervals
        train_batch_size = args.per_device_train_batch_size * args.world_size
        self.samples_seen_this_log += train_batch_size

        # approx tokens: batch_size * seq_len;
        seq_len = getattr(args, "max_seq_length", None) or 2048
        self.tokens_seen_this_log += train_batch_size * seq_len

        if state.global_step % args.logging_steps == 0 and state.global_step > 0:
            now = time.time()
            elapsed = now - self.last_log_time
            metrics = {}

            if torch.cuda.is_available():
                metrics["memory/allocated_gb"] = torch.cuda.memory_allocated() / 1e9
                metrics["memory/reserved_gb"] = torch.cuda.memory_reserved() / 1e9

                if torch.distributed.is_initialized():
                    peak_tensor = torch.tensor(torch.cuda.max_memory_allocated(), device="cuda")
                    torch.distributed.all_reduce(peak_tensor, op=torch.distributed.ReduceOp.MAX)
                    metrics["memory/peak_gb_across_gpus"] = peak_tensor.item() / 1e9

                else:
                    metrics["memory/peak_gb_across_gpus"] = torch.cuda.max_memory_allocated() / 1e9

                if self.mem_samples:
                    metrics["memory/average_gb"] = sum(self.mem_samples) / len(self.mem_samples)

            if elapsed > 0:
                metrics["throughput/tokens_per_sec"] = self.tokens_seen_this_log / elapsed
                metrics["throughput/samples_per_sec"] = self.samples_seen_this_log / elapsed

            if state.is_world_process_zero:
                self._log(metrics, state)

            # reset the interval accumulators
            self.tokens_seen_this_log = 0
            self.samples_seen_this_log = 0
            self.last_log_time = now

    def on_train_end(self, args, state, control, **kwargs):
        total_time = time.time() - self.train_start_time
        if state.is_world_process_zero:
            self._log({"time/total_training_seconds": total_time}, state)

    def on_save(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return
        ckpt_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        if ckpt_dir.is_dir():
            size_bytes = sum(f.stat().st_size for f in ckpt_dir.rglob("*") if f.is_file())
            self._log({"checkpoint/size_gb": size_bytes / 1e9}, state)

    def _log(self, metrics, state):
        import wandb

        if wandb.run is not None:
            wandb.log(metrics, step=state.global_step)
