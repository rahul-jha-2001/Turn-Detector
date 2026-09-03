"""
final_eval.py

Runs both checkpoints (stage-1 and stage-2) against both test sets
(pipecat stage-1 synthetic test, and stage-2 real Indic dialogue eval),
with per-language breakdowns. Produces the final comparison table for
the report.
"""

import os
import torch
from safetensors.torch import load_file
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import Trainer, TrainingArguments, WhisperFeatureExtractor, WhisperConfig

from model import SmartTurnV3Model
from dataloader import WhisperCollator, CHUNK_LENGTH_SECONDS
from dataloader_stage2 import load_stage2_datasets, Stage2Collator


def load_finetuned_model(checkpoint_dir: str) -> SmartTurnV3Model:
    config = WhisperConfig.from_pretrained(checkpoint_dir)
    model = SmartTurnV3Model(config)
    safetensors_path = os.path.join(checkpoint_dir, "model.safetensors")
    bin_path = os.path.join(checkpoint_dir, "pytorch_model.bin")
    if os.path.exists(safetensors_path):
        state_dict = load_file(safetensors_path)
    elif os.path.exists(bin_path):
        state_dict = torch.load(bin_path, map_location="cpu")
    else:
        raise FileNotFoundError(f"No weights file in {checkpoint_dir}")
    model.load_state_dict(state_dict, strict=False)
    return model


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = logits.squeeze()
    preds = (probs > 0.5).astype(int)
    labels = labels.astype(int)
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0),
    }


def slice_accuracy(trainer, dataset, get_key_fn, label_name, top_n=None):
    predictions = trainer.predict(dataset)
    probs = predictions.predictions.squeeze()
    preds = (probs > 0.5).astype(int)
    labels = predictions.label_ids.astype(int)

    slice_stats = {}
    for idx in range(len(dataset)):
        item = dataset[idx]
        key = get_key_fn(item)
        slice_stats.setdefault(key, [0, 0])
        slice_stats[key][0] += int(preds[idx] == labels[idx])
        slice_stats[key][1] += 1

    print(f"\n  --- Accuracy by {label_name} ---")
    sorted_items = sorted(slice_stats.items(), key=lambda x: -x[1][0] / x[1][1])
    if top_n:
        sorted_items = sorted_items[:top_n]
    for k, (correct, total) in sorted_items:
        print(f"    {str(k):>15}: {correct/total:.4f}  (n={total})")
    return slice_stats


def evaluate_checkpoint_on_stage1(checkpoint_name, checkpoint_dir, stage1_test_ds, fe):
    print(f"\n{'='*70}")
    print(f"CHECKPOINT: {checkpoint_name}  |  DATASET: stage-1 (pipecat synthetic test)")
    print(f"{'='*70}")

    model = load_finetuned_model(checkpoint_dir)
    collate_fn = WhisperCollator(fe)

    trainer = Trainer(
        model=model,
        args=TrainingArguments(output_dir="/tmp/final_eval", per_device_eval_batch_size=16,
                                report_to=[], remove_unused_columns=False, fp16=True,
                                dataloader_num_workers=0),
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
    )
    metrics = trainer.evaluate(stage1_test_ds)
    print(f"Overall: accuracy={metrics['eval_accuracy']:.4f} f1={metrics['eval_f1']:.4f}")

    slice_accuracy(trainer, stage1_test_ds, lambda item: item["language"], "language")
    return metrics


def evaluate_checkpoint_on_stage2(checkpoint_name, checkpoint_dir, stage2_eval_ds, fe):
    print(f"\n{'='*70}")
    print(f"CHECKPOINT: {checkpoint_name}  |  DATASET: stage-2 (real Indic dialogue)")
    print(f"{'='*70}")

    model = load_finetuned_model(checkpoint_dir)
    collate_fn = Stage2Collator(fe)

    trainer = Trainer(
        model=model,
        args=TrainingArguments(output_dir="/tmp/final_eval", per_device_eval_batch_size=16,
                                report_to=[], remove_unused_columns=False, fp16=True,
                                dataloader_num_workers=0),
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
    )
    metrics = trainer.evaluate(stage2_eval_ds)
    print(f"Overall: accuracy={metrics['eval_accuracy']:.4f} f1={metrics['eval_f1']:.4f}")

    slice_accuracy(trainer, stage2_eval_ds, lambda item: item["language"], "language")
    slice_accuracy(trainer, stage2_eval_ds, lambda item: item["kind"], "kind")
    return metrics


def main():
    fe = WhisperFeatureExtractor(chunk_length=CHUNK_LENGTH_SECONDS)

    print("Loading datasets...")
    stage1_test_ds = load_dataset("parquet", data_files="data_full_raw/test/data/*.parquet", split="train")
    _, stage2_eval_ds = load_stage2_datasets()

    checkpoints = {
        "stage1_final": "checkpoints/final_model",
        "stage2_final": "checkpoints_stage2/final_model",
    }

    results = {}
    for name, path in checkpoints.items():
        results[(name, "stage1")] = evaluate_checkpoint_on_stage1(name, path, stage1_test_ds, fe)
        results[(name, "stage2")] = evaluate_checkpoint_on_stage2(name, path, stage2_eval_ds, fe)

    print(f"\n\n{'='*70}")
    print("FINAL SUMMARY TABLE")
    print(f"{'='*70}")
    print(f"{'Checkpoint':<15} {'Dataset':<10} {'Accuracy':>10} {'F1':>10}")
    for (name, dset), metrics in results.items():
        print(f"{name:<15} {dset:<10} {metrics['eval_accuracy']:>10.4f} {metrics['eval_f1']:>10.4f}")


if __name__ == "__main__":
    main()