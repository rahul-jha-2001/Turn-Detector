"""
train.py

Uses HF Trainer directly, matching pipecat's own train.py approach.
Data loads directly from local raw Parquet shards (via dataloader.load_datasets).

Extra logging added throughout so progress/timing/config is visible in
real time on a long run, not just at the end.
"""

import argparse
import logging
import time
import sys
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import Trainer, TrainingArguments, WhisperFeatureExtractor, TrainerCallback

from model import load_model
from dataloader import load_datasets, WhisperCollator, CHUNK_LENGTH_SECONDS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger("train")


class TimingCallback(TrainerCallback):
    """Logs epoch/step timing and throughput - makes it obvious whether
    data loading or GPU compute is the bottleneck, and gives a live ETA."""

    def on_train_begin(self, args, state, control, **kwargs):
        self.train_start = time.time()
        self.epoch_start = time.time()
        log.info(f"Training started: {state.max_steps} total steps planned "
                  f"({args.num_train_epochs} epochs)")

    def on_epoch_begin(self, args, state, control, **kwargs):
        self.epoch_start = time.time()
        log.info(f"--- Epoch {int(state.epoch) + 1}/{int(args.num_train_epochs)} starting ---")

    def on_epoch_end(self, args, state, control, **kwargs):
        elapsed = time.time() - self.epoch_start
        log.info(f"--- Epoch {int(state.epoch)} finished in {elapsed/60:.1f} min ---")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        elapsed = time.time() - self.train_start
        pct = 100 * state.global_step / max(state.max_steps, 1)
        parts = [f"step {state.global_step}/{state.max_steps} ({pct:.1f}%)",
                 f"elapsed {elapsed/60:.1f}min"]
        for k in ("loss", "eval_loss", "eval_accuracy", "eval_f1",
                  "test_loss", "test_accuracy", "test_f1", "learning_rate"):
            if k in logs:
                parts.append(f"{k}={logs[k]:.4f}" if isinstance(logs[k], float) else f"{k}={logs[k]}")
        log.info(" | ".join(parts))

    def on_train_end(self, args, state, control, **kwargs):
        elapsed = time.time() - self.train_start
        log.info(f"Training complete in {elapsed/60:.1f} min total "
                  f"({elapsed/max(state.global_step,1):.2f} sec/step avg)")


def log_dataset_stats(name: str, dataset):
    """Prints label balance and a couple of sanity-check fields so you can
    eyeball whether the data looks right before spending hours training on it."""
    log.info(f"[{name}] {len(dataset)} rows, columns: {dataset.column_names}")
    if "endpoint_bool" in dataset.column_names:
        labels = dataset["endpoint_bool"]
        pos = sum(1 for v in labels if v)
        log.info(f"[{name}] label balance: {pos}/{len(labels)} positive "
                 f"({100*pos/len(labels):.1f}%)")
    if "language" in dataset.column_names:
        from collections import Counter
        langs = Counter(dataset["language"])
        top5 = langs.most_common(5)
        log.info(f"[{name}] top languages: {top5}")
    if "synthetic" in dataset.column_names:
        synth = dataset["synthetic"]
        synth_pct = 100 * sum(1 for v in synth if v) / len(synth)
        log.info(f"[{name}] synthetic: {synth_pct:.1f}%")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = logits.squeeze()
    preds = (probs > 0.5).astype(int)
    labels = labels.astype(int)
    metrics = {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0),
    }
    log.info(f"[eval] n={len(labels)} " + " ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
    return metrics


def slice_accuracy(trainer, dataset, column: str):
    log.info(f"Running slice_accuracy predictions on {len(dataset)} rows...")
    t0 = time.time()
    predictions = trainer.predict(dataset)
    log.info(f"Prediction pass took {time.time()-t0:.1f}s")

    probs = predictions.predictions.squeeze()
    preds = (probs > 0.5).astype(int)
    labels = predictions.label_ids.astype(int)

    if column not in dataset.column_names:
        log.warning(f"Column '{column}' not present, skipping slice breakdown")
        return {}

    values = dataset[column]
    slice_stats = {}
    for v, p, l in zip(values, preds, labels):
        key = str(v) if v is not None else "unknown"
        slice_stats.setdefault(key, [0, 0])
        slice_stats[key][0] += int(p == l)
        slice_stats[key][1] += 1

    slice_acc = {k: c / t for k, (c, t) in slice_stats.items()}
    log.info(f"=== Accuracy by {column} ===")
    for k, acc in sorted(slice_acc.items(), key=lambda x: -x[1]):
        log.info(f"    {k:>6}: {acc:.4f}  (n={slice_stats[k][1]})")
    return slice_acc


def main(
    train_parquet_glob: str = "data_full_raw/train/data/*.parquet",
    test_parquet_glob: str = "data_full_raw/test/data/*.parquet",
    out_dir: str = "checkpoints",
    batch_size: int = 16,
    num_epochs: int = 4,
    learning_rate: float = 5e-5,
    warmup_ratio: float = 0.2,
    run_name: str = "smart-turn-repro",
    slice_by: str = "language",
):
    log.info("=" * 60)
    log.info(f"CONFIG: batch_size={batch_size} epochs={num_epochs} "
              f"lr={learning_rate} warmup_ratio={warmup_ratio} run_name={run_name}")
    log.info("=" * 60)

    import torch
    log.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        log.info(f"GPU: {torch.cuda.get_device_name(0)}, "
                  f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    log.info("Loading model...")
    t0 = time.time()
    model = load_model()
    n_params = sum(p.numel() for p in model.parameters())
    log.info(f"Model loaded in {time.time()-t0:.1f}s - {n_params:,} params")

    log.info("Loading data...")
    t0 = time.time()
    train_ds, eval_ds, test_ds = load_datasets(
        train_parquet_glob=train_parquet_glob,
        test_parquet_glob=test_parquet_glob,
    )
    log.info(f"Data loaded in {time.time()-t0:.1f}s")
    log_dataset_stats("train", train_ds)
    log_dataset_stats("eval", eval_ds)
    log_dataset_stats("test", test_ds)

    fe = WhisperFeatureExtractor(chunk_length=CHUNK_LENGTH_SECONDS)
    collate_fn = WhisperCollator(fe)

    total_steps = (len(train_ds) // batch_size) * num_epochs
    warmup_steps = int(warmup_ratio * total_steps)
    log.info(f"Total training steps: {total_steps}, warmup steps: {warmup_steps}")

    training_args = TrainingArguments(
        output_dir=out_dir,
        run_name=run_name,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        lr_scheduler_type="cosine",
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=20,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        report_to=["wandb"],
        remove_unused_columns=False,
        fp16=True,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
        callbacks=[TimingCallback()],
    )

    log.info("Starting training...")
    trainer.train()

    log.info("=== Final eval-set metrics (in-distribution) ===")
    eval_metrics = trainer.evaluate(eval_ds)
    log.info(eval_metrics)

    log.info("=== Final test-set metrics (out-of-distribution) ===")
    test_metrics = trainer.evaluate(test_ds, metric_key_prefix="test")
    log.info(test_metrics)

    slice_accuracy(trainer, test_ds, slice_by)

    final_path = f"{out_dir}/final_model"
    trainer.save_model(final_path)
    log.info(f"Model saved to {final_path}")
    log.info("Done.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train_parquet_glob", default="data_full_raw/train/data/*.parquet")
    p.add_argument("--test_parquet_glob", default="data_full_raw/test/data/*.parquet")
    p.add_argument("--out_dir", default="checkpoints")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_epochs", type=int, default=4)
    p.add_argument("--learning_rate", type=float, default=5e-5)
    p.add_argument("--warmup_ratio", type=float, default=0.2)
    p.add_argument("--run_name", type=str, default="smart-turn-repro")
    p.add_argument("--slice_by", type=str, default="language")
    args = p.parse_args()

    main(
        train_parquet_glob=args.train_parquet_glob,
        test_parquet_glob=args.test_parquet_glob,
        out_dir=args.out_dir,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        run_name=args.run_name,
        slice_by=args.slice_by,
    )