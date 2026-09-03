"""
train_stage2.py

Stage 2: fine-tune the stage-1 checkpoint on real Hinglish/Indic dialogue
data (sarvamai/indic-diarbench, materialized via extract_diarbench_materialized.py).

Confirmed zero-shot baseline (separate run, zero_shot_eval_stage2.py):
  stage2 eval_accuracy=0.5885, eval_f1=0.5138
Not recomputed here - skips ~50 min of redundant evaluation before training
even starts. Stage-1's own confirmed test accuracy (93.4%, from the
original full training run) is the pre-fine-tune reference point for the
forgetting check.

Tracks stage-1 test set accuracy every epoch (forgetting check) alongside
stage-2 eval accuracy (adaptation check). All artifacts (checkpoints,
final model, forgetting curve, full log) saved under out_dir (default:
checkpoints_stage2/), kept separate from stage-1's checkpoints/.

Uses load_finetuned_model() instead of from_pretrained() to avoid the
transformers 5.16.1 tied-weights finalization bug (confirmed fix).

Uses load_stage1_test_only() instead of load_datasets() for the forgetting
check - load_datasets() would otherwise load+split the FULL 271k-row train
set as an unnecessary side effect just to get the test split.

dataloader_num_workers=0 - Python 3.14's multiprocessing forkserver is
broken (confirmed: UnpicklingError / "pickle data was truncated" on
worker init). Single-process loading is slower (~5-10 rows/sec) but
reliable - confirmed no hangs.
"""

import argparse
import json
import logging
import os
import sys
import torch
from safetensors.torch import load_file
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import Trainer, TrainingArguments, WhisperFeatureExtractor, TrainerCallback, WhisperConfig

from model import SmartTurnV3Model
from dataloader import WhisperCollator, CHUNK_LENGTH_SECONDS
from dataloader_stage2 import load_stage2_datasets, Stage2Collator

KNOWN_ZERO_SHOT_STAGE2_ACC = 0.5885
KNOWN_ZERO_SHOT_STAGE2_F1 = 0.5138
KNOWN_STAGE1_TEST_ACC = 0.9336  # from original stage-1 training run


def setup_logging(out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, "train_stage2.log")

    logger = logging.getLogger("train_stage2")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")

    file_handler = logging.FileHandler(log_path, mode="a")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    print(f"Logging to {log_path} (and console)")
    return logger


def load_finetuned_model(checkpoint_dir: str, log) -> SmartTurnV3Model:
    """Loads a saved SmartTurnV3Model checkpoint WITHOUT going through
    PreTrainedModel.from_pretrained()'s finalization step (crashes on this
    transformers version's tied-weights bookkeeping)."""
    config = WhisperConfig.from_pretrained(checkpoint_dir)
    model = SmartTurnV3Model(config)

    safetensors_path = os.path.join(checkpoint_dir, "model.safetensors")
    bin_path = os.path.join(checkpoint_dir, "pytorch_model.bin")

    if os.path.exists(safetensors_path):
        state_dict = load_file(safetensors_path)
    elif os.path.exists(bin_path):
        state_dict = torch.load(bin_path, map_location="cpu")
    else:
        raise FileNotFoundError(f"No weights file found in {checkpoint_dir}")

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    log.info(f"Loaded checkpoint - missing: {missing}, unexpected: {unexpected}")
    return model


def load_stage1_test_only(test_parquet_glob: str, log):
    """Loads ONLY the stage-1 test set directly - skips the expensive,
    unnecessary train-set load+split that load_datasets() does as a side
    effect."""
    log.info(f"Loading stage-1 test set directly from {test_parquet_glob}...")
    return load_dataset("parquet", data_files=test_parquet_glob, split="train")


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


class Stage1ForgettingCallback(TrainerCallback):
    """Evaluates on the STAGE-1 test set every time Trainer evaluates on
    the stage-2 eval set (every epoch, given eval_strategy='epoch')."""

    def __init__(self, trainer_ref_holder, stage1_test_ds, stage1_collate, log):
        self.trainer_ref_holder = trainer_ref_holder
        self.stage1_test_ds = stage1_test_ds
        self.stage1_collate = stage1_collate
        self.log = log
        self.history = []
        self._last_evaluated_epoch = None  # guard against duplicate calls

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        # Trainer may call evaluate() more than once around the same epoch
        # boundary (e.g. load_best_model_at_end bookkeeping) - skip repeats.
        if self._last_evaluated_epoch is not None and abs(self._last_evaluated_epoch - state.epoch) < 1e-6:
            return
        self._last_evaluated_epoch = state.epoch
        trainer = self.trainer_ref_holder["trainer"]
        original_collator = trainer.data_collator

        trainer.data_collator = self.stage1_collate
        stage1_metrics = trainer.evaluate(self.stage1_test_ds, metric_key_prefix="stage1test")
        trainer.data_collator = original_collator

        row = {
            "epoch": state.epoch,
            "stage2_eval_accuracy": metrics.get("eval_accuracy") if metrics else None,
            "stage2_eval_f1": metrics.get("eval_f1") if metrics else None,
            "stage1test_accuracy": stage1_metrics.get("stage1test_accuracy"),
            "stage1test_f1": stage1_metrics.get("stage1test_f1"),
        }
        self.history.append(row)
        self.log.info(f"[epoch {state.epoch:.1f}] stage2_acc={row['stage2_eval_accuracy']:.4f} "
                       f"stage1test_acc={row['stage1test_accuracy']:.4f}")


def main(
    stage1_checkpoint: str = "checkpoints/final_model",
    stage2_manifest: str = "data_stage2_materialized/manifest.csv",
    stage2_data_root: str = "data_stage2_materialized",
    stage1_test_parquet_glob: str = "data_full_raw/test/data/*.parquet",
    out_dir: str = "checkpoints_stage2",
    batch_size: int = 16,
    num_epochs: int = 3,
    learning_rate: float = 5e-6,
    warmup_ratio: float = 0.1,
    run_name: str = "smart-turn-stage2",
):
    log = setup_logging(out_dir)

    log.info("=" * 60)
    log.info(f"CONFIG: lr={learning_rate} epochs={num_epochs} batch_size={batch_size} "
              f"stage1_checkpoint={stage1_checkpoint} out_dir={out_dir}")
    log.info(f"Known baselines (not recomputed): zero-shot stage2_acc={KNOWN_ZERO_SHOT_STAGE2_ACC:.4f}, "
              f"stage1 test_acc={KNOWN_STAGE1_TEST_ACC:.4f}")
    log.info("=" * 60)

    log.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        log.info(f"GPU: {torch.cuda.get_device_name(0)}, "
                  f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    log.info(f"Loading stage-1 checkpoint from {stage1_checkpoint}...")
    model = load_finetuned_model(stage1_checkpoint, log)
    n_params = sum(p.numel() for p in model.parameters())
    log.info(f"Stage-1 checkpoint loaded - {n_params:,} params")

    fe = WhisperFeatureExtractor(chunk_length=CHUNK_LENGTH_SECONDS)

    log.info("Loading stage-2 (real Indic dialogue) train/eval data...")
    stage2_train, stage2_eval = load_stage2_datasets(stage2_manifest, stage2_data_root)
    stage2_collate = Stage2Collator(fe)

    stage1_test_ds = load_stage1_test_only(stage1_test_parquet_glob, log)
    log.info(f"Stage-1 test set loaded: {len(stage1_test_ds)} rows")
    stage1_collate = WhisperCollator(fe)

    total_steps = (len(stage2_train) // batch_size) * num_epochs
    warmup_steps = int(warmup_ratio * total_steps)
    log.info(f"Total stage-2 steps: {total_steps}, warmup: {warmup_steps}")

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
        dataloader_num_workers=0,
    )

    trainer_ref_holder = {}
    forgetting_callback = Stage1ForgettingCallback(trainer_ref_holder, stage1_test_ds, stage1_collate, log)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=stage2_train,
        eval_dataset=stage2_eval,
        data_collator=stage2_collate,
        compute_metrics=compute_metrics,
        callbacks=[forgetting_callback],
    )
    trainer_ref_holder["trainer"] = trainer
    log.info(f"Trainer is using device: {trainer.args.device}")

    # NOTE: no zero-shot pre-eval here - both baselines are already known
    # (see KNOWN_ZERO_SHOT_STAGE2_ACC / KNOWN_STAGE1_TEST_ACC above), so we
    # skip straight to training. Saves ~50 min of redundant evaluation.
    log.info("Starting stage-2 fine-tuning...")
    trainer.train()

    log.info("=== FULL forgetting-vs-adaptation curve ===")
    log.info(f"  epoch 0 (known baseline): stage2_acc={KNOWN_ZERO_SHOT_STAGE2_ACC:.4f} "
              f"stage1test_acc={KNOWN_STAGE1_TEST_ACC:.4f}")
    for row in forgetting_callback.history:
        log.info(f"  epoch {row['epoch']:.1f}: stage2_acc={row['stage2_eval_accuracy']:.4f} "
                  f"stage1test_acc={row['stage1test_accuracy']:.4f}")

    curve_path = f"{out_dir}/forgetting_curve.json"
    full_curve = [{
        "epoch": 0.0,
        "stage2_eval_accuracy": KNOWN_ZERO_SHOT_STAGE2_ACC,
        "stage2_eval_f1": KNOWN_ZERO_SHOT_STAGE2_F1,
        "stage1test_accuracy": KNOWN_STAGE1_TEST_ACC,
        "stage1test_f1": None,
        "note": "epoch 0 values are known baselines from separate prior runs, not recomputed here",
    }] + forgetting_callback.history
    with open(curve_path, "w") as f:
        json.dump(full_curve, f, indent=2)
    log.info(f"Forgetting-vs-adaptation curve saved to {curve_path}")

    final_path = f"{out_dir}/final_model"
    trainer.save_model(final_path)
    log.info(f"Model saved to {final_path}")
    log.info("Done.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--stage1_checkpoint", default="checkpoints/final_model")
    p.add_argument("--stage2_manifest", default="data_stage2_materialized/manifest.csv")
    p.add_argument("--stage2_data_root", default="data_stage2_materialized")
    p.add_argument("--stage1_test_parquet_glob", default="data_full_raw/test/data/*.parquet")
    p.add_argument("--out_dir", default="checkpoints_stage2")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_epochs", type=int, default=3)
    p.add_argument("--learning_rate", type=float, default=5e-6)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--run_name", default="smart-turn-stage2")
    args = p.parse_args()

    main(
        stage1_checkpoint=args.stage1_checkpoint,
        stage2_manifest=args.stage2_manifest,
        stage2_data_root=args.stage2_data_root,
        stage1_test_parquet_glob=args.stage1_test_parquet_glob,
        out_dir=args.out_dir,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        run_name=args.run_name,
    )