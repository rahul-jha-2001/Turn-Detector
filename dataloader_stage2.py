"""
dataloader_stage2.py

Stage 2: real Hinglish/Indic dialogue data (materialized clips).
From sarvamai/indic-diarbench, extracted via extract_diarbench_materialized.py.

Kept separate from dataloader.py (stage 1 / pipecat) since the two datasets
have different source formats (materialized .wav files here vs. HF Audio
column with lazy decode there).
"""

import csv
import os
import torch
import soundfile as sf
from torch.utils.data import Dataset, DataLoader
from transformers import WhisperFeatureExtractor

CHUNK_LENGTH_SECONDS = 8


class Stage2TurnDataset(Dataset):
    """Reads materialized .wav clips + manifest.csv produced by
    extract_diarbench_materialized.py."""

    def __init__(self, manifest_path: str, data_root: str):
        self.data_root = data_root
        self.rows = []
        with open(manifest_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                row["label"] = row["label"] == "True"
                self.rows.append(row)
        print(f"Loaded stage2 manifest: {len(self.rows)} clips from {manifest_path}")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        wav_path = os.path.join(self.data_root, row["path"])
        waveform, sr = sf.read(wav_path, dtype="float32")

        return {
            "waveform": waveform,
            "sample_rate": sr,
            "label": 1.0 if row["label"] else 0.0,
            "language": row["language"],
            "kind": row["kind"],
        }


class Stage2Collator:
    """Picklable collator (class, not closure)."""

    def __init__(self, feature_extractor: WhisperFeatureExtractor):
        self.feature_extractor = feature_extractor

    def __call__(self, batch):
        sr = batch[0]["sample_rate"]
        audio_arrays = [item["waveform"] for item in batch]

        inputs = self.feature_extractor(
            audio_arrays,
            sampling_rate=sr,
            return_tensors="pt",
            padding="max_length",
            max_length=int(CHUNK_LENGTH_SECONDS * sr),
            truncation=True,
            do_normalize=True,
        )

        labels = torch.tensor([item["label"] for item in batch], dtype=torch.float32)
        meta = [{"language": item["language"], "kind": item["kind"]} for item in batch]

        return {
            "input_features": inputs.input_features,
            "labels": labels,
            "meta": meta,
        }


def load_stage2_datasets(
    manifest_path: str = "data_stage2_materialized/manifest.csv",
    data_root: str = "data_stage2_materialized",
    eval_fraction: float = 0.15,
    seed: int = 42,
):
    """Returns (train_ds, eval_ds) as torch Subset objects."""
    full_dataset = Stage2TurnDataset(manifest_path, data_root)

    n = len(full_dataset)
    n_eval = int(n * eval_fraction)
    n_train = n - n_eval

    generator = torch.Generator().manual_seed(seed)
    train_ds, eval_ds = torch.utils.data.random_split(
        full_dataset, [n_train, n_eval], generator=generator
    )
    print(f"Stage 2 - train: {n_train} | eval: {n_eval}")
    return train_ds, eval_ds


def build_stage2_dataloaders(
    manifest_path: str = "data_stage2_materialized/manifest.csv",
    data_root: str = "data_stage2_materialized",
    batch_size: int = 16,
    num_workers: int = 0,
    eval_fraction: float = 0.15,
    seed: int = 42,
):
    train_ds, eval_ds = load_stage2_datasets(manifest_path, data_root, eval_fraction, seed)

    fe = WhisperFeatureExtractor(chunk_length=CHUNK_LENGTH_SECONDS)
    collate_fn = Stage2Collator(fe)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, collate_fn=collate_fn,
    )
    eval_loader = DataLoader(
        eval_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=collate_fn,
    )
    return train_loader, eval_loader


if __name__ == "__main__":
    train_loader, eval_loader = build_stage2_dataloaders(batch_size=4)
    batch = next(iter(train_loader))
    print("\ninput_features shape:", batch["input_features"].shape)
    print("labels:", batch["labels"])
    print("meta[0]:", batch["meta"][0])