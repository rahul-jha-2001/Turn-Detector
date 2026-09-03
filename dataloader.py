"""
dataloader.py

Builds train/eval/test datasets directly from local raw Parquet shards
(downloaded via `hf download ... --local-dir data_full_raw/...`), NOT via
load_from_disk - we deliberately avoid datasets' Arrow-cache rebuild step
since it multiplies disk usage far beyond the raw compressed Parquet size.

Uses WhisperFeatureExtractor(chunk_length=8) to match pipecat's actual
config - this produces [80, 800] mel frames, matching model.py's
max_source_positions=400 (800 = 400 * 2, since Whisper's conv frontend
downsamples time by 2x).

NOTE: audio access uses torchcodec's AudioDecoder API (current datasets
version) - row["audio"] is an AudioDecoder, not a plain dict. Call
.get_all_samples() to get an AudioSamples dataclass with .data
(torch.Tensor, shape [channels, samples]) and .sample_rate.
"""

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import WhisperFeatureExtractor

CHUNK_LENGTH_SECONDS = 8
SAMPLE_RATE = 16000


class WhisperCollator:
    """Picklable collator (class, not closure) - needed for num_workers > 0
    to work without pickling errors."""

    def __init__(self, feature_extractor: WhisperFeatureExtractor):
        self.feature_extractor = feature_extractor

    def __call__(self, batch):
        audio_arrays = []
        sr = None
        for row in batch:
            audio_samples = row["audio"].get_all_samples()
            waveform = audio_samples.data.squeeze(0).numpy()  # [1, N] -> [N], numpy
            sr = audio_samples.sample_rate

            max_samples = int(CHUNK_LENGTH_SECONDS * sr)
            if len(waveform) > max_samples:
                waveform = waveform[-max_samples:]  # keep the END - turn-end signal concentrates there

            audio_arrays.append(waveform)

        inputs = self.feature_extractor(
            audio_arrays,
            sampling_rate=sr,
            return_tensors="pt",
            padding="max_length",
            max_length=CHUNK_LENGTH_SECONDS * sr,
            truncation=True,
            do_normalize=True,
        )

        labels = torch.tensor(
            [1.0 if row["endpoint_bool"] else 0.0 for row in batch],
            dtype=torch.float32,
        )

        meta = []
        for row in batch:
            meta.append({
                "language": row.get("language"),
                "synthetic": row.get("synthetic"),
                "midfiller": row.get("midfiller"),
                "endfiller": row.get("endfiller"),
                "dataset": row.get("dataset"),
            })

        return {
            "input_features": inputs.input_features,
            "labels": labels,
            "meta": meta,
        }


def load_datasets(
    train_parquet_glob: str = "data_full_raw/train/data/*.parquet",
    test_parquet_glob: str = "data_full_raw/test/data/*.parquet",
    eval_fraction: float = 0.1,
    seed: int = 42,
):
    """Loads raw HF Dataset objects (not DataLoaders) - for use with
    Trainer, which needs Dataset + data_collator, not pre-built DataLoaders."""
    print("Loading train parquet shards...")
    train_full = load_dataset("parquet", data_files=train_parquet_glob, split="train")
    print("Loading test parquet shards...")
    test_ds = load_dataset("parquet", data_files=test_parquet_glob, split="train")

    split = train_full.train_test_split(test_size=eval_fraction, seed=seed)
    train_ds, eval_ds = split["train"], split["test"]

    print(f"train: {len(train_ds)} | eval: {len(eval_ds)} | test: {len(test_ds)}")
    return train_ds, eval_ds, test_ds


def load_stage1_test_only(test_parquet_glob: str):
    """Loads ONLY the stage-1 test set, skipping the expensive unnecessary
    train-set load+split that load_datasets() does as a side effect."""
    return load_dataset("parquet", data_files=test_parquet_glob, split="train")

def build_dataloaders(
    train_parquet_glob: str = "data_full_raw/train/data/*.parquet",
    test_parquet_glob: str = "data_full_raw/test/data/*.parquet",
    batch_size: int = 16,
    num_workers: int = 0,
    eval_fraction: float = 0.1,
    seed: int = 42,
):
    """Same as load_datasets but returns DataLoaders instead - for manual
    training loops rather than Trainer."""
    train_ds, eval_ds, test_ds = load_datasets(
        train_parquet_glob, test_parquet_glob, eval_fraction, seed
    )

    fe = WhisperFeatureExtractor(chunk_length=CHUNK_LENGTH_SECONDS)
    collate_fn = WhisperCollator(fe)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
    )
    eval_loader = DataLoader(
        eval_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, eval_loader, test_loader


if __name__ == "__main__":
    train_loader, eval_loader, test_loader = build_dataloaders(batch_size=4)

    batch = next(iter(train_loader))
    print("\ninput_features shape:", batch["input_features"].shape)
    print("labels:", batch["labels"])
    print("meta[0]:", batch["meta"][0])