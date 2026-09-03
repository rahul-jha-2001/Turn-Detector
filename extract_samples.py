"""
extract_samples.py

Pulls a small set of representative audio clips from both datasets into
samples/ as plain .wav files - small enough to commit to git.

Deliberately avoids the `datasets` library's Audio-feature auto-decoding
path entirely (which requires torchcodec and has been the source of
repeated, unresolvable environment issues this session). Instead reads
raw Parquet files directly with pandas and decodes audio bytes with
soundfile - simple, no CUDA/native-library dependencies, works on CPU
regardless of torch/torchcodec state.
"""

import os
import glob
import io
import random
import csv
import pandas as pd
import soundfile as sf

random.seed(42)
OUT_DIR = "samples"


def extract_stage1_samples(n=12, target_language="hin"):
    print(f"Loading stage-1 test set (pandas, no torchcodec), language={target_language}...")
    files = sorted(glob.glob("data_full_raw/test/data/*.parquet"))

    out_dir = os.path.join(OUT_DIR, "stage1")
    os.makedirs(out_dir, exist_ok=True)

    manifest = []
    saved = 0

    for file_path in files:
        if saved >= n:
            break
        df = pd.read_parquet(
            file_path,
            columns=["audio", "id", "language", "endpoint_bool", "synthetic"],
        )
        df = df[df["language"] == target_language]
        if len(df) == 0:
            continue

        indices = list(range(len(df)))
        random.shuffle(indices)

        for idx in indices:
            if saved >= n:
                break
            row = df.iloc[idx]
            audio_field = row["audio"]
            audio_bytes = audio_field["bytes"] if isinstance(audio_field, dict) else audio_field

            try:
                waveform, sr = sf.read(io.BytesIO(audio_bytes))
            except Exception as e:
                print(f"  skipping unreadable row: {e}")
                continue

            fname = f"stage1_{saved:02d}_label{row['endpoint_bool']}_{row['language']}.wav"
            sf.write(os.path.join(out_dir, fname), waveform, sr)
            manifest.append({
                "file": fname,
                "label": bool(row["endpoint_bool"]),
                "language": row["language"],
                "synthetic": bool(row["synthetic"]),
            })
            saved += 1

    print(f"Saved {saved} stage-1 samples (language={target_language}) to {out_dir}/")
    return manifest


def extract_stage2_samples(n_hindi=10, n_other=5, other_languages=None):
    print("Loading stage-2 manifest...")
    manifest_path = "data_stage2_materialized/manifest.csv"
    with open(manifest_path, newline="") as f:
        rows = list(csv.DictReader(f))

    out_dir = os.path.join(OUT_DIR, "stage2")
    os.makedirs(out_dir, exist_ok=True)

    hindi_rows = [r for r in rows if r["language"] == "Hindi"]
    other_rows = [r for r in rows if r["language"] != "Hindi"]

    if other_languages:
        other_rows = [r for r in other_rows if r["language"] in other_languages]

    random.shuffle(hindi_rows)
    random.shuffle(other_rows)

    selected = hindi_rows[:n_hindi * 2] + other_rows[:n_other * 2]  # oversample, filter unreadable below
    random.shuffle(selected)

    saved = 0
    saved_hindi = 0
    saved_other = 0
    manifest = []

    for row in selected:
        if saved_hindi >= n_hindi and saved_other >= n_other:
            break
        is_hindi = row["language"] == "Hindi"
        if is_hindi and saved_hindi >= n_hindi:
            continue
        if not is_hindi and saved_other >= n_other:
            continue

        src_path = os.path.join("data_stage2_materialized", row["path"])
        try:
            waveform, sr = sf.read(src_path)
        except Exception as e:
            print(f"  skipping unreadable row: {e}")
            continue

        fname = f"stage2_{saved:02d}_label{row['label']}_{row['language']}.wav"
        sf.write(os.path.join(out_dir, fname), waveform, sr)
        manifest.append({
            "file": fname,
            "label": row["label"] == "True",
            "language": row["language"],
            "kind": row["kind"],
        })
        saved += 1
        if is_hindi:
            saved_hindi += 1
        else:
            saved_other += 1

    print(f"Saved {saved} stage-2 samples to {out_dir}/ "
          f"({saved_hindi} Hindi, {saved_other} other languages)")
    return manifest


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    stage1_manifest = extract_stage1_samples(n=12, target_language="hin")
    stage2_manifest = extract_stage2_samples(n_hindi=10, n_other=5)

    import json
    with open(os.path.join(OUT_DIR, "samples_manifest.json"), "w") as f:
        json.dump({"stage1": stage1_manifest, "stage2": stage2_manifest}, f, indent=2)

    print("\nDone. samples/ folder ready to commit to git.")


if __name__ == "__main__":
    main()