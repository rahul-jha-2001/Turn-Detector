"""
extract_diarbench_materialized.py

Extracts and MATERIALIZES labeled turn-detection clips from
sarvamai/indic-diarbench, across all languages - writes actual .wav files
to disk plus a manifest CSV.

=== Confirmed extraction logic ===

Step 1 - Substantive segment filter:
  - Strip all <...> and (...) bracketed content from the transcript.
  - Not substantive if nothing real remains after stripping (pure noise/
    annotation tag, any spelling: <unintelligible>, <noise>, <laughter>,
    <background speech>, etc. - and their many misspelled variants).
  - Not substantive if duration < MIN_SUBSTANTIVE_DURATION (1.0s) even with
    real text (backchannel-style acknowledgments like "haan", "ji").
  - Otherwise: substantive, eligible as a candidate.

Step 2 - Cross-segment examples (consecutive substantive segment pairs):
  - Window = [end_time - CLIP_SECONDS, end_time], clamped to 0.
  - Reject if window < MIN_CLIP_DURATION (3.0s) - not enough context near
    recording start.
  - Reject if window is "dirty": any OTHER segment in the ENTIRE recording,
    different speaker, with real speech (after tag-stripping) overlaps the
    window at all. Same-speaker segments and pure-noise/tag segments never
    count as contamination.
  - If clean: same speaker as next segment -> label FALSE (paused, continued).
             different speaker -> label TRUE (genuine floor change).

Step 3 - Within-segment mid-cut examples (extra negatives, long segments):
  - For substantive segments > CLIP_SECONDS long: cut at the segment's
    midpoint (not its end). Same clean-window check. If clean -> label FALSE.

Output:
  <out_dir>/<language>/<clip_id>.wav  (PCM_16, mono, native sample rate)
  <out_dir>/manifest.csv  (path, language, label, kind, speaker_id, row_idx, end_time)
"""

import glob
import os
import re
import csv
import argparse
from datasets import load_dataset
import soundfile as sf

BRACKET_PATTERN = re.compile(r"<[^>]*>|\([^)]*\)")
MIN_SUBSTANTIVE_DURATION = 1.0
CLIP_SECONDS = 8.0
MIN_CLIP_DURATION = 3.0

DATA_ROOT = "data_diarbench_raw"


def get_languages():
    return sorted([
        d for d in os.listdir(DATA_ROOT)
        if os.path.isdir(f"{DATA_ROOT}/{d}") and d != ".cache"
    ])


def strip_tags(text: str) -> str:
    return BRACKET_PATTERN.sub("", text).strip()


def is_substantive(seg) -> bool:
    dur = seg["end_time"] - seg["start_time"]
    if dur <= 0 or dur < MIN_SUBSTANTIVE_DURATION:
        return False
    return len(strip_tags(seg["transcript"])) > 0


def has_real_speech(seg) -> bool:
    """Used only for window-contamination checks - ignores the duration
    threshold (even a short real-speech segment counts as contamination)."""
    return len(strip_tags(seg["transcript"])) > 0


def window_is_clean(segs, window_start, window_end, speaker_id, exclude_idx) -> bool:
    for k, seg in enumerate(segs):
        if k == exclude_idx:
            continue
        if seg["speaker_id"] == speaker_id:
            continue
        if not has_real_speech(seg):
            continue
        if seg["start_time"] < window_end and seg["end_time"] > window_start:
            return False
    return True


def build_candidates_for_sample(segs):
    """Returns list of (window_start, end_time, label, kind, speaker_id)."""
    flags = [is_substantive(s) for s in segs]
    idx = [i for i, f in enumerate(flags) if f]

    candidates = []

    # Step 2: cross-segment
    for pos in range(len(idx) - 1):
        i, j = idx[pos], idx[pos + 1]
        cur, nxt = segs[i], segs[j]
        same_speaker = cur["speaker_id"] == nxt["speaker_id"]

        end_time = cur["end_time"]
        window_start = max(0.0, end_time - CLIP_SECONDS)
        if end_time - window_start < MIN_CLIP_DURATION:
            continue
        if not window_is_clean(segs, window_start, end_time, cur["speaker_id"], i):
            continue

        label = False if same_speaker else True
        candidates.append((window_start, end_time, label, "cross_segment", cur["speaker_id"]))

    # Step 3: within-segment mid-cut
    for i in idx:
        seg = segs[i]
        dur = seg["end_time"] - seg["start_time"]
        if dur > CLIP_SECONDS:
            mid_time = seg["start_time"] + dur * 0.5
            window_start = max(0.0, mid_time - CLIP_SECONDS)
            if window_is_clean(segs, window_start, mid_time, seg["speaker_id"], i):
                candidates.append((window_start, mid_time, False, "mid_cut", seg["speaker_id"]))

    return candidates


def process_language(language, out_dir, manifest_writer, max_per_language=None):
    files = sorted(glob.glob(f"{DATA_ROOT}/{language}/**/*.parquet", recursive=True))
    if not files:
        print(f"  [{language}] no parquet files found, skipping")
        return 0

    ds = load_dataset("parquet", data_files=files, split="train")
    lang_dir = os.path.join(out_dir, language)
    os.makedirs(lang_dir, exist_ok=True)

    n_written = 0
    for row_idx in range(len(ds)):
        sample = ds[row_idx]
        segs = sample["annotated_transcript"]
        candidates = build_candidates_for_sample(segs)

        for k, (window_start, end_time, label, kind, speaker_id) in enumerate(candidates):
            try:
                clip = sample["audio"].get_samples_played_in_range(window_start, end_time)
            except Exception:
                continue  # skip malformed timing ranges rather than crash the whole run
            waveform = clip.data.squeeze(0).numpy()
            sr = clip.sample_rate

            clip_id = f"{language}_{row_idx}_{k}"
            rel_path = f"{language}/{clip_id}.wav"
            abs_path = os.path.join(out_dir, rel_path)

            sf.write(abs_path, waveform, sr, subtype="PCM_16")

            manifest_writer.writerow({
                "path": rel_path, "language": language, "label": label,
                "kind": kind, "speaker_id": speaker_id,
                "row_idx": row_idx, "end_time": round(end_time, 3),
                "duration": round(end_time - window_start, 3),
            })
            n_written += 1

        if max_per_language and n_written >= max_per_language:
            break

    print(f"  [{language}] {len(ds)} recordings -> {n_written} clips written")
    return n_written


def main(out_dir="data_stage2_materialized", languages=None, max_per_language=None):
    os.makedirs(out_dir, exist_ok=True)
    languages = languages or get_languages()
    print(f"Extracting from {len(languages)} languages: {languages}\n")

    manifest_path = os.path.join(out_dir, "manifest.csv")
    total = 0
    with open(manifest_path, "w", newline="") as f:
        fieldnames = ["path", "language", "label", "kind", "speaker_id",
                      "row_idx", "end_time", "duration"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for lang in languages:
            print(f"Processing {lang}...")
            n = process_language(lang, out_dir, writer, max_per_language)
            total += n

    print(f"\nTotal clips written: {total}")
    print(f"Manifest saved to {manifest_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", default="data_stage2_materialized")
    p.add_argument("--languages", nargs="*", default=None,
                    help="specific languages, e.g. --languages Hindi Bengali. Omit for all.")
    p.add_argument("--max_per_language", type=int, default=None,
                    help="cap clips per language (for a quick test run)")
    args = p.parse_args()
    main(args.out_dir, args.languages, args.max_per_language)