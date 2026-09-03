"""
evaluate_onnx.py

Evaluates ONNX models on both stage-1 (pipecat synthetic) test set and
stage-2 (real Indic dialogue) eval set. FP32 models run on GPU
(CUDAExecutionProvider), int8 models run on CPU (CPUExecutionProvider) -
int8 quantized ops aren't well-supported/accelerated by CUDA anyway,
and CPU is the actual deployment target these are optimized for.

Fixed bug from previous run: model.py's forward() already applies
sigmoid internally (see SmartTurnV3Model.forward() -> returns
{"logits": torch.sigmoid(logits)}), so the ONNX-exported model's output
IS the probability already - do NOT apply sigmoid again here. The
previous version double-applied sigmoid, causing every prediction to
collapse toward True (accuracy=0.504, recall=1.0 on stage1_test).
"""

import time
import numpy as np
import onnxruntime as ort
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import WhisperFeatureExtractor

from dataloader_stage2 import load_stage2_datasets

CHUNK_LENGTH_SECONDS = 8
SAMPLE_RATE = 16000
MAX_SAMPLES = 2000


def truncate_audio(audio_array, n_seconds=8, sr=16000):
    max_samples = int(n_seconds * sr)
    if len(audio_array) > max_samples:
        return audio_array[-max_samples:]
    return audio_array


def get_input_features(audio_array, sr, feature_extractor):
    audio_array = truncate_audio(audio_array, n_seconds=CHUNK_LENGTH_SECONDS, sr=sr)
    inputs = feature_extractor(
        audio_array, sampling_rate=sr, return_tensors="np",
        padding="max_length", max_length=CHUNK_LENGTH_SECONDS * sr,
        truncation=True, do_normalize=True,
    )
    return inputs.input_features.astype(np.float32)  # [1, 80, 800]


def make_session(onnx_path: str, use_gpu: bool):
    if use_gpu:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]
    session = ort.InferenceSession(onnx_path, providers=providers)
    actual_provider = session.get_providers()[0]
    print(f"  Session using provider: {actual_provider}")
    return session


def evaluate_onnx_model(onnx_path: str, dataset, get_audio_fn, get_label_fn, fe,
                         model_name: str, dataset_name: str, use_gpu: bool,
                         max_samples=MAX_SAMPLES):
    print(f"\n=== {model_name} on {dataset_name} ({'GPU' if use_gpu else 'CPU'}) ===")

    session = make_session(onnx_path, use_gpu)
    input_name = session.get_inputs()[0].name

    n = len(dataset) if max_samples is None else min(max_samples, len(dataset))

    preds, labels, latencies = [], [], []

    for i in range(n):
        sample = dataset[i]
        audio_array = get_audio_fn(sample)
        label = get_label_fn(sample)

        input_features = get_input_features(audio_array, SAMPLE_RATE, fe)

        t0 = time.perf_counter()
        logits = session.run(None, {input_name: input_features})[0]
        latency_ms = (time.perf_counter() - t0) * 1000

        # Model already applies sigmoid internally (see model.py forward()) -
        # ONNX output IS the probability, do not re-apply sigmoid.
        prob = logits.squeeze()
        pred = int(prob > 0.5)

        preds.append(pred)
        labels.append(int(label))
        latencies.append(latency_ms)

        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{n} processed...")

    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, zero_division=0)
    rec = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)

    latencies = np.array(latencies)
    print(f"n={n}  accuracy={acc:.4f}  precision={prec:.4f}  recall={rec:.4f}  f1={f1:.4f}")
    print(f"Latency (ms): mean={latencies.mean():.2f}  median={np.median(latencies):.2f}  "
          f"p95={np.percentile(latencies, 95):.2f}  min={latencies.min():.2f}  max={latencies.max():.2f}")

    return {
        "model": model_name, "dataset": dataset_name, "device": "GPU" if use_gpu else "CPU", "n": n,
        "accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
        "latency_mean_ms": float(latencies.mean()), "latency_median_ms": float(np.median(latencies)),
        "latency_p95_ms": float(np.percentile(latencies, 95)),
    }


def get_stage1_audio(s):
    audio = s["audio"]
    if isinstance(audio, dict):
        return audio["array"]
    return audio.get_all_samples().data.squeeze(0).numpy()


def main():
    fe = WhisperFeatureExtractor(chunk_length=CHUNK_LENGTH_SECONDS)

    print("Loading stage-1 test set...")
    stage1_test_ds = load_dataset("parquet", data_files="data_full_raw/test/data/*.parquet", split="train")

    print("Loading stage-2 eval set...")
    _, stage2_eval_ds = load_stage2_datasets()

    # (onnx_path, use_gpu) - fp32 on GPU, int8 (dynamic-quantized) on CPU
    onnx_models = {
        "stage1_fp32": ("onnx_models/stage1_final_fp32.onnx", True),
        "stage1_int8_dynamic": ("onnx_models/stage1_final_int8.onnx", False),
        "stage2_fp32": ("onnx_models/stage2_final_fp32.onnx", True),
        "stage2_int8_dynamic": ("onnx_models/stage2_final_int8.onnx", False),
    }

    results = []

    for model_name, (onnx_path, use_gpu) in onnx_models.items():
        r = evaluate_onnx_model(
            onnx_path, stage1_test_ds,
            get_audio_fn=get_stage1_audio,
            get_label_fn=lambda s: s["endpoint_bool"],
            fe=fe, model_name=model_name, dataset_name="stage1_test", use_gpu=use_gpu,
        )
        results.append(r)

        r = evaluate_onnx_model(
            onnx_path, stage2_eval_ds,
            get_audio_fn=lambda s: s["waveform"],
            get_label_fn=lambda s: s["label"],
            fe=fe, model_name=model_name, dataset_name="stage2_eval", use_gpu=use_gpu,
        )
        results.append(r)

    print(f"\n\n{'='*100}")
    print("FINAL SUMMARY")
    print(f"{'='*100}")
    print(f"{'Model':<22} {'Dataset':<14} {'Device':<6} {'Acc':>8} {'F1':>8} {'Lat(mean)':>12} {'Lat(p95)':>10}")
    for r in results:
        print(f"{r['model']:<22} {r['dataset']:<14} {r['device']:<6} {r['accuracy']:>8.4f} {r['f1']:>8.4f} "
              f"{r['latency_mean_ms']:>10.2f}ms {r['latency_p95_ms']:>8.2f}ms")


if __name__ == "__main__":
    main()