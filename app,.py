"""
app.py

Gradio demo: live inference with stage-1/stage-2 ONNX models, sample
clips from both datasets, and a placeholder Report tab (filled in later
with tables/charts).
"""

import json
import numpy as np
import onnxruntime as ort
import gradio as gr
from transformers import WhisperFeatureExtractor

CHUNK_LENGTH_SECONDS = 8
SAMPLE_RATE = 16000

fe = WhisperFeatureExtractor(chunk_length=CHUNK_LENGTH_SECONDS)

MODELS = {
    "Stage 1 (pipecat reproduction)": "onnx_models/stage1_final_fp32.onnx",
    "Stage 2 (fine-tuned on real Indic dialogue)": "onnx_models/stage2_final_fp32.onnx",
}
_sessions = {}


def get_session(model_key):
    if model_key not in _sessions:
        _sessions[model_key] = ort.InferenceSession(MODELS[model_key], providers=["CPUExecutionProvider"])
    return _sessions[model_key]


def truncate_audio(audio_array, n_seconds=8, sr=16000):
    max_samples = int(n_seconds * sr)
    if len(audio_array) > max_samples:
        return audio_array[-max_samples:]
    return audio_array


def predict(audio, model_choice):
    if audio is None:
        return "No audio provided."

    sr, waveform = audio
    waveform = waveform.astype(np.float32)
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)  # stereo -> mono
    waveform = waveform / (np.abs(waveform).max() + 1e-8)  # normalize

    waveform = truncate_audio(waveform, CHUNK_LENGTH_SECONDS, sr)

    inputs = fe(
        waveform, sampling_rate=sr, return_tensors="np",
        padding="max_length", max_length=CHUNK_LENGTH_SECONDS * SAMPLE_RATE,
        truncation=True, do_normalize=True,
    )
    input_features = inputs.input_features.astype(np.float32)

    session = get_session(model_choice)
    input_name = session.get_inputs()[0].name
    logits = session.run(None, {input_name: input_features})[0]
    prob = float(logits.squeeze())

    label = "Complete (turn ended)" if prob > 0.5 else "Incomplete (still speaking)"
    return f"{label}\n\nConfidence: {prob:.3f}"


def predict_both(audio):
    if audio is None:
        return "No audio provided.", "No audio provided."
    r1 = predict(audio, "Stage 1 (pipecat reproduction)")
    r2 = predict(audio, "Stage 2 (fine-tuned on real Indic dialogue)")
    return r1, r2


with gr.Blocks(title="Turn Detection Demo") as demo:
    gr.Markdown("# Turn Detection: Stage 1 vs Stage 2")
    gr.Markdown(
        "Upload or record an 8-second (or shorter) audio clip. "
        "Compare the pipecat-reproduction model (trained on synthetic data) "
        "against the stage-2 model (fine-tuned on real Indic dialogue)."
    )

    with gr.Tab("Live Demo"):
        audio_input = gr.Audio(sources=["upload", "microphone"], type="numpy", label="Audio input")
        run_btn = gr.Button("Run both models", variant="primary")
        with gr.Row():
            out1 = gr.Textbox(label="Stage 1 (pipecat reproduction)")
            out2 = gr.Textbox(label="Stage 2 (fine-tuned)")
        run_btn.click(predict_both, inputs=audio_input, outputs=[out1, out2])

    with gr.Tab("Dataset Examples"):
        gr.Markdown("### Sample clips from both datasets")
        with open("samples/samples_manifest.json") as f:
            manifest = json.load(f)

        gr.Markdown("**Stage 1 (pipecat synthetic test set)**")
        for item in manifest["stage1"]:
            with gr.Row():
                gr.Audio(f"samples/stage1/{item['file']}", label=f"{item['language']} | label={item['label']}")

        gr.Markdown("**Stage 2 (real Indic dialogue)**")
        for item in manifest["stage2"]:
            with gr.Row():
                gr.Audio(f"samples/stage2/{item['file']}", label=f"{item['language']} | label={item['label']} | {item['kind']}")

    with gr.Tab("Report"):
        gr.Markdown("## Report — coming soon")
        gr.Markdown("Tables and charts covering EDA, training results, forgetting-vs-adaptation curve, and ONNX/quantization comparison will go here.")


if __name__ == "__main__":
    demo.launch()