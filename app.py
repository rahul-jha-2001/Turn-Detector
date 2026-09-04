"""
app.py

Gradio demo: live inference with stage-1/stage-2 ONNX models (CPU only,
via ONNX Runtime), sample clips from both datasets with correctness
checking, and a placeholder Report tab.
"""

import json
import os
import numpy as np
import onnxruntime as ort
import gradio as gr
import soundfile as sf
from transformers import WhisperFeatureExtractor

CHUNK_LENGTH_SECONDS = 8
SAMPLE_RATE = 16000

fe = WhisperFeatureExtractor(chunk_length=CHUNK_LENGTH_SECONDS)

MODELS = {
    "Stage 1 (pipecat reproduction)": "onnx_models/stage1_final_int8.onnx",
    "Stage 2 (fine-tuned on real Indic dialogue)": "onnx_models/stage2_final_int8.onnx",
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

@spaces.GPU
def predict(audio, model_choice, true_label=None):
    if audio is None:
        return "No audio provided."

    sr, waveform = audio
    waveform = waveform.astype(np.float32)
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)
    waveform = waveform / (np.abs(waveform).max() + 1e-8)
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
    prob_complete = float(logits.squeeze())

    predicted_complete = prob_complete > 0.5
    label = "Complete (turn ended)" if predicted_complete else "Incomplete (still speaking)"
    confidence = prob_complete if predicted_complete else 1 - prob_complete

    result = f"{label}\n\nConfidence: {confidence:.3f}"

    if true_label is not None:
        true_complete = str(true_label) == "True"
        correct = predicted_complete == true_complete
        result += f"\n\n{'✅ Correct' if correct else '❌ Wrong'} (true label: {'Complete' if true_complete else 'Incomplete'})"

    return result


def predict_both(audio, true_label):
    if audio is None:
        return "No audio provided.", "No audio provided."
    r1 = predict(audio, "Stage 1 (pipecat reproduction)", true_label)
    r2 = predict(audio, "Stage 2 (fine-tuned on real Indic dialogue)", true_label)
    return r1, r2


def load_sample_choices():
    choices = []
    if os.path.exists("samples/samples_manifest.json"):
        with open("samples/samples_manifest.json") as f:
            manifest = json.load(f)
        for item in manifest.get("stage1", []):
            display = f"{item['language']} | {'Complete' if item['label'] else 'Incomplete'} | Stage 1"
            choices.append((display, f"samples/stage1/{item['file']}|{item['label']}"))
        for item in manifest.get("stage2", []):
            display = f"{item['language']} | {'Complete' if item['label'] else 'Incomplete'} | Stage 2"
            choices.append((display, f"samples/stage2/{item['file']}|{item['label']}"))
    return choices


def load_sample_audio(sample_value):
    if not sample_value:
        return None, None
    path, label = sample_value.rsplit("|", 1)
    waveform, sr = sf.read(path)
    return (sr, waveform), label


with gr.Blocks(title="Turn Detection Demo") as demo:
    gr.Markdown("# Turn Detection: Stage 1 vs Stage 2")
    gr.Markdown(
        "Upload or record an 8-second (or shorter) audio clip, or pick a sample below. "
        "Compare the pipecat-reproduction model (trained on synthetic data) "
        "against the stage-2 model (fine-tuned on real Indic dialogue)."
    )

    with gr.Tab("Live Demo"):
        sample_choices = load_sample_choices()
        sample_dropdown = gr.Dropdown(choices=sample_choices, label="Or pick a sample clip", value=None)
        audio_input = gr.Audio(sources=["upload", "microphone"], type="numpy", label="Audio input")
        true_label_state = gr.State(value=None)
        run_btn = gr.Button("Run both models", variant="primary")
        with gr.Row():
            out1 = gr.Textbox(label="Stage 1 (pipecat reproduction)")
            out2 = gr.Textbox(label="Stage 2 (fine-tuned)")

        sample_dropdown.change(load_sample_audio, inputs=sample_dropdown, outputs=[audio_input, true_label_state])
        run_btn.click(predict_both, inputs=[audio_input, true_label_state], outputs=[out1, out2])

    with gr.Tab("Dataset Examples"):
        gr.Markdown("### Sample clips from both datasets")
        manifest = None
        try:
            with open("samples/samples_manifest.json") as f:
                manifest = json.load(f)
        except FileNotFoundError:
            gr.Markdown("Sample files not found.")

        if manifest:
            gr.Markdown("**Stage 1 (pipecat synthetic test set)**")
            for item in manifest["stage1"]:
                with gr.Row():
                    gr.Audio(f"samples/stage1/{item['file']}",
                              label=f"{item['language']} | {'Complete' if item['label'] else 'Incomplete'}")

            gr.Markdown("**Stage 2 (real Indic dialogue)**")
            for item in manifest["stage2"]:
                with gr.Row():
                    gr.Audio(f"samples/stage2/{item['file']}",
                              label=f"{item['language']} | {'Complete' if item['label'] else 'Incomplete'}")

    with gr.Tab("Report"):
        gr.Markdown("## Report — coming soon")
        gr.Markdown("Tables and charts covering EDA, training results, forgetting-vs-adaptation curve, and ONNX/quantization comparison will go here.")


if __name__ == "__main__":
    demo.launch()