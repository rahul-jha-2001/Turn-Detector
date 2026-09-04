"""
app.py

Streamlit demo: live inference with stage-1/stage-2 ONNX models (CPU
only, via ONNX Runtime), sample clips from both datasets with
correctness checking, and a placeholder Report tab.
"""

import json
import os
import numpy as np
import onnxruntime as ort
import soundfile as sf
import streamlit as st
from transformers import WhisperFeatureExtractor

CHUNK_LENGTH_SECONDS = 8
SAMPLE_RATE = 16000

MODELS = {
    "Stage 1 (pipecat reproduction)": "onnx_models/stage1_final_int8.onnx",
    "Stage 2 (fine-tuned on real Indic dialogue)": "onnx_models/stage2_final_int8.onnx",
}


@st.cache_resource
def get_feature_extractor():
    return WhisperFeatureExtractor(chunk_length=CHUNK_LENGTH_SECONDS)


@st.cache_resource
def get_session(model_key):
    return ort.InferenceSession(MODELS[model_key], providers=["CPUExecutionProvider"])


def truncate_audio(audio_array, n_seconds=8, sr=16000):
    max_samples = int(n_seconds * sr)
    if len(audio_array) > max_samples:
        return audio_array[-max_samples:]
    return audio_array


def predict(waveform, sr, model_key, fe):
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

    session = get_session(model_key)
    input_name = session.get_inputs()[0].name
    logits = session.run(None, {input_name: input_features})[0]
    prob_complete = float(logits.squeeze())

    predicted_complete = prob_complete > 0.5
    label = "Complete (turn ended)" if predicted_complete else "Incomplete (still speaking)"
    confidence = prob_complete if predicted_complete else 1 - prob_complete
    return label, confidence, predicted_complete


@st.cache_data
def load_manifest():
    if os.path.exists("samples/samples_manifest.json"):
        with open("samples/samples_manifest.json") as f:
            return json.load(f)
    return None


st.set_page_config(page_title="Turn Detection: Stage 1 vs Stage 2", layout="wide")
st.title("Turn Detection: Stage 1 vs Stage 2")
st.write(
    "Upload an 8-second (or shorter) audio clip, or pick a sample below. "
    "Compare the pipecat-reproduction model (trained on synthetic data) "
    "against the stage-2 model (fine-tuned on real Indic dialogue)."
)

tab1, tab2, tab3 = st.tabs(["Live Demo", "Dataset Examples", "Report"])

fe = get_feature_extractor()
manifest = load_manifest()

with tab1:
    st.subheader("Try it out")

    sample_options = ["-- none --"]
    sample_lookup = {}
    if manifest:
        for item in manifest.get("stage1", []):
            key = f"[Stage 1] {item['language']} | {'Complete' if item['label'] else 'Incomplete'} | {item['file']}"
            sample_options.append(key)
            sample_lookup[key] = (f"samples/stage1/{item['file']}", item["label"])
        for item in manifest.get("stage2", []):
            key = f"[Stage 2] {item['language']} | {'Complete' if item['label'] else 'Incomplete'} | {item['file']}"
            sample_options.append(key)
            sample_lookup[key] = (f"samples/stage2/{item['file']}", item["label"])

    selected_sample = st.selectbox("Or pick a sample clip", sample_options)
    uploaded_file = st.file_uploader("Or upload your own audio", type=["wav", "mp3", "flac"])

    waveform, sr, true_label = None, None, None

    if uploaded_file is not None:
        waveform, sr = sf.read(uploaded_file)
    elif selected_sample != "-- none --":
        path, true_label = sample_lookup[selected_sample]
        waveform, sr = sf.read(path)
        st.audio(path)

    if waveform is not None and st.button("Run both models", type="primary"):
        col1, col2 = st.columns(2)
        for col, model_key in zip([col1, col2], MODELS.keys()):
            with col:
                st.markdown(f"**{model_key}**")
                label, confidence, predicted_complete = predict(waveform, sr, model_key, fe)
                st.write(label)
                st.write(f"Confidence: {confidence:.3f}")
                if true_label is not None:
                    true_complete = bool(true_label)
                    correct = predicted_complete == true_complete
                    st.write(f"{'✅ Correct' if correct else '❌ Wrong'} "
                              f"(true label: {'Complete' if true_complete else 'Incomplete'})")

with tab2:
    st.subheader("Sample clips from both datasets")
    if manifest:
        st.markdown("**Stage 1 (pipecat synthetic test set)**")
        for item in manifest["stage1"]:
            st.write(f"{item['language']} | {'Complete' if item['label'] else 'Incomplete'}")
            st.audio(f"samples/stage1/{item['file']}")

        st.markdown("**Stage 2 (real Indic dialogue)**")
        for item in manifest["stage2"]:
            st.write(f"{item['language']} | {'Complete' if item['label'] else 'Incomplete'}")
            st.audio(f"samples/stage2/{item['file']}")
    else:
        st.write("Sample files not found.")

with tab3:
    st.subheader("Report — coming soon")
    st.write("Tables and charts covering EDA, training results, forgetting-vs-adaptation "
             "curve, and ONNX/quantization comparison will go here.")