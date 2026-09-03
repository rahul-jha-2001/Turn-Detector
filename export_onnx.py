"""
export_onnx.py

Exports a SmartTurnV3Model checkpoint to ONNX, then applies dynamic int8
quantization - matching pipecat's own approach (confirmed: ONNX export +
dynamic quantization, NOT static - community results show static loses
4-12+ accuracy points on this architecture, dynamic is lossless).

Produces two files per checkpoint:
  <name>_fp32.onnx       - unquantized ONNX export
  <name>_int8.onnx       - dynamically quantized version
"""

import os
import torch
from transformers import WhisperConfig
from safetensors.torch import load_file
from onnxruntime.quantization import quantize_dynamic, QuantType

from model import SmartTurnV3Model, MAX_SOURCE_POSITIONS


class ONNXWrapper(torch.nn.Module):
    """torch.onnx.export needs a model whose forward() returns a plain
    tensor, not a dict - SmartTurnV3Model.forward() returns
    {"logits": ...} (or {"loss":..., "logits":...} with labels). This
    wrapper strips that down to just the tensor ONNX needs."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_features):
        return self.model(input_features)["logits"]


def load_finetuned_model(checkpoint_dir: str) -> SmartTurnV3Model:
    config = WhisperConfig.from_pretrained(checkpoint_dir)
    model = SmartTurnV3Model(config)
    safetensors_path = os.path.join(checkpoint_dir, "model.safetensors")
    bin_path = os.path.join(checkpoint_dir, "pytorch_model.bin")
    if os.path.exists(safetensors_path):
        state_dict = load_file(safetensors_path)
    elif os.path.exists(bin_path):
        state_dict = torch.load(bin_path, map_location="cpu")
    else:
        raise FileNotFoundError(f"No weights file in {checkpoint_dir}")
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def export_to_onnx(model: torch.nn.Module, out_path: str):
    """Exports with dynamic batch size, fixed input shape otherwise -
    matches model.py's max_source_positions=400 -> input_features
    shape [B, 80, 800]."""
    dummy_input = torch.randn(1, 80, MAX_SOURCE_POSITIONS * 2)  # [1, 80, 800]

    torch.onnx.export(
    model,
    (dummy_input,),
    out_path,
    input_names=["input_features"],
    output_names=["logits"],
    dynamic_axes={
        "input_features": {0: "batch_size"},
        "logits": {0: "batch_size"},
    },
    opset_version=17,
    do_constant_folding=True,
    dynamo=False,  # force legacy TorchScript-based exporter - more mature/reliable for custom models
)
    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"Exported {out_path} ({size_mb:.2f} MB)")


def quantize_onnx(fp32_path: str, int8_path: str):
    quantize_dynamic(
        model_input=fp32_path,
        model_output=int8_path,
        weight_type=QuantType.QInt8,
    )
    size_mb = os.path.getsize(int8_path) / (1024 * 1024)
    print(f"Quantized {int8_path} ({size_mb:.2f} MB)")


def process_checkpoint(name: str, checkpoint_dir: str, out_dir: str = "onnx_models"):
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n=== Processing {name} ({checkpoint_dir}) ===")

    model = load_finetuned_model(checkpoint_dir)
    model = ONNXWrapper(model)
    model.eval()

    fp32_path = os.path.join(out_dir, f"{name}_fp32.onnx")
    int8_path = os.path.join(out_dir, f"{name}_int8.onnx")

    export_to_onnx(model, fp32_path)
    quantize_onnx(fp32_path, int8_path)

    return fp32_path, int8_path


def main():
    checkpoints = {
        "stage1_final": "checkpoints/final_model",
        "stage2_final": "checkpoints_stage2/final_model",
    }

    for name, path in checkpoints.items():
        process_checkpoint(name, path)

    print("\nDone. ONNX models saved in onnx_models/")


if __name__ == "__main__":
    main()