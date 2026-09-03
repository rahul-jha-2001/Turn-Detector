"""
model.py

Rewritten to match pipecat's actual SmartTurnV3Model architecture from
their train.py, rather than our earlier from-scratch design:

  - max_source_positions=400 (8 seconds of audio, not the default 30s/1500)
  - attention pooling: Linear(hidden,256) -> Tanh -> Linear(256,1) -> softmax
  - classifier: Linear(hidden,256) -> LayerNorm -> GELU -> Dropout ->
                Linear(256,64) -> GELU -> Linear(64,1)
  - loaded via WhisperPreTrainedModel.from_pretrained(..., ignore_mismatched_sizes=True)
    so the positional embedding (sized for 1500 by default) is randomly
    reinitialized at 400 rather than erroring on shape mismatch - this
    matches what pipecat's own code does.

Freeze/unfreeze controls kept from our version, since pipecat's own script
doesn't do progressive unfreezing (they just fine-tune everything at
learning_rate=5e-5 from the start) - we're keeping our safer schedule as
a deliberate deviation, not a mistake, given our smaller fine-tuning data.
"""

import torch
import torch.nn as nn
from torch.nn.functional import softmax
from transformers import WhisperPreTrainedModel, WhisperConfig, WhisperModel
from transformers.models.whisper.modeling_whisper import WhisperEncoder

BASE_MODEL_NAME = "openai/whisper-tiny"
MAX_SOURCE_POSITIONS = 400  # 8 seconds - matches pipecat's WhisperFeatureExtractor(chunk_length=8)


class SmartTurnV3Model(WhisperPreTrainedModel):
    """Matches pipecat's own model class - encoder-only Whisper + attention
    pooling + MLP classifier."""

    _tied_weights_keys = []  # no decoder -> nothing to tie

    def tie_weights(self):
        # No real tying needed (no decoder/lm_head), but some transformers
        # versions expect `all_tied_weights_keys` to exist as a side effect
        # of this method running - set it explicitly rather than skipping
        # the method entirely, which left the attribute missing and crashed
        # a later finalization step.
        self.all_tied_weights_keys = {}

    def __init__(self, config: WhisperConfig):
        super().__init__(config)
        config.max_source_positions = MAX_SOURCE_POSITIONS
        self.encoder = WhisperEncoder(config)
        hidden_size = config.d_model

        self.pool_attention = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.Tanh(),
            nn.Linear(256, 1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )

        for module in self.classifier:
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=0.1)
                if module.bias is not None:
                    module.bias.data.zero_()
        for module in self.pool_attention:
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=0.1)
                if module.bias is not None:
                    module.bias.data.zero_()

        # HF's WhisperEncoder freezes embed_positions by default (it's meant
        # to hold fixed sinusoidal weights loaded from the pretrained
        # checkpoint). Since we resized max_source_positions to 400, the
        # pretrained [1500,384] table no longer fits and gets discarded -
        # embed_positions is now randomly initialized, but HF's freeze still
        # applies, leaving it stuck at random noise forever unless we
        # explicitly re-enable it here.
        self.encoder.embed_positions.requires_grad_(True)

    def forward(self, input_features, labels=None, **kwargs):
        # **kwargs absorbs extra collator output (e.g. 'meta') that Trainer
        # passes through as **inputs but forward() doesn't need directly.
        encoder_outputs = self.encoder(input_features=input_features)
        hidden_states = encoder_outputs.last_hidden_state  # [B, seq, hidden]

        attention_weights = self.pool_attention(hidden_states)      # [B, seq, 1]
        attention_weights = softmax(attention_weights, dim=1)
        pooled = torch.sum(hidden_states * attention_weights, dim=1)  # [B, hidden]

        logits = self.classifier(pooled)  # [B, 1]

        if labels is not None:
            pos_weight = ((labels == 0).sum() / (labels == 1).sum()).clamp(min=0.1, max=10.0)
            loss_fct = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            labels = labels.float()
            loss = loss_fct(logits.view(-1), labels.view(-1))
            return {"loss": loss, "logits": torch.sigmoid(logits.detach())}

        return {"logits": torch.sigmoid(logits)}

    # --- freeze/unfreeze controls, kept from our earlier design ---
    # (pipecat's own script doesn't do this - they fine-tune the whole
    # encoder from step 0. We keep progressive unfreezing as a deliberate,
    # safer choice given our smaller stage-2 fine-tuning data later.)

    # --- freeze/unfreeze controls removed ---
    # Matching pipecat's own train.py: full encoder fine-tuning from step 0,
    # no freezing. Simpler, and removes an entire class of bugs (see the
    # inverted-condition issue above) that progressive freezing introduced
    # without a corresponding benefit once we're following their exact
    # training recipe (4 epochs, lr=5e-5, full dataset).

    def trainable_param_groups(self, head_lr: float, encoder_lr: float):
        encoder_params = list(self.encoder.parameters())
        fast_params = (
            list(self.classifier.parameters())
            + list(self.pool_attention.parameters())
        )
        # embed_positions is part of encoder_params here since nothing is
        # frozen anymore - it just trains at encoder_lr along with everything
        # else, same as pipecat does.
        return [
            {"params": fast_params, "lr": head_lr},
            {"params": encoder_params, "lr": encoder_lr},
        ]


def load_model(sliced_pos_embed: bool = False) -> SmartTurnV3Model:
    """Default matches pipecat's own train.py exactly: from_pretrained with
    ignore_mismatched_sizes=True, which leaves embed_positions randomly
    reinitialized (shape mismatch: pretrained is [1500,384], ours is
    [400,384]). Full encoder fine-tuning from step 0, no freezing - matches
    their approach exactly.

    sliced_pos_embed=True is our own deviation/ablation, kept separately -
    reuses the first 400 rows of the pretrained [1500,384] table instead of
    discarding it. This is NOT what pipecat does - only use for the explicit
    ablation comparison if you want to test it later.
    """
    if sliced_pos_embed:
        config = WhisperConfig.from_pretrained(BASE_MODEL_NAME)
        model = SmartTurnV3Model(config)

        pretrained = WhisperModel.from_pretrained(BASE_MODEL_NAME)
        encoder_state_dict = pretrained.encoder.state_dict()
        pretrained_pos = encoder_state_dict["embed_positions.weight"]  # [1500, 384]
        encoder_state_dict["embed_positions.weight"] = pretrained_pos[:MAX_SOURCE_POSITIONS].clone()

        missing, unexpected = model.encoder.load_state_dict(encoder_state_dict, strict=False)
        print(f"Encoder load (sliced pos embed) - missing: {missing}, unexpected: {unexpected}")
    else:
        config = WhisperConfig.from_pretrained(BASE_MODEL_NAME)
        model = SmartTurnV3Model(config)

        pretrained = WhisperModel.from_pretrained(BASE_MODEL_NAME)
        encoder_state_dict = pretrained.encoder.state_dict()
        encoder_state_dict.pop("embed_positions.weight", None)

        missing, unexpected = model.encoder.load_state_dict(encoder_state_dict, strict=False)
        print(f"Encoder load - missing: {missing}, unexpected: {unexpected}")

    return model


if __name__ == "__main__":
    model = load_model()

    dummy_input = torch.randn(4, 80, MAX_SOURCE_POSITIONS * 2)  # [B, 80, 800]
    out = model(dummy_input)
    print("Output logits shape:", out["logits"].shape)  # expect [4, 1]

    dummy_labels = torch.tensor([1, 0, 1, 0])
    out = model(dummy_input, labels=dummy_labels)
    print("Loss:", out["loss"].item())

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {n_trainable:,} / {n_total:,} (should be equal - no freezing)")