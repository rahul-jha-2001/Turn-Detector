"""
generate_report.py

Data + polished Altair chart builders for the Streamlit "Report" tab.
All numbers are confirmed from actual training/eval runs in this
project - not estimates.

Colors are set explicitly for visibility against Streamlit's dark
theme (labelColor/titleColor/text color all set to light tones,
transparent chart background) - Altair's defaults render as
dark-on-dark and become invisible otherwise.
"""

import pandas as pd
import altair as alt

# ---- Shared theme ----
PRIMARY = "#3b82f6"     # blue - stage 1 / general
ACCENT = "#f87171"      # red - stage 2 / gap / risk
GOOD = "#4ade80"        # green - positive finding
WARN = "#fbbf24"        # amber - secondary/harder case
NEUTRAL_LIGHT = "#93c5fd"
LABEL_COLOR = "#f9fafb"  # near-white, for data labels
AXIS_COLOR = "#e5e7eb"   # light gray, for axis labels/titles

alt.themes.enable("none")


def _configure(chart):
    """Applied ONCE, only on the final top-level chart (never on
    sub-charts that get combined via + or hconcat/vconcat - Altair
    rejects configure_* calls on chart specs used inside a composition)."""
    return chart.configure_axis(
        labelFontSize=12, titleFontSize=13,
        labelColor=AXIS_COLOR, titleColor=AXIS_COLOR,
        domainColor="#6b7280", gridColor="#374151",
    ).configure_title(
        fontSize=15, anchor="start", color=LABEL_COLOR,
    ).configure_view(strokeWidth=0).configure(background="transparent")


def _base(chart, height=320):
    return _configure(chart.properties(height=height))


# ============================================================
# Confirmed numbers
# ============================================================

STAGE1_TEST_ACCURACY = 0.9336
STAGE1_TEST_F1 = 0.934

SYNTHETIC_PCT = 0.8243
REAL_PCT = 0.1757
REAL_LANGUAGES = {"eng": 43617, "spa": 3986}

STAGE1_LANG_ACCURACY = {
    "kor": 0.98, "tur": 0.97, "jpn": 0.97, "hin": 0.9283,
    "zho": 0.90, "spa": 0.90, "ara": 0.89, "mar": 0.87,
    "ben": 0.84, "vie": 0.82,
}

ZERO_SHOT_ACCURACY = 0.5885
ZERO_SHOT_F1 = 0.5138
ZERO_SHOT_PRECISION = 0.407
ZERO_SHOT_RECALL = 0.697
ZERO_SHOT_BY_KIND = {"mid_cut": 0.613, "cross_segment": 0.574}

DIARBENCH_EXTRACTION = {
    "total_true": 4620, "total_false_cross": 4730, "total_false_midcut": 5467,
    "hindi_total": 1297,
}

FORGETTING_CURVE = [
    {"epoch": 0, "stage2_accuracy": 0.5885, "stage1_accuracy": 0.9336},
    {"epoch": 1, "stage2_accuracy": 0.7075, "stage1_accuracy": 0.9203},
    {"epoch": 2, "stage2_accuracy": 0.7268, "stage1_accuracy": 0.9133},
    {"epoch": 3, "stage2_accuracy": 0.7223, "stage1_accuracy": 0.9142},
]

FINAL_2X2 = [
    {"model": "Stage 1 model", "dataset": "Synthetic test", "accuracy": 0.9336},
    {"model": "Stage 1 model", "dataset": "Real test", "accuracy": 0.5788},
    {"model": "Stage 2 model", "dataset": "Synthetic test", "accuracy": 0.9142},
    {"model": "Stage 2 model", "dataset": "Real test", "accuracy": 0.7223},
]

HINDI_IMPROVEMENT = {"stage1_model": 0.6429, "stage2_model": 0.8010}
KIND_BREAKDOWN_STAGE2_MODEL = {"mid_cut": 0.900, "cross_segment": 0.621}

QUANTIZATION = [
    {"model": "Stage 1", "format": "fp32", "size_mb": 30.58, "accuracy": 0.9360},
    {"model": "Stage 1", "format": "int8", "size_mb": 9.78, "accuracy": 0.9345},
    {"model": "Stage 2", "format": "fp32", "size_mb": 30.58, "accuracy": 0.7260},
    {"model": "Stage 2", "format": "int8", "size_mb": 9.78, "accuracy": 0.7245},
]


# ============================================================
# Chart builders
# ============================================================

def chart_synthetic_split():
    df = pd.DataFrame([
        {"category": "Synthetic (TTS)", "pct": SYNTHETIC_PCT},
        {"category": "Real (English + Spanish only)", "pct": REAL_PCT},
    ])
    bars = alt.Chart(df).mark_bar(size=60, cornerRadiusEnd=4).encode(
        x=alt.X("category:N", title=None, sort=None, axis=alt.Axis(labelAngle=0)),
        y=alt.Y("pct:Q", title="Share of training data", axis=alt.Axis(format="%")),
        color=alt.Color("category:N", legend=None, scale=alt.Scale(range=[ACCENT, PRIMARY])),
    )
    labels = alt.Chart(df).mark_text(dy=-10, fontSize=14, fontWeight="bold", color=LABEL_COLOR).encode(
        x=alt.X("category:N", sort=None),
        y="pct:Q",
        text=alt.Text("pct:Q", format=".1%"),
    )
    chart = (bars + labels).properties(title="Pipecat Training Data: 82% Synthetic, 0% Real Hindi")
    return _base(chart, height=300)


def chart_stage1_language_accuracy():
    df = pd.DataFrame([{"language": k, "accuracy": v} for k, v in STAGE1_LANG_ACCURACY.items()])
    df = df.sort_values("accuracy", ascending=False)
    bars = alt.Chart(df).mark_bar(cornerRadiusEnd=3).encode(
        x=alt.X("language:N", sort="-y", title="Language", axis=alt.Axis(labelAngle=0)),
        y=alt.Y("accuracy:Q", title="Test Accuracy", scale=alt.Scale(domain=[0.75, 1.0])),
        color=alt.condition(alt.datum.language == "hin", alt.value(PRIMARY), alt.value(NEUTRAL_LIGHT)),
        tooltip=["language", alt.Tooltip("accuracy:Q", format=".1%")],
    )
    labels = alt.Chart(df).mark_text(dy=-8, fontSize=11, color=LABEL_COLOR).encode(
        x=alt.X("language:N", sort="-y"), y="accuracy:Q",
        text=alt.Text("accuracy:Q", format=".0%"),
    )
    chart = (bars + labels).properties(title="Stage-1 Test Accuracy by Language (Hindi highlighted in blue)")
    return _base(chart, height=340)


def chart_zero_shot_gap():
    df = pd.DataFrame([
        {"test_set": "Synthetic test set", "accuracy": STAGE1_TEST_ACCURACY},
        {"test_set": "Real dialogue (zero-shot)", "accuracy": ZERO_SHOT_ACCURACY},
    ])
    bars = alt.Chart(df).mark_bar(size=70, cornerRadiusEnd=4).encode(
        x=alt.X("test_set:N", title=None, sort=None, axis=alt.Axis(labelAngle=0)),
        y=alt.Y("accuracy:Q", title="Accuracy", scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("test_set:N", legend=None, scale=alt.Scale(range=[PRIMARY, ACCENT])),
    )
    labels = alt.Chart(df).mark_text(dy=-10, fontSize=15, fontWeight="bold", color=LABEL_COLOR).encode(
        x=alt.X("test_set:N", sort=None), y="accuracy:Q",
        text=alt.Text("accuracy:Q", format=".1%"),
    )
    chart = (bars + labels).properties(
        title=f"The Generalization Gap: {STAGE1_TEST_ACCURACY-ZERO_SHOT_ACCURACY:.0%}-point drop on real audio"
    )
    return _base(chart, height=320)


def chart_forgetting_curve():
    df = pd.DataFrame(FORGETTING_CURVE)
    df_long = df.melt(id_vars="epoch", value_vars=["stage2_accuracy", "stage1_accuracy"],
                       var_name="series", value_name="accuracy")
    df_long["series"] = df_long["series"].map({
        "stage2_accuracy": "Stage-2 accuracy (real data - adaptation)",
        "stage1_accuracy": "Stage-1 accuracy (synthetic - forgetting check)",
    })
    lines = alt.Chart(df_long).mark_line(point=alt.OverlayMarkDef(size=80), strokeWidth=3).encode(
        x=alt.X("epoch:O", title="Epoch"),
        y=alt.Y("accuracy:Q", title="Accuracy", scale=alt.Scale(domain=[0.5, 1.0])),
        color=alt.Color("series:N", title=None, scale=alt.Scale(range=[ACCENT, PRIMARY])),
        tooltip=["epoch", "series", alt.Tooltip("accuracy:Q", format=".2%")],
    )
    labels = alt.Chart(df_long).mark_text(dy=-14, fontSize=11, color=LABEL_COLOR).encode(
        x="epoch:O", y="accuracy:Q", text=alt.Text("accuracy:Q", format=".1%"),
    )
    chart = (lines + labels).properties(title="Fine-tuning on Real Data: Adaptation Gain vs. Forgetting Cost")
    return _base(chart, height=360)


def chart_final_2x2():
    df = pd.DataFrame(FINAL_2X2)
    bars = alt.Chart(df).mark_bar(cornerRadiusEnd=3).encode(
        x=alt.X("dataset:N", title=None, axis=alt.Axis(labelAngle=0)),
        y=alt.Y("accuracy:Q", title="Accuracy", scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("model:N", title=None, scale=alt.Scale(range=[PRIMARY, ACCENT])),
        xOffset="model:N",
        tooltip=["model", "dataset", alt.Tooltip("accuracy:Q", format=".2%")],
    )
    labels = alt.Chart(df).mark_text(dy=-8, fontSize=11, color=LABEL_COLOR).encode(
        x=alt.X("dataset:N"), y="accuracy:Q", xOffset="model:N",
        text=alt.Text("accuracy:Q", format=".0%"),
    )
    chart = (bars + labels).properties(title="Final Comparison: Both Models x Both Test Sets")
    return _base(chart, height=360)


def chart_hindi_improvement():
    df = pd.DataFrame([
        {"model": "Before (Stage 1 model)", "accuracy": HINDI_IMPROVEMENT["stage1_model"]},
        {"model": "After (Stage 2 model)", "accuracy": HINDI_IMPROVEMENT["stage2_model"]},
    ])
    bars = alt.Chart(df).mark_bar(size=70, cornerRadiusEnd=4).encode(
        x=alt.X("model:N", title=None, sort=None, axis=alt.Axis(labelAngle=0)),
        y=alt.Y("accuracy:Q", title="Accuracy on Hindi (real dialogue)", scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("model:N", legend=None, scale=alt.Scale(range=[ACCENT, GOOD])),
    )
    labels = alt.Chart(df).mark_text(dy=-10, fontSize=15, fontWeight="bold", color=LABEL_COLOR).encode(
        x=alt.X("model:N", sort=None), y="accuracy:Q",
        text=alt.Text("accuracy:Q", format=".1%"),
    )
    chart = (bars + labels).properties(
        title=f"Hindi Accuracy: +{HINDI_IMPROVEMENT['stage2_model']-HINDI_IMPROVEMENT['stage1_model']:.0%} points after real-data fine-tuning"
    )
    return _base(chart, height=320)


def chart_kind_breakdown():
    df = pd.DataFrame([{"kind": k, "accuracy": v} for k, v in KIND_BREAKDOWN_STAGE2_MODEL.items()])
    df["kind"] = df["kind"].map({"mid_cut": "Mid-utterance cut (easier)",
                                  "cross_segment": "Cross-speaker change (harder)"})
    bars = alt.Chart(df).mark_bar(size=70, cornerRadiusEnd=4).encode(
        x=alt.X("kind:N", title=None, sort=None, axis=alt.Axis(labelAngle=0)),
        y=alt.Y("accuracy:Q", title="Accuracy", scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("kind:N", legend=None, scale=alt.Scale(range=[GOOD, WARN])),
    )
    labels = alt.Chart(df).mark_text(dy=-10, fontSize=15, fontWeight="bold", color=LABEL_COLOR).encode(
        x=alt.X("kind:N", sort=None), y="accuracy:Q",
        text=alt.Text("accuracy:Q", format=".1%"),
    )
    chart = (bars + labels).properties(title="Where the Gains Come From: Easy vs. Hard Cases")
    return _base(chart, height=320)


def chart_quantization():
    df = pd.DataFrame(QUANTIZATION)
    df["label"] = df["model"] + " " + df["format"]

    size_chart = alt.Chart(df).mark_bar(cornerRadiusEnd=3).encode(
        x=alt.X("label:N", title=None, sort=None, axis=alt.Axis(labelAngle=-20)),
        y=alt.Y("size_mb:Q", title="Size (MB)"),
        color=alt.Color("format:N", title="Format", scale=alt.Scale(range=[NEUTRAL_LIGHT, PRIMARY])),
        tooltip=["label", "size_mb"],
    )
    size_labels = alt.Chart(df).mark_text(dy=-8, fontSize=11, color=LABEL_COLOR).encode(
        x=alt.X("label:N", sort=None), y="size_mb:Q",
        text=alt.Text("size_mb:Q", format=".1f"),
    )
    size_final = (size_chart + size_labels).properties(height=300, title="Model Size", width=280)

    acc_chart = alt.Chart(df).mark_bar(cornerRadiusEnd=3).encode(
        x=alt.X("label:N", title=None, sort=None, axis=alt.Axis(labelAngle=-20)),
        y=alt.Y("accuracy:Q", title="Accuracy (own test set)", scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("format:N", title="Format", scale=alt.Scale(range=[NEUTRAL_LIGHT, PRIMARY])),
        tooltip=["label", alt.Tooltip("accuracy:Q", format=".2%")],
    )
    acc_labels = alt.Chart(df).mark_text(dy=-8, fontSize=11, color=LABEL_COLOR).encode(
        x=alt.X("label:N", sort=None), y="accuracy:Q",
        text=alt.Text("accuracy:Q", format=".1%"),
    )
    acc_final = (acc_chart + acc_labels).properties(height=300, title="Accuracy", width=280)

    combined = alt.hconcat(size_final, acc_final).resolve_scale(color="independent")
    return _configure(combined)