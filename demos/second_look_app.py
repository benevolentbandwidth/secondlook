"""Second Look — demo UI (plumbing prototype).

Single-screen Gradio app that demonstrates the end-to-end on-device flow:
    image -> preprocessing -> binary model -> confidence -> concern tier.

IMPORTANT: the bundled checkpoint is a 1-epoch smoke model. Predictions are
PLACEHOLDERS and carry no clinical meaning. The banner in the UI says so; do
not remove it. This app exists to show the pipeline + tier UX, not performance.

Run:
    python demos/second_look_app.py
then open the printed http://127.0.0.1:7860 URL.
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np
import pandas as pd
import gradio as gr
import tensorflow as tf

from data_pipeline.preprocessor import preprocess, load_image
from data_pipeline.label_mapper import confidence_to_tier, display_label

CKPT = REPO / "modeling" / "checkpoints" / "smoke" / "best.keras"
MANIFEST = REPO / "data" / "manifest.csv"

TIER_COLORS = {"Low": "#2e7d32", "Moderate": "#f9a825", "Elevated": "#c62828"}

_model = tf.keras.models.load_model(str(CKPT)) if CKPT.exists() else None


def _sample_choices() -> dict[str, str]:
    """Map a human label -> image path for cached manifest rows."""
    if not MANIFEST.exists():
        return {}
    m = pd.read_csv(MANIFEST)
    has = ~(m["image_local_path"].isna()
            | (m["image_local_path"].astype(str).str.strip() == ""))
    m = m[has]
    out = {}
    for _, r in m.head(20).iterrows():
        truth = "WORTH" if int(r["canonical_label"]) == 1 else "NOT WORTH"
        out[f"{r['case_folder']}  (truth: {truth})"] = r["image_local_path"]
    return out


SAMPLES = _sample_choices()


def _tier_html(prob: float) -> str:
    tier = confidence_to_tier(prob)
    color = TIER_COLORS[tier]
    return (
        f"<div style='text-align:center;padding:18px;border-radius:12px;"
        f"background:{color};color:white;font-family:sans-serif;'>"
        f"<div style='font-size:14px;opacity:0.9;'>Concern tier</div>"
        f"<div style='font-size:30px;font-weight:700;margin-top:4px;'>"
        f"{display_label(tier)}</div>"
        f"<div style='font-size:13px;opacity:0.85;margin-top:8px;'>"
        f"model confidence (placeholder): {prob:.2f}</div></div>"
    )


def analyze(sample_key: str, uploaded: np.ndarray | None):
    if uploaded is not None:
        raw = uploaded
    elif sample_key and sample_key in SAMPLES:
        raw = load_image(SAMPLES[sample_key])
    else:
        return None, "<div style='padding:18px;'>Pick a sample or upload an image.</div>"

    proc = preprocess(raw)                     # (224, 224, 1) float32 [0,1]
    disp = (proc[:, :, 0] * 255).astype(np.uint8)

    if _model is None:
        return disp, "<div style='padding:18px;color:#c62828;'>No checkpoint found.</div>"

    prob = float(_model.predict(proc[None, ...], verbose=0).ravel()[0])
    return disp, _tier_html(prob)


BANNER = (
    "## 🔍 Second Look — pipeline prototype\n"
    "**⚠️ Placeholder model — NOT yet trained.** The checkpoint behind this app "
    "is a 1-epoch smoke model; tiers shown are *meaningless* and for plumbing "
    "demonstration only. This screen shows the **preprocessing → binary model → "
    "concern-tier UX**, not real performance. Nothing is uploaded or stored — "
    "the target is fully on-device."
)

with gr.Blocks(title="Second Look (prototype)") as demo:
    gr.Markdown(BANNER)
    with gr.Row():
        with gr.Column():
            sample = gr.Dropdown(
                choices=list(SAMPLES.keys()),
                label="Sample mammogram (cached CBIS-DDSM)",
                value=(list(SAMPLES.keys())[0] if SAMPLES else None),
            )
            upload = gr.Image(label="…or upload your own", type="numpy", image_mode="L")
            run = gr.Button("Run Second Look", variant="primary")
        with gr.Column():
            out_img = gr.Image(label="Preprocessed model input (224×224)")
            out_tier = gr.HTML()
    run.click(analyze, inputs=[sample, upload], outputs=[out_img, out_tier])

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, inbrowser=False)
