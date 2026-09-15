import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import glob
import numpy as np
import torch
import torch.nn.functional as F
import gradio as gr
from transformers import AutoTokenizer
from src.models.student import EmotionStudent
from src.spectrum.valence_arousal import EmotionSpectrum
from src.spectrum.visualiser import plot_circumplex_trajectory

LABELS_27 = [
    "admiration","amusement","anger","annoyance","approval","caring","confusion",
    "curiosity","desire","disappointment","disapproval","disgust","embarrassment",
    "excitement","fear","gratitude","grief","joy","love","nervousness","optimism",
    "pride","realization","relief","remorse","sadness","surprise"
]

CKPT_DIR = "outputs/checkpoints"
candidates = sorted(
    glob.glob(os.path.join(CKPT_DIR, "student_ep*.pt")),
    key=lambda p: int("".join(c for c in os.path.basename(p) if c.isdigit()) or 0)
)
CKPT = candidates[-1] if candidates else None

print("=" * 55)
print("Using checkpoint: " + (CKPT if CKPT else "NONE"))
print("=" * 55)

tok = AutoTokenizer.from_pretrained("microsoft/deberta-v3-xsmall")
model = EmotionStudent(num_emotions=27)

if CKPT and os.path.exists(CKPT):
    model.load_state_dict(torch.load(CKPT, map_location="cpu"), strict=False)
    print("[OK] Loaded " + CKPT)
else:
    print("[WARN] No checkpoint - using random weights")

model.eval()

spectrum = EmotionSpectrum(labels=LABELS_27)
print("Spectrum: " + str(len(spectrum.labels)) + " labels | VA shape " + str(spectrum.va.shape))


def classify(text_block):
    lines = [l.strip() for l in text_block.split("\n") if l.strip()]
    if not lines:
        return "No input.", "", None

    probs_seq = []
    out = ""
    with torch.no_grad():
        for i, line in enumerate(lines):
            enc = tok(line, return_tensors="pt", truncation=True, max_length=128)
            lg, _ = model(enc["input_ids"], enc["attention_mask"])
            p = F.softmax(lg, dim=-1).numpy()[0].astype(np.float32)
            p = np.nan_to_num(p, nan=0.0)
            probs_seq.append(p)

            top = np.argsort(p)[::-1][:3]
            top_strs = [LABELS_27[j] + " (" + format(p[j], ".3f") + ")" for j in top]
            out += "**" + str(i + 1) + ".** " + line + "\n"
            out += "  -> " + ", ".join(top_strs) + "\n\n"

    traj = spectrum.compute_trajectory(probs_seq)
    fig = plot_circumplex_trajectory(traj, spectrum.labels)

    metrics = (
        "**Path length:** " + format(traj["total_path_length"], ".3f") + " | "
        "**Mean shift:** " + format(traj["mean_shift"], ".3f") + " | "
        "**Max shift:** " + format(traj["max_shift"], ".3f") + " | "
        "**Mean entropy:** " + format(float(traj["spectrum_entropy"].mean()), ".3f")
    )
    return out, metrics, fig


demo = gr.Interface(
    fn=classify,
    inputs=gr.Textbox(
        lines=6,
        label="Utterances (one per line)",
        value=(
            "I just got promoted at work!\n"
            "I am so happy and excited!\n"
            "But wait, I have so much more responsibility now.\n"
            "What if I fail?\n"
            "I feel so nervous."
        ),
    ),
    outputs=[
        gr.Markdown(label="Emotion Classification"),
        gr.Markdown(label="Spectrum Metrics"),
        gr.Plot(label="Emotion Spectrum Evolution"),
    ],
    title="Emotion Spectrum Evolution - Distilled Student Model",
    description="Track emotional trajectories using a lightweight student model.",
)

if __name__ == "__main__":
    demo.launch()
