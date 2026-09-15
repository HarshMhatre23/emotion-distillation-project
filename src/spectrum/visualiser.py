import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.data.emotion_mapper import EMOTION_VA_MAP
def plot_circumplex_trajectory(traj, labels):
    fig, ax = plt.subplots(figsize=(8,8))
    for e,(v,a) in EMOTION_VA_MAP.items():
        ax.scatter(v, a, s=20, alpha=0.4, color="gray")
        ax.annotate(e, (v,a), fontsize=7, alpha=0.6)
    path = traj["va_path"]
    ax.plot(path[:,0], path[:,1], "-o", color="red", lw=2, label="Trajectory")
    for i,(x,y) in enumerate(path): ax.annotate(str(i+1), (x,y), fontsize=9, color="darkred")
    ax.set_xlim(-1.1,1.1); ax.set_ylim(-1.1,1.1)
    ax.axhline(0, color="k", lw=0.5); ax.axvline(0, color="k", lw=0.5)
    ax.set_xlabel("Valence"); ax.set_ylabel("Arousal")
    ax.set_title("Emotion Spectrum Evolution (Circumplex)"); ax.legend()
    plt.tight_layout(); return fig
