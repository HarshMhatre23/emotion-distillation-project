"""Emotion Spectrum Evolution - Valence-Arousal trajectory framework."""
import numpy as np
from src.data.emotion_mapper import EMOTION_VA_MAP

class EmotionSpectrum:
    """Maps 27-dimensional emotion probabilities to Valence-Arousal coordinates
    and tracks emotional trajectory across a sequence of utterances."""

    def __init__(self, labels=None):
        if labels is None:
            self.labels = list(EMOTION_VA_MAP.keys())
        else:
            self.labels = list(labels)
        self.va = np.array([EMOTION_VA_MAP[e] for e in self.labels], dtype=np.float32)

    def to_va_coordinates(self, probs):
        probs = np.asarray(probs, dtype=np.float32)
        if probs.ndim == 1:
            probs = probs[np.newaxis, :]
        n = min(probs.shape[-1], self.va.shape[0])
        probs = probs[..., :n]
        va = self.va[:n]
        p = probs / (probs.sum(axis=-1, keepdims=True) + 1e-8)
        return p @ va

    def compute_trajectory(self, seq_probs):
        T = len(seq_probs)
        va_path = np.array([self.to_va_coordinates(p)[0] for p in seq_probs])
        shifts = np.linalg.norm(np.diff(va_path, axis=0), axis=1) if T > 1 else np.array([0.0])
        dominant = [self.labels[int(np.argmax(p[:len(self.labels)]))] for p in seq_probs]
        entropy = []
        for p in seq_probs:
            pn = p / (p.sum() + 1e-8)
            entropy.append(float(-np.sum(pn * np.log(pn + 1e-8))))
        return {
            "va_path": va_path,
            "shift_magnitudes": shifts,
            "drift_velocity": shifts,
            "dominant_emotions": dominant,
            "spectrum_entropy": np.array(entropy),
            "total_path_length": float(shifts.sum()),
            "mean_shift": float(shifts.mean()),
            "max_shift": float(shifts.max()),
        }
