<div align="center">

# 🧠 EmotionDistill

### Knowledge Distillation of Large Language Models for Lightweight Emotion Classification with Emotion Spectrum Evolution

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-EE4C2C.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/🤗%20Transformers-4.40+-yellow.svg)](https://huggingface.co/transformers/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**Compress a 500M-parameter teacher LLM into a 10K-parameter deployable student — 50,000× smaller, sub-5ms inference on CPU — with a novel Valence-Arousal trajectory tracker for emotional evolution.**

[Key Features](#-key-features) •
[Architecture](#-architecture) •
[Quickstart](#-quickstart) •
[Results](#-results) •
[Demo](#-interactive-demo) •
[Citation](#-citation)

</div>

---

## 📖 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Architecture](#-architecture)
- [Emotion Spectrum Evolution](#-emotion-spectrum-evolution)
- [Installation](#-installation)
- [Quickstart](#-quickstart)
- [Project Structure](#-project-structure)
- [Results](#-results)
- [Interactive Demo](#-interactive-demo)
- [Configuration](#-configuration)
- [Roadmap](#-roadmap)
- [Citation](#-citation)
- [License](#-license)

---

## 🌟 Overview

**EmotionDistill** is a research-grade pipeline that distills the emotion-understanding capability of Large Language Models (LLMs) into lightweight, deployable student models. It targets real-world applications where **latency, memory, and offline operation** matter — chatbots, mobile apps, edge devices, and real-time sentiment monitoring.

Unlike traditional emotion classifiers that output a single discrete label, EmotionDistill introduces the **Emotion Spectrum Evolution** framework: a novel module that maps 27 fine-grained emotion predictions into continuous **Valence-Arousal (circumplex) space** and tracks how emotions *evolve over a conversation*.

### 🎯 The Problem

| Challenge | Impact |
|-----------|--------|
| LLMs are huge (500M – 70B params) | Cannot deploy on mobile / edge |
| High latency (200–500ms per query) | Unsuitable for real-time chat |
| Requires GPU + internet | Fails in offline/privacy-critical settings |
| Single-utterance classification | Misses emotional arc of conversations |

### 💡 Our Solution

- **Knowledge Distillation** from a Qwen2.5 teacher into a DeBERTa-v3 student
- **Dual-head architecture**: categorical emotion classifier + dimensional (V-A) regressor
- **Emotion Spectrum Evolution**: continuous VA trajectory tracking
- **Production-ready**: INT8 ONNX export, sub-5ms CPU inference, 100% offline

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| 🔬 **Multi-Strategy Distillation** | Soft-label KD, Hidden-state alignment, Contrastive KD, Optimal Transport KD, Composite loss |
| 🎭 **27-Emotion Classification** | Trained on GoEmotions — the largest fine-grained emotion dataset |
| 📈 **Emotion Spectrum Evolution** | Novel VA-trajectory framework based on Russell's Circumplex Model |
| 🎯 **Dual-Head Architecture** | Classification head (27 emotions) + regression head (Valence, Arousal) |
| ⚡ **Sub-5ms Inference** | 100-200× faster than the teacher LLM on CPU |
| 📦 **ONNX + INT8 Quantization** | Deployable on mobile, browser (ONNX.js), and edge devices |
| 🌐 **100% Offline** | No API keys, no internet, no cloud dependency |
| 📊 **Pareto Trade-off Analysis** | Automated accuracy-vs-efficiency reports |
| 🖥️ **Interactive Gradio Demo** | Live emotion trajectory visualization |
| 🧪 **Fully Tested** | pytest suite covering data, models, losses, and spectrum |

---

## 🏗️ Architecture
