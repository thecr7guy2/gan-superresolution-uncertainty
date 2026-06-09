# Uncertainty Estimation in GANs for Super-Resolution

> **Master's Thesis** — Maniraj Sai Adapa | University of Groningen

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/Framework-PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![Deep Learning](https://img.shields.io/badge/Domain-Deep%20Learning-blueviolet)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

---

## Overview

This repository contains the full implementation and supplementary materials for the Master's Thesis:

**"Estimating Uncertainty in Generative Adversarial Neural Networks for Super-Resolution"**

The core idea: standard GANs reconstruct high-resolution images but give no signal about *how confident* they are. This work integrates uncertainty estimation directly into SRGAN and ESRGAN, so every super-resolved image comes with a reliability map — telling you *where* the model is guessing.

---

## Motivation

Super-resolution GANs produce visually impressive results, but noise and hallucinated artifacts can be hard to detect. In high-stakes domains like **medical imaging**, **surveillance**, and **aerial imaging**, an overconfident wrong prediction can have real consequences.

This thesis addresses that gap by combining:
- State-of-the-art GANs (SRGAN, ESRGAN)
- Uncertainty estimation techniques (Monte Carlo Dropout, Ensemble Methods)

---

## Key Findings

### 1. Uncertainty-Aware Super-Resolution
Integrating uncertainty estimation produces enhanced images with a **quantified reliability score per pixel** — making it possible to pinpoint regions where the model is less confident and distinguish genuine anomalies from model artifacts.

### 2. Comparison of Uncertainty Methods
A systematic comparison of **Monte Carlo Dropout** vs **Ensemble Methods** across performance, training stability, and convergence behaviour in GAN training.

### 3. Improved Interpretability
Uncertainty maps allow researchers and practitioners to assess deployment risk — giving a concrete signal for when to trust (or not trust) a model's output.

---

## Repository Structure

```
gan-superresolution-uncertainty/
│
├── First_gan/          # Baseline GAN experiments
├── DCGAN/              # Deep Convolutional GAN implementation
├── Anime_DCGAN/        # DCGAN applied to anime dataset
├── dog_dcgan/          # DCGAN applied to dog dataset
├── CGAN/               # Conditional GAN
├── SRGAN/              # Super-Resolution GAN + uncertainty estimation
├── ESRGAN/             # Enhanced SRGAN + uncertainty estimation
├── Infrence/           # Inference scripts and uncertainty visualization
└── referneces/         # Papers and reference materials
```

---

## Methods

| Technique | Description |
|---|---|
| **SRGAN** | Super-Resolution GAN — baseline super-resolution model |
| **ESRGAN** | Enhanced SRGAN — improved perceptual quality |
| **MC Dropout** | Monte Carlo Dropout for approximate Bayesian inference |
| **Deep Ensembles** | Multiple model instances for uncertainty quantification |

---

## Citation

If you use this work, please cite:

```bibtex
@mastersthesis{adapa2023uncertainty,
  author  = {Maniraj Sai Adapa},
  title   = {Estimating Uncertainty in Generative Adversarial Neural Networks for Super-Resolution},
  school  = {University of Groningen},
  year    = {2023}
}
```

---

## Author

**Maniraj Sai Adapa** — [GitHub](https://github.com/thecr7guy2)