# FOC-Motion

**Text-Driven Human Motion Generation via Frequency-Orthogonal Representation and Confidence Guidance**

FOC-Motion introduces three targeted improvements to the MoMask framework for text-driven 3D human motion generation, addressing three previously overlooked aspects: frequency-domain perception, inter-layer representation constraints, and cross-stage information transfer.

> **Note:** The code and pretrained weights will be released here upon acceptance of the paper. This repository currently serves as a placeholder describing the method and planned release.

---

## Overview

Text-driven 3D human motion generation aims to synthesize realistic, semantically aligned human motion sequences from natural language descriptions. Most state-of-the-art masked generative models are built on a Residual Vector Quantization (RVQ-VAE) motion codebook whose representation quality directly bounds downstream generation quality.

FOC-Motion enhances this pipeline along three axes:

1. **Adaptive-Weight Multi-Scale DWT Frequency-Domain Loss** — A differentiable three-level `db4` wavelet decomposition applied along the time axis during RVQ-VAE training. Reconstruction weights are adaptively allocated according to the per-band energy of the ground-truth motion, explicitly decoupling low-frequency backbone semantics from high-frequency detail and preventing the network from smoothing away transient motion cues.

2. **MCR² Residual Subspace Orthogonality Constraint** — A Maximal Coding Rate Reduction (MCR²) regularizer applied to the gradient-bearing residual latents. An inter-layer orthogonality term expands the overall representation space while an intra-layer expansion term (bounded effective rank) preserves per-layer diversity, driving different residual layers to occupy mutually separated subspaces and reducing inter-layer redundancy.

3. **M-Transformer → R-Transformer Confidence Propagation** — The per-position maximum-softmax confidence produced by the base-layer M-Transformer is projected and injected into the R-Transformer input, establishing a cross-stage information channel that lets the residual stage treat low-confidence base-layer positions with greater caution.

The first two act on the **RVQ-VAE training stage** and add **no inference cost**; the third adds only a single confidence computation at inference.

---

## Method at a Glance

```
Input Motion ──► RVQ-VAE Encoder ──► 6-level Residual Vector Quantization ──► Decoder ──► Reconstructed Motion
                                         │
                    Training losses: Reconstruction + Embedding + [MCR² Loss] + [DWT Frequency Loss]

Text ──► CLIP ──► M-Transformer (base-layer tokens) ──[per-position confidence]──► R-Transformer (residual-layer tokens)
```

---

## Results (Highlights)

On **HumanML3D**, relative to our reproduced MoMask baseline:

- MCR² + DWT alone: **FID 0.080 → 0.059** (26.3% relative improvement, statistically significant, non-overlapping 95% CIs).
- Full method (+ confidence propagation): **FID 0.080 → 0.056** (30.0% relative improvement), with all three R-Precision metrics improving concurrently.

On **KIT-ML**, the frequency-domain loss yields an even more pronounced marginal benefit in the small-data regime.

Internal-mechanism verification (bootstrap RateReduction and per-layer EffRank) confirms that MCR² shifts the residual representations in the direction its design intends, providing a mechanistic explanation for the observed gains rather than a coincidental metric improvement.

*(See the paper for full tables, ablations, and confidence intervals.)*

---

## Planned Repository Structure

```
FOC-Motion/
├── configs/            # training / evaluation configs
├── models/             # RVQ-VAE, M-Transformer, R-Transformer
├── losses/             # DWT frequency-domain loss, MCR² regularizer
├── datasets/           # HumanML3D / KIT-ML loaders
├── scripts/            # train / evaluate / generate
├── checkpoints/        # pretrained weights (released on acceptance)
└── README.md
```

---

## Datasets

- **HumanML3D** — 14,616 motion sequences, 44,970 text descriptions, 263-dim pose features, 20 FPS.
- **KIT-ML** — 3,911 motion sequences, 251-dim pose features.

Both are processed following the T2M preprocessing pipeline and split 0.8 / 0.15 / 0.05 (train / test / val). Please obtain the datasets from their original sources and follow their respective licenses.

---

## Environment (planned)

- Single NVIDIA GeForce RTX 4080 (24 GB) is sufficient for training.
- PyTorch, PyWavelets (`db4` DWT), and standard motion-generation evaluation tooling.

A full `requirements.txt` and reproduction instructions will accompany the code release.

---



---

## Acknowledgements

This work builds upon [MoMask](https://github.com/EricGuo5513/momask-codes) and related open-source motion-generation projects. We thank the authors of MoMask, T2M, HumanML3D, and KIT-ML for making their data and code available.

## License

To be specified upon release.
