# StyleGAN Interpretability: Disentangling Facial Attributes

This repository implements a framework for discovering and manipulating interpretable directions in StyleGAN2's latent space using CLIP-guided approaches for facial attribute editing.

## Overview

The project provides a comprehensive toolkit for:
- CLIP-guided optimization for flexible attribute manipulation
- Localized and interpretable editing of facial attributes

## Project Structure

```
.
├── modules/                        # External dependencies and model implementations
│   ├── stylegan2-ada-pytorch/      # NVIDIA's StyleGAN2-ADA implementation (with our modifications)
│   └── encoder4editing/            # E4E encoder for latent space projection
│
├── outputs/                        # Generated results and visualizations
│   ├── inference/                 # Results from applying directions to images
│   └── training_steps/            # Intermediate results during training
│
└── ganexplainer/                  # Virtual Environment for project dependencies
```

### Detailed Component Descriptions

## Setup

1. Setup virtual environment:
```bash
source ganexplainer/bin/activate
```

3. Download pre-trained models:
- StyleGAN2-ADA model
- E4E encoder
- CLIP model