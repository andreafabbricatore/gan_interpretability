# StyleGAN Interpretability: Disentangling Facial Attributes

This repository implements a hybrid framework for discovering and manipulating interpretable directions in StyleGAN2's latent space, combining supervised, unsupervised, and CLIP-guided approaches for facial attribute editing.

## Overview

The project provides a comprehensive toolkit for:
- Supervised discovery of disentangled directions using SVM classifiers
- Unsupervised PCA-based direction discovery inspired by GANSpace
- CLIP-guided optimization for flexible attribute manipulation
- Localized and interpretable editing of facial attributes

## Project Structure

```
.
├── data/                           # Dataset and training data storage
│   ├── celeba/                     # CelebA dataset images and annotations
│   ├── svm_training_data/          # Preprocessed data for SVM training
│   └── feature_based/              # Feature-based data splitting
│
├── modules/                        # External dependencies and model implementations
│   ├── stylegan2-ada-pytorch/      # NVIDIA's StyleGAN2-ADA implementation (with our modifications)
│   └── encoder4editing/            # E4E encoder for latent space projection
│
├── notebooks/                      # Interactive development and analysis
│   ├── dataset.ipynb              # Dataset preparation and preprocessing
│   ├── training.ipynb             # Model training and direction discovery
│   └── inference.ipynb            # Interactive inference and visualization
│
├── outputs/                        # Generated results and visualizations
│   ├── directions/                # Discovered latent directions for each attribute
│   ├── inference/                 # Results from applying directions to images
│   ├── latents/                   # Generated and processed latent codes
│   └── training_steps/            # Intermediate results during training
│
└── ganexplainer/                  # Virtual Environment for project dependencies
```

### Detailed Component Descriptions

#### Data Directory
- **celeba/**: Contains the CelebA dataset images and attribute annotations used for training and evaluation
- **svm_training_data/**: Stores preprocessed latent codes and attribute labels for SVM training
- **feature_based/**: Contains results from feature-based analysis of discovered directions

#### Modules Directory
- **stylegan2-ada-pytorch/**: NVIDIA's official StyleGAN2-ADA implementation, used as the base generative model
- **encoder4editing/**: E4E encoder implementation for projecting real images into the StyleGAN2 latent space

#### Notebooks Directory
- **dataset.ipynb**: Handles dataset preparation, including:
  - CelebA dataset loading and preprocessing
  - Attribute annotation processing
  - Latent code generation for training data
- **training.ipynb**: Implements the core training procedures:
  - SVM classifier training for supervised direction discovery
  - PCA-based unsupervised direction discovery
  - CLIP-guided optimization setup
- **inference.ipynb**: Provides interactive tools for:
  - Loading and applying discovered directions
  - Real-time attribute editing
  - Result visualization and comparison

#### Outputs Directory
- **directions/**: Stores discovered latent directions for each attribute
- **inference/**: Contains results from applying discovered directions to images
- **latents/**: Stores generated and processed latent codes
- **training_steps/**: Contains intermediate results and checkpoints during the training process

## Setup

1. Setup virtual environment:
```bash
source ganexplainer/bin/activate
```

3. Download pre-trained models:
- StyleGAN2-ADA model
- E4E encoder
- CLIP model

## Features

- **Supervised Learning**: Train SVM classifiers on W+ latent codes to discover attribute-specific directions
- **Unsupervised Analysis**: Extract principal components from layer activations for semantic variations
- **CLIP Integration**: Use CLIP for flexible, text-guided attribute manipulation
- **Localized Editing**: Precise control over facial attributes with minimal interference
- **Visualization Tools**: Interactive notebooks for exploring and visualizing latent directions

## Acknowledgments

- StyleGAN2-ADA implementation from NVIDIA
- E4E encoder implementation
- CLIP model from OpenAI
- GANSpace for inspiration on unsupervised direction discovery
