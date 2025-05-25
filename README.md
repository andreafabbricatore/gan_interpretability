# StyleGAN Interpretability: Disentangling Facial Attributes

This repository presents a flexible and interpretable framework for manipulating facial attributes in StyleGAN2’s latent space using CLIP-guided optimization. Our goal is to enable fine-grained, localized edits of facial features through semantically meaningful latent directions.

---

## Project Structure

```

.
├── modules/                         # External dependencies and model code
│   └── stylegan2-ada-pytorch/       # NVIDIA's StyleGAN2-ADA (with our modifications)
│
├── outputs/                         # Generated outputs and visualizations
│   ├── inference/                   # Results from applying learned directions to test images
│   └── training_steps_*/            # Intermediate training artifacts and final latent directions
│
└── ganexplainer/                    # Virtual environment for managing project dependencies

````

---

## Our modifications

In the stylegan2-ada-pytorch/ folder the following files were edited by us:
- generate_trainable.py - auxiliary function to allow training of directions (used by train_direction_variant_a and train_direction_variant_c)
- generate_trainable_alphas.py  - auxiliary function to allow training of directions (used by train_direction_variant_b and train_direction_variant_d)
- generate_from_trainable.py - auxiliary function to allow inference (version where only directions are edited)
- generate_from_trainable_alphas.py - auxiliary function to allow inference (version where alphas and directions are edited)

- train_direction_variant_a.py (as explained in paper)
- train_direction_variant_b.py (as explained in paper)
- train_direction_variant_c.py (as explained in paper)
- train_direction_variant_d.py (as explained in paper)

## Setup Instructions

1. Install dependencies

    ```bash
    pip install -r requirements.txt
    ````

2. Download the FFHQ weights from the official NVIDIA release:

   [https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl](https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl)

   Place the file inside:

   ```
   modules/stylegan2-ada-pytorch/
   ```
