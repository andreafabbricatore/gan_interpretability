import os
import re
from typing import List, Optional

import click
import dnnlib
import numpy as np
import PIL.Image
import torch

import legacy

# -------------------------
# CLI helpers
# -------------------------

def num_range(s: str) -> List[int]:
    """Parse a string input into a list of integers.
    
    The input can be either:
    - A comma-separated list of numbers (e.g. '1,2,3')
    - A range of numbers (e.g. '1-3')
    
    Returns:
        List[int]: List of parsed integers
    """
    range_re = re.compile(r'^(\d+)-(\d+)$')
    m = range_re.match(s)
    if m:
        return list(range(int(m.group(1)), int(m.group(2)) + 1))
    return [int(x) for x in s.split(',')]

def apply_directions_to_w(w, directions, alpha=1.0):
    """
    Apply 18 per-layer direction vectors to W+ latent w
    Args:
        w: [1, 18, 512]
        directions: [18, 512]
        alpha: scalar or [18]
    Returns:
        Modified w with directions applied
    """
    directions = directions.to(w.device)
    directions = directions / directions.norm(dim=1, keepdim=True)

    if isinstance(alpha, torch.Tensor):
        alpha = alpha.to(w.device).view(1, -1, 1)  # shape [1, 18, 1]
    else:
        alpha = torch.tensor(alpha, device=w.device).view(1, 1, 1)  # broadcast scalar

    return w + alpha * directions.unsqueeze(0)

# -------------------------
# Main command
# -------------------------

@click.command()
@click.option('--network', 'network_pkl', required=True, help='Network pickle filename')
@click.option('--seeds', type=num_range, required=True, help='List of random seeds')
@click.option('--trunc', 'truncation_psi', type=float, default=1.0, show_default=True)
@click.option('--noise-mode', type=click.Choice(['const', 'random', 'none']), default='const', show_default=True)
@click.option('--direction-pt', type=str, help='Path to 18-layer direction .pt file (from CLIP training)')
@click.option('--alpha', type=float, default=1.0, help='Magnitude of directional edit')
@click.option('--outdir', type=str, required=True, help='Directory to save images')
def generate_images(
    network_pkl: str,
    seeds: List[int],
    truncation_psi: float,
    noise_mode: str,
    direction_pt: Optional[str],
    alpha: float,
    outdir: str
):
    """Generate images using pretrained StyleGAN2 network with optional attribute editing.

    This function generates images from random seeds using a pretrained StyleGAN2 network.
    If a direction file is provided, it will also generate edited versions of each image
    by applying the learned attribute directions with both positive and negative alpha values.

    Args:
        network_pkl (str): Path to pretrained StyleGAN2 network pickle file
        seeds (List[int]): List of random seeds to generate images from
        truncation_psi (float): Truncation psi value for latent space sampling
        noise_mode (str): Noise mode for image generation ('const', 'random', or 'none')
        direction_pt (Optional[str]): Path to .pt file containing learned attribute directions and alphas
        alpha (float): Magnitude of directional edit to apply
        outdir (str): Directory to save generated images

    The function will:
    1. Load the pretrained StyleGAN2 generator
    2. For each seed:
        - Generate the original image
        - If directions and alphas are provided, generate edited versions with +alpha and -alpha
    3. Save all images to the specified output directory
    """
    print(f'Loading network from "{network_pkl}"...')
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)

    os.makedirs(outdir, exist_ok=True)

    # Load attribute directions (optional)
    directions = None
    if direction_pt:
        print(f"Loading directions from {direction_pt}")
        direction_name = direction_pt.split('/')[-2]
        direction_data = torch.load(direction_pt)
        directions = direction_data['directions'].to(device)
        alphas = direction_data.get('alphas', torch.ones(G.num_ws, device=device) * alpha)
        assert directions.shape == (G.num_ws, G.w_dim), f"Expected [18, 512], got {directions.shape}"

    label = torch.zeros([1, G.c_dim], device=device)

    for seed_idx, seed in enumerate(seeds):
        print(f'Generating image for seed {seed}...')
        z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)
        w = G.mapping(z, label, truncation_psi=truncation_psi)

        # Generate original
        img_orig = G.synthesis(w, noise_mode=noise_mode)
        img_orig = (img_orig.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        os.makedirs(f'{outdir}/{direction_name}', exist_ok=True)
        PIL.Image.fromarray(img_orig[0].cpu().numpy(), 'RGB').save(f'{outdir}/{direction_name}/seed{seed:04d}_original.png')

        # If direction is provided, edit and generate
        if directions is not None:
            w_more = apply_directions_to_w(w, directions, alpha=alphas)
            w_less = apply_directions_to_w(w, directions, alpha=-alphas)


            for kind, w_mod in [('more', w_more), ('less', w_less)]:
                img = G.synthesis(w_mod, noise_mode=noise_mode)
                img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
                PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'{outdir}/{direction_name}/{seed:04d}_{kind}_edit.png')

    print("Done.")

# -------------------------

if __name__ == "__main__":
    generate_images()
