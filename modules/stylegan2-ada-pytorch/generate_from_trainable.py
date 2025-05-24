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
        alpha: scalar
    Returns:
        Modified w with directions applied
    """
    directions = directions.to(w.device)
    directions = directions / directions.norm(dim=1, keepdim=True)
    return w + alpha * directions.unsqueeze(0)

# -------------------------
# Main command
# -------------------------

@click.command()
@click.pass_context
@click.option('--network', 'network_pkl', required=True, help='Network pickle filename')
@click.option('--seeds', type=num_range, required=True, help='List of random seeds')
@click.option('--trunc', 'truncation_psi', type=float, default=1.0, show_default=True)
@click.option('--noise-mode', type=click.Choice(['const', 'random', 'none']), default='const', show_default=True)
@click.option('--direction-pt', type=str, help='Path to 18-layer direction .pt file (from CLIP training)')
@click.option('--alpha', type=float, default=1.0, help='Magnitude of directional edit')
@click.option('--outdir', type=str, required=True, help='Directory to save images')
def generate_images(
    ctx: click.Context,
    network_pkl: str,
    seeds: List[int],
    truncation_psi: float,
    noise_mode: str,
    direction_pt: Optional[str],
    alpha: float,
    outdir: str
):
    print(f'Loading network from "{network_pkl}"...')
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)

    os.makedirs(outdir, exist_ok=True)

    # Load attribute directions (optional)
    directions = None
    if direction_pt:
        print(f"Loading directions from {direction_pt}")
        direction_name = direction_pt.split('/')[-1].split('_directions.pt')[0]
        directions = torch.load(direction_pt)['directions'].to(device)  # [18, 512]
        assert directions.shape == (G.num_ws, G.w_dim), f"Expected [18, 512], got {directions.shape}"

    label = torch.zeros([1, G.c_dim], device=device)

    for seed_idx, seed in enumerate(seeds):
        print(f'Generating image for seed {seed}...')
        z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)
        w = G.mapping(z, label, truncation_psi=truncation_psi)

        # Generate original
        img_orig = G.synthesis(w, noise_mode=noise_mode)
        img_orig = (img_orig.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        PIL.Image.fromarray(img_orig[0].cpu().numpy(), 'RGB').save(f'{outdir}/{direction_name}/seed{seed:04d}_original.png')

        # If direction is provided, edit and generate
        if directions is not None:
            w_more = apply_directions_to_w(w, directions, alpha=alpha)
            w_less = apply_directions_to_w(w, directions, alpha=-alpha)

            for kind, w_mod in [('more', w_more), ('less', w_less)]:
                img = G.synthesis(w_mod, noise_mode=noise_mode)
                img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
                PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'{outdir}/{direction_name}/{seed:04d}_{kind}_edit.png')

    print("Done.")

# -------------------------

if __name__ == "__main__":
    generate_images()
