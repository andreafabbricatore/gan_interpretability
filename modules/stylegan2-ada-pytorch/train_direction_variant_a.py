import os
import torch
import clip
import dnnlib
import legacy
import click
import numpy as np
from typing import List
from torchvision.transforms import Normalize, Resize, Compose
from torchvision.utils import save_image
from generate_trainable import generate_image_from_w
from PIL import Image

def load_generator(network_pkl, device='cuda'):
    """
    Load a pre-trained StyleGAN2 generator from a pickle file.
    
    Args:
        network_pkl (str): Path to the network pickle file
        device (str): Device to load the model on ('cuda' or 'cpu')
    
    Returns:
        torch.nn.Module: The loaded generator model
    """
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)
    return G

def sample_w(G, seed=42, trunc=0.7, device='cuda'):
    """
    Sample a latent vector w from the generator's latent space.
    
    Args:
        G: StyleGAN2 generator
        seed (int): Random seed for reproducibility
        trunc (float): Truncation psi value
        device (str): Device to use
    
    Returns:
        torch.Tensor: Sampled latent vector w
    """
    z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)
    label = torch.zeros([1, G.c_dim], device=device)
    w = G.mapping(z, label, truncation_psi=trunc)
    assert w.shape == (1, G.num_ws, G.w_dim), f"Expected w shape [1, {G.num_ws}, {G.w_dim}], got {w.shape}"
    return w

def preprocess_for_clip(img_tensor):
    """
    Preprocess an image tensor for CLIP model input.
    
    Args:
        img_tensor (torch.Tensor): Input image tensor in range [-1, 1]
    
    Returns:
        torch.Tensor: Preprocessed image tensor
    """
    transform = Compose([
        Resize((224, 224)),
        Normalize((0.48145466, 0.4578275, 0.40821073),
                  (0.26862954, 0.26130258, 0.27577711))
    ])
    return transform((img_tensor + 1) / 2)

def train_direction_for_target(
    G,
    clip_model,
    w_list: List[torch.Tensor],
    target_text,
    steps=1000,
    lr=0.05,
    alpha=1.0,
    log_interval=10,
    patience=3,
    save_dir="out/train_direction"
):
    """
    Train a semantic direction for a specific target attribute.
    
    Args:
        G: StyleGAN2 generator
        clip_model: CLIP model for text-image similarity
        w_list: List of latent vectors to train on
        target_text (str): Target attribute text description
        steps (int): Number of training steps
        lr (float): Learning rate
        alpha (float): Direction application strength
        log_interval (int): Interval for logging progress
        patience (int): Early stopping patience
        save_dir (str): Directory to save results
    """
    device = clip_model.visual.conv1.weight.device
    target_slug = target_text.replace(" ", "_")
    target_path = os.path.join(save_dir, target_slug)
    os.makedirs(target_path, exist_ok=True)

    # Initialize trainable direction vector
    directions = torch.randn(G.num_ws, G.w_dim, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([directions], lr=lr)
    tokens = clip.tokenize([target_text]).to(device)

    # Training loop variables
    best_loss = float('inf')
    best_directions = None
    best_img = None
    patience_counter = 0

    for step in range(steps):
        optimizer.zero_grad()
        total_loss = 0.0
        
        # Process each latent vector in the batch
        for w in w_list:
            # Generate image with current direction
            img, _ = generate_image_from_w(G, w, directions, alpha=alpha)
            clip_img = preprocess_for_clip(img).to(device)
            logits_per_image, _ = clip_model(clip_img, tokens)
            loss = -logits_per_image.mean()  # Negative because we want to maximize similarity
            loss.backward()
            total_loss += loss.item()

        optimizer.step()
        avg_loss = total_loss / len(w_list)

        # Logging and visualization
        if step % log_interval == 0 or step == steps - 1:
            print(f"[{target_text}] Step {step}/{steps} | Avg Loss: {avg_loss:.4f}")
            save_image((img.clamp(-1, 1) + 1) / 2, os.path.join(target_path, f'step{step}.png'))

        # Early stopping check
        if avg_loss < best_loss - 1e-4:
            best_loss = avg_loss
            best_directions = directions.detach().clone()
            best_img = img.detach().clone()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"[{target_text}] Early stopping at step {step}. No improvement for {patience} steps.")
                break

    # Save final results
    if best_img is not None:
        final_np = ((best_img.clamp(-1, 1) + 1) * 127.5).permute(0, 2, 3, 1).to(torch.uint8).cpu().numpy()
        Image.fromarray(final_np[0]).save(os.path.join(target_path, 'final_result.png'))

    torch.save({'directions': best_directions.cpu()}, os.path.join(target_path, 'directions.pt'))
    print(f"[{target_text}] Done. Saved to: {target_path}")

@click.command()
@click.option('--network', 'network_pkl', default='ffhq.pkl', help='Network pickle filename')
@click.option('--targets', default='blond hair,beard,smiling,bald,eyeglasses', help='Comma-separated list of target attributes')
@click.option('--seeds', default='0-49', help='Range of seeds to use (e.g. 0-49)')
@click.option('--steps', default=1000, help='Number of training steps')
@click.option('--lr', default=0.05, help='Learning rate')
@click.option('--alpha', default=1.0, help='Alpha value for direction application')
@click.option('--log-interval', default=25, help='Interval for logging progress')
@click.option('--patience', default=10, help='Patience for early stopping')
@click.option('--save-dir', default='../../outputs/training_steps_variant_aaa/', help='Directory to save results')
@click.option('--top-k', default=5, help='Number of top samples to keep after filtering')
def main(network_pkl, targets, seeds, steps, lr, alpha, log_interval, patience, save_dir, top_k):
    """
    Main training script for finding semantic directions in StyleGAN2's latent space.
    
    Args:
        network_pkl (str): Path to the StyleGAN2 network pickle file
        targets (str): Comma-separated list of target attributes
        seeds (str): Range or list of seeds to use for sampling
        steps (int): Number of training steps
        lr (float): Learning rate
        alpha (float): Direction application strength
        log_interval (int): Interval for logging progress
        patience (int): Early stopping patience
        save_dir (str): Directory to save results
        top_k (int): Number of top samples to keep after filtering
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Parse targets and seeds
    targets = [t.strip() for t in targets.split(',')]
    if '-' in seeds:
        start, end = map(int, seeds.split('-'))
        seeds = list(range(start, end + 1))
    else:
        seeds = [int(s) for s in seeds.split(',')]

    # Load models
    G = load_generator(network_pkl, device)
    clip_model, _ = clip.load("ViT-B/32", device=device)
    clip_model.eval()

    # Filter latent vectors for each target
    filtered_w_lists = {}
    for target in targets:
        tokens = clip.tokenize([target]).to(device)
        scores = []
        w_pool = [sample_w(G, seed=seed, device=device) for seed in seeds]
        
        # Score each latent vector based on CLIP similarity
        with torch.no_grad():
            for w in w_pool:
                img, _ = generate_image_from_w(G, w, directions=None, alpha=0.0)
                clip_img = preprocess_for_clip(img).to(device)
                logits_per_image, _ = clip_model(clip_img, tokens)
                score = logits_per_image.item()
                scores.append((score, w))

        # sort by lowest similarity and keep bottom-k
        scores.sort(key=lambda x: x[0])
        w_list = [w for _, w in scores[:top_k]]
        filtered_w_lists[target] = w_list

    for target in targets:
        train_direction_for_target(
            G=G,
            clip_model=clip_model,
            w_list=filtered_w_lists[target],
            target_text=target,
            steps=steps,
            lr=lr,
            alpha=alpha,
            log_interval=log_interval,
            patience=patience,
            save_dir=save_dir
        )

if __name__ == "__main__":
    main()
