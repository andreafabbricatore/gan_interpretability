import os
import torch
import clip
import dnnlib
import legacy
import numpy as np
from typing import List
from torchvision.transforms import Normalize, Resize, Compose
from torchvision.utils import save_image
from generate_trainable import generate_image_from_w
from PIL import Image
import click


def load_generator(network_pkl, device='cuda'):
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)
    return G

def sample_w(G, seed=42, trunc=0.7, device='cuda'):
    z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)
    label = torch.zeros([1, G.c_dim], device=device)
    w = G.mapping(z, label, truncation_psi=trunc)
    assert w.shape == (1, G.num_ws, G.w_dim), f"Expected w shape [1, {G.num_ws}, {G.w_dim}], got {w.shape}"
    return w

def preprocess_for_clip(img_tensor):
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
    img_list: List[torch.Tensor],
    target_text,
    steps=1000,
    lr=0.05,
    alpha=1.0,
    log_interval=10,
    patience=3,
    lambda_id=0.8,
    save_dir="out/train_direction"
):
    device = clip_model.visual.conv1.weight.device
    target_slug = target_text.replace(" ", "_")
    target_path = os.path.join(save_dir, target_slug)
    os.makedirs(target_path, exist_ok=True)

    directions = torch.nn.Parameter(torch.randn(G.num_ws, G.w_dim, device=device))
    optimizer = torch.optim.Adam([directions], lr=lr)  # only directions

    # Precompute text embedding
    text_tokens = clip.tokenize([target_text]).to(device)
    with torch.no_grad():
        text_embed = clip_model.encode_text(text_tokens).float()
        text_embed = text_embed / text_embed.norm(dim=-1, keepdim=True)
        text_embed = text_embed.squeeze(0)

    best_loss = float('inf')
    best_directions = None
    best_img = None
    patience_counter = 0

    for step in range(steps):
        optimizer.zero_grad()
        total_loss = 0.0

        for i, w in enumerate(w_list):
            # Get original image and its embedding
            with torch.no_grad():
                img_before, _ = generate_image_from_w(G, w, directions=None, alpha=alpha)
                clip_img_before = preprocess_for_clip(img_before).to(device)
                embed_before = clip_model.encode_image(clip_img_before).float()
                embed_before = embed_before / embed_before.norm(dim=-1, keepdim=True)

            # Get edited image and its embedding
            img_after, _ = generate_image_from_w(G, w, directions, alpha=alpha)
            clip_img_after = preprocess_for_clip(img_after).to(device)
            embed_after = clip_model.encode_image(clip_img_after).float()
            embed_after = embed_after / embed_after.norm(dim=-1, keepdim=True)

            # Directional alignment loss
            direction_target = (text_embed - embed_before).detach()
            direction_pred = (embed_after - embed_before)
            loss_attr = -torch.cosine_similarity(direction_pred, direction_target, dim=-1).mean()

            # Identity preservation loss
            loss_id = torch.nn.functional.mse_loss(embed_after, embed_before)

            # Total loss
            loss = loss_attr + lambda_id * loss_id
            loss.backward()
            total_loss += loss.item()

        optimizer.step()
        avg_loss = total_loss / len(w_list)

        if step % log_interval == 0 or step == steps - 1:
            print(f"[{target_text}] Step {step}/{steps} | Avg Loss: {avg_loss:.4f}")
            save_image((img_after.clamp(-1, 1) + 1) / 2, os.path.join(target_path, f'step{step}.png'))

        if avg_loss < best_loss - 1e-4:
            best_loss = avg_loss
            best_directions = directions.detach().clone()
            best_img = img_after.detach().clone()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"[{target_text}] Early stopping at step {step}. No improvement for {patience} steps.")
                break

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
@click.option('--save-dir', default='../../outputs/training_steps_variant_c/', help='Directory to save results')
@click.option('--top-k', default=5, help='Number of top samples to keep after filtering')
@click.option('--lambda-id', default=0.8, help='Lambda value for identity preservation')
def main(network_pkl, targets, seeds, steps, lr, alpha, log_interval, patience, save_dir, top_k, lambda_id):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Parse targets and seeds
    targets = [t.strip() for t in targets.split(',')]
    if '-' in seeds:
        start, end = map(int, seeds.split('-'))
        seeds = list(range(start, end + 1))
    else:
        seeds = [int(s) for s in seeds.split(',')]

    G = load_generator(network_pkl, device)
    clip_model, _ = clip.load("ViT-B/32", device=device)
    clip_model.eval()

    filtered_w_lists = {}
    filtered_img_lists = {}

    for target in targets:
        tokens = clip.tokenize([target]).to(device)
        scores = []
        w_pool = [sample_w(G, seed=seed, device=device) for seed in seeds]

        with torch.no_grad():
            for w in w_pool:
                img, _ = generate_image_from_w(G, w, directions=None, alpha=0.0)
                clip_img = preprocess_for_clip(img)  # shape [3, 224, 224]
                clip_img = clip_img.to(device)  # shape [1, 3, 224, 224]
                logits_per_image, _ = clip_model(clip_img, tokens)
                score = logits_per_image.item()
                scores.append((score, w, img))

        scores.sort(key=lambda x: x[0])
        w_list = [w for _, w, _ in scores[:top_k]]
        img_list = [img for _, _, img in scores[:top_k]]
        filtered_w_lists[target] = w_list
        filtered_img_lists[target] = img_list

    for target in targets:
        train_direction_for_target(
            G=G,
            clip_model=clip_model,
            w_list=filtered_w_lists[target],
            img_list=filtered_img_lists[target],
            target_text=target,
            steps=steps,
            lr=lr,
            alpha=alpha,
            log_interval=log_interval,
            patience=patience,
            save_dir=save_dir,
            lambda_id=lambda_id
        )

if __name__ == "__main__":
    main()
