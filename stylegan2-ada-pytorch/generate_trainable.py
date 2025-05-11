def generate_image_from_w(G, w, directions=None, alpha=1.0, noise_mode='const'):
    """
    Applies 18 layer-wise directions to the W+ latent and returns generated image.

    Args:
        G: StyleGAN2 Generator
        w: Latent tensor [1, 18, 512]
        directions: Tensor [18, 512], one direction per layer
        alpha: Step size to apply directions
        noise_mode: 'const', 'random', or 'none'

    Returns:
        img: Generated image tensor
        edited_w: Edited latent vector
    """
    w = w.clone()

    if directions is not None:
        directions = directions / directions.norm(dim=1, keepdim=True)  # normalize each
        w += alpha * directions.unsqueeze(0)  # broadcast: [1, 18, 512]

    img = G.synthesis(w, noise_mode=noise_mode)
    return img, w
