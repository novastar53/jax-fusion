import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

from jax_fusion.datasets.fashion_mnist import DataConfig, make_dataloader

cfg = DataConfig(batch_size=16, num_epochs=1, shuffle=True, as_chw=False, sample_size=64)
it = make_dataloader("test", cfg)
imgs, labs = next(it)


# Load and preprocess grayscale image
def load_image(path, resize=(64, 64)):
    img = Image.open(path).convert("L").resize(resize)
    img_np = np.asarray(img).astype(np.float32) / 255.0
    return jnp.array(img_np)

# Forward diffusion step (reparameterization form)
def q_xt_given_xt_minus_1(x_prev, beta_t, rng):
    noise = jax.random.normal(rng, shape=x_prev.shape)
    mean = jnp.sqrt(1 - beta_t) * x_prev
    std = jnp.sqrt(beta_t)
    return mean + std * noise

# Run T steps of forward diffusion
def forward_diffusion(x0, betas, key):
    xts = [x0]
    x_prev = x0
    for t in range(len(betas)):
        key, subkey = jax.random.split(key)
        x_next = q_xt_given_xt_minus_1(x_prev, betas[t], subkey)
        xts.append(x_next)
        x_prev = x_next
    return xts

# Plot multiple images
def show_images(images, titles=None, cols=5):
    rows = (len(images) + cols - 1) // cols
    plt.figure(figsize=(cols * 2, rows * 2))
    for i, img in enumerate(images):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(np.clip(np.array(img), 0, 1), cmap='gray')
        if titles: plt.title(titles[i])
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# === Main ===
if __name__ == "__main__":
    x0 = imgs[0]  # Use the first image from the dataloader
    x0 = x0.astype(jnp.float32) / 255.0  # Ensure it's in [0,1] range

    T = 10  # number of diffusion steps
    betas = jnp.linspace(1e-5, 0.001, T)
    rng = jax.random.PRNGKey(42)

    xts = forward_diffusion(x0, betas, rng)

    # Show selected steps
    selected = [0, 1, 2, 3, 5, 7, 9, T]
    show_images([xts[t] for t in selected],
                titles=[f"t={t}" for t in selected])