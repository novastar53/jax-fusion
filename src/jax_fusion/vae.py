from dataclasses import dataclass

import jax
import jax.numpy as jnp
import flax.nnx as nnx

import optax

from jax_fusion.datasets.fashion_mnist import make_dataloader, visualize_batch, FASHION_LABELS

train_it = make_dataloader("train")
test_it = make_dataloader("test")

@dataclass
class Config:
    hidden_size: int = 8


class VAE(nnx.Module):
    def __init__(self, config: Config, rngs: nnx.Rngs):
        self.config = config
        self.conv1 = nnx.Conv(in_features=1, out_features=32, kernel_size=(3, 3), strides=(2, 2), padding='SAME', rngs=rngs)
        self.conv2 = nnx.Conv(in_features=32, out_features=64, kernel_size=(3, 3), strides=(2, 2), padding='SAME', rngs=rngs)
        self.linear1 = nnx.Linear(7*7*64, 2 * config.hidden_size, rngs=rngs)
        self.linear2 = nnx.Linear(config.hidden_size, 7*7*64, rngs=rngs)
        self.deconv1 = nnx.ConvTranspose(in_features=64, out_features=32, kernel_size=(3, 3), strides=(2, 2), padding='SAME', rngs=rngs)
        self.deconv2 = nnx.ConvTranspose(in_features=32, out_features=1, kernel_size=(3, 3), strides=(2, 2), padding='SAME', rngs=rngs)
    

    def __call__(self, batch, key: jnp.ndarray = None):
        B, _, _, _ = batch.shape
        batch = batch.transpose(0, 2, 3, 1)
        x = self.conv1(batch)
        x = self.conv2(x)
        x = x.reshape(B, -1)
        x = self.linear1(x)
        mu, log_var = jnp.split(x, 2, axis=1)
        key, subkey = jax.random.split(key)
        epsilon = jax.random.normal(subkey, log_var.shape)
        l = mu + log_var * epsilon
        x = self.linear2(l)
        x = x.reshape(B, 7, 7, 64)
        x = self.deconv1(x)
        y = self.deconv2(x)
        assert(batch.shape == y.shape)
        y = y.transpose(0, 3, 1, 2)
        return y, key

rngs = nnx.Rngs(default=0)
m = VAE(Config(), rngs)    

tx = optax.adam(1e-2)
optimizer = nnx.Optimizer(m, tx, wrt=nnx.Param)


def loss_fn(m, x, key):
    y, key = m(x, key)
    assert(y.shape == x.shape)
    loss =  jnp.sum((y - x) ** 2) / y.shape[0]
    return loss, (y, key)


def step_fn(m, x, key):
    (loss, (y, key)), grads = nnx.value_and_grad(loss_fn, has_aux=True)(m, x, key)
    optimizer.update(m, grads)
    return loss, y, key

key = jax.random.key(42)
for i, (x, labels) in enumerate(train_it):
    loss, y, key = step_fn(m, x, key)
    print(i, loss)

visualize_batch(y, labels, FASHION_LABELS)