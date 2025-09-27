import jax
import jax.numpy as jnp
import flax.nnx as nnx

from jax_fusion.datasets.fashion_mnist import make_dataloader

train_dataloader = make_dataloader("train")
test_dataloader = make_dataloader("test")


