from dataclasses import dataclass

import jax
import jax.numpy as jnp
import flax.nnx as nnx

@dataclass
class Config:
    init_dim: int = 0

class UNet(nnx.Module):
    def __init__(self, config):
        self.ups = []
        self.downs = []
    

    def __call__(self, x):
        pass
