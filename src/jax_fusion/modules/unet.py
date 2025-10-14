from dataclasses import dataclass

import jax
import jax.numpy as jnp
import flax.nnx as nnx


class ConvBlock(nnx.Module):
    def __init__(self, in_channels, out_channels, *, rngs):
        self.conv1 = nnx.Conv(in_channels, out_channels, kernel_size=(3, 3), padding='SAME', rngs=rngs)
        self.conv2 = nnx.Conv(out_channels, out_channels, kernel_size=(3, 3), padding='SAME', rngs=rngs)

    def __call__(self, x):
        x = nnx.relu(self.conv1(x))
        x = nnx.relu(self.conv2(x))
        return x
class Down(nnx.Module):
    def __init__(self, in_channels, out_channels, *, rngs):
        self.block = ConvBlock(in_channels, out_channels, rngs=rngs)
    
    def __call__(self, x):
        x = nnx.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = self.block(x)
        return x
class Up(nnx.Module):
    def __init__(self, in_channels, skip_channels, out_channels, *, rngs):
        self.fup = nnx.ConvTranspose(in_channels, out_channels, kernel_size=(2, 2), strides=(2, 2), rngs=rngs)
        self.block = ConvBlock(out_channels+skip_channels, out_channels, rngs=rngs)
    
    def __call__(self, x, skip):
        x = self.fup(x)
        x = jnp.concatenate([x, skip], axis=-1)
        x = self.block(x)
        return x
class UNet(nnx.Module):
    def __init__(self, in_channels, num_classes, *, rngs):
        self.enc1 = ConvBlock(in_channels, 64, rngs=rngs)
        self.enc2 = Down(64, 128, rngs=rngs) 
        self.enc3 = Down(128, 256, rngs=rngs)
        self.enc4 = Down(256, 512, rngs=rngs)

        self.middle = ConvBlock(512, 1024, rngs=rngs)

        self.up1 = Up(1024, 256, 256, rngs=rngs)
        self.up2 = Up(256, 128, 128, rngs=rngs)
        self.up3 = Up(128, 64, 64, rngs=rngs)

        self.out_conv = nnx.Conv(64, num_classes, kernel_size=(1, 1), rngs=rngs)

    def __call__(self, x):
        e1 = self.enc1(x) # 1, 128, 128, 64
        e2 = self.enc2(e1) # 1, 64, 64, 128
        e3 = self.enc3(e2) # 1, 32, 32, 256
        e4 = self.enc4(e3) # 1, 16, 16, 512
        
        mid = self.middle(e4) # 1, 16, 16, 1024

        d1 = self.up1(mid, e3) # 1, 32, 32, 256
        d2 = self.up2(d1, e2) # 1, 64, 64, 128
        d3 = self.up3(d2, e1) # 1, 128, 128, 64

        logits = self.out_conv(d3) # 1, 256, 256, num_classes
        return logits
    


if __name__ == "__main__":
    rngs = nnx.Rngs(default=0)
    model = UNet(in_channels=3, num_classes=10, rngs=rngs)
    x = jnp.ones((1, 128, 128, 3))  # batch size 1, RGB input
    y = model(x)
    print(y.shape)