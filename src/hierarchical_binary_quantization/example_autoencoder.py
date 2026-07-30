"""
    Thx, claude for writing an example quantizing autoencoder
"""
import torch
from torch import nn, Tensor
from jaxtyping import Float, Int, jaxtyped
from beartype import beartype
from dataclasses import dataclass

@dataclass
class AutoencoderResults:
    latents: Tensor

@dataclass
class QuantizingAutoencoderResults:
    latents: Tensor
    quantized_latents: Tensor
    bit_codes: Tensor
    tokens: Tensor | None


def _gn(channels:int) -> nn.GroupNorm:
    num_groups = 8
    while channels % num_groups != 0:
        num_groups //= 2
    return nn.GroupNorm(num_groups, channels)

class GroupNormResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.blocks = nn.Sequential(
            _gn(channels),
            nn.GELU(),
            nn.Conv2d(channels,channels,kernel_size=3,padding=1),
            _gn(channels),
            nn.GELU(),
            nn.Conv2d(channels,channels,kernel_size=3,padding=1),                      
        )
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return x + self.blocks(x)


class ExampleAutoencoder(nn.Module):
    def __init__(
            self,
            in_channels:int=3, 
            base_dim:int=64,
            channel_multipliers: tuple[int,...] = (1,2,4,4),
            latent_dim:int=256, 
            res_blocks: int=2,
            ):
        super().__init__()
        dims = [base_dim * m for m in channel_multipliers]

        enc: list[nn.Module] = [
            nn.Conv2d(in_channels,dims[0],kernel_size=3,padding=1),
            _gn(dims[0]), nn.GELU(),
            *[GroupNormResBlock(dims[0]) for _ in range(res_blocks)],
        ]
        for i in range(len(dims)-1):
            enc += [
                nn.Conv2d(dims[i],dims[i+1], kernel_size=4,stride=2,padding=1),
                _gn(dims[i+1]), nn.GELU(),
                *[GroupNormResBlock(dims[i+1]) for _ in range(res_blocks)],
            ]
        enc.append(nn.Conv2d(dims[-1],latent_dim,kernel_size=1))
        self.encoder = nn.Sequential(*enc)

        dec: list[nn.Module] = [
            nn.Conv2d(latent_dim, dims[-1],kernel_size=1),
            _gn(dims[-1]),nn.GELU(),
        ]
        for i in range(len(dims)-1,0,-1):
            dec += [*[GroupNormResBlock(dims[i]) for _ in range(res_blocks)]]
            dec += [
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv2d(dims[i],dims[i-1],kernel_size=3,padding=1),
                _gn(dims[i-1]),nn.GELU(),
            ]
        dec += [
            *[GroupNormResBlock(dims[0]) for _ in range(res_blocks)],
            nn.Conv2d(dims[0],in_channels, kernel_size=3, padding=1),
            nn.Tanh()
        ]
        self.decoder = nn.Sequential(*dec)
    
    @jaxtyped(typechecker=beartype)
    def encode(self, images: Float[Tensor, "B C H W"]) -> Float[Tensor, "B L LH LW"]:
        return self.encoder(images)
    
    @jaxtyped(typechecker=beartype)
    def decode(self, latents: Float[Tensor, "B L LH LW"]) -> Float[Tensor, "B C H W"]:
        return self.decoder(latents)
    
    def forward(self, images: Float[Tensor, "B C H W"]) -> tuple[torch.Tensor,AutoencoderResults]:
        latents = self.encode(images)
        reconstructions = self.decode(latents)
        return reconstructions,AutoencoderResults(latents=latents)

from .hbq import HBQQuantizer

@dataclass 
class HBQAutoencoderConfig:
    in_channels: int=3
    base_dim: int=128
    channel_multipliers: tuple[int, ...] = (1,2,4,4)
    latent_dim: int=16
    quant_dim: int=16
    n_rounds: int=4
    res_blocks: int=2

class ExampleQuantizingAutoencoder(nn.Module):
    def __init__(self,conf:HBQAutoencoderConfig | None=None, **kwargs):
        super().__init__()
        if conf is None:
            conf = HBQAutoencoderConfig(**kwargs)
        self.config = conf

        self.backbone = ExampleAutoencoder(
            conf.in_channels,conf.base_dim,conf.channel_multipliers, conf.latent_dim, conf.res_blocks)

        self.quantizer = HBQQuantizer(n_rounds=conf.n_rounds)

        self.pre_quant = nn.Sequential(
            GroupNormResBlock(conf.latent_dim),
            nn.Conv2d(conf.latent_dim,conf.quant_dim,kernel_size=1),
        )
        self.post_quant = nn.Sequential(
            nn.Conv2d(conf.quant_dim, conf.latent_dim, kernel_size=1),
            GroupNormResBlock(conf.latent_dim),
        )
            
    @jaxtyped(typechecker=beartype)
    def forward(self, images: Float[Tensor,"B C H W"]) -> tuple[torch.Tensor,QuantizingAutoencoderResults]:
        latents = self.backbone.encode(images)
        z = self.pre_quant(latents)
        q_out, q_aux = self.quantizer(z)
        z = self.post_quant(q_out)
        reconstructions = self.backbone.decode(z)
        return reconstructions,QuantizingAutoencoderResults(latents=latents, quantized_latents = q_out, bit_codes = q_aux.bit_codes, tokens = q_aux.tokens)
