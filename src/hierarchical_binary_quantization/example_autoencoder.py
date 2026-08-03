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
    quant_info: object = None # Type depends on the optional encoder.

def _gn(channels:int)->nn.GroupNorm:
    g=8
    while channels % g != 0 and g>1:
        g//=2
    return nn.GroupNorm(g, channels)

class ResBlock(nn.Module):
    def __init__(self,c):
        super().__init__()
        self.net=nn.Sequential(
            _gn(c), nn.SiLU(),
            nn.Conv2d(c,c,3,padding=1,padding_mode="reflect"),
            _gn(c), nn.SiLU(),
            nn.Conv2d(c,c,3,padding=1,padding_mode="reflect"),
        )
    def forward(self,x):
        return x+self.net(x)

class AttentionBlock(nn.Module):
    def __init__(self,c,heads=8):
        super().__init__()
        self.norm=_gn(c)
        self.attn=nn.MultiheadAttention(c, heads, batch_first=True)
    def forward(self,x):
        b,c,h,w=x.shape
        y=self.norm(x).flatten(2).transpose(1,2)
        y,_=self.attn(y,y,y,need_weights=False)
        y=y.transpose(1,2).reshape(b,c,h,w)
        return x+y

class ExampleAutoencoder(nn.Module):
    def __init__(
        self,
        in_channels=3,
        base_dim=96,
        channel_multipliers=(1,2,4,6,6),
        latent_dim=32,
        res_blocks=(1,2,3,4,4),
        use_attention=True,
    ):
        super().__init__()
        dims=[base_dim*m for m in channel_multipliers]
        if isinstance(res_blocks,int):
            res_blocks=(res_blocks,)*len(dims)

        enc=[
            nn.Conv2d(in_channels,dims[0],3,padding=1,padding_mode="reflect"),
            _gn(dims[0]), nn.SiLU(),
            *[ResBlock(dims[0]) for _ in range(res_blocks[0])]
        ]
        for i in range(len(dims)-1):
            enc += [
                nn.Conv2d(dims[i],dims[i+1],4,stride=2,padding=1),
                _gn(dims[i+1]), nn.SiLU(),
                *[ResBlock(dims[i+1]) for _ in range(res_blocks[i+1])]
            ]
        if use_attention:
            enc.append(AttentionBlock(dims[-1]))
        enc.append(nn.Conv2d(dims[-1],latent_dim,1))
        self.encoder=nn.Sequential(*enc)

        dec=[
            nn.Conv2d(latent_dim,dims[-1],1),
            _gn(dims[-1]), nn.SiLU()
        ]
        if use_attention:
            dec.append(AttentionBlock(dims[-1]))
        for i in range(len(dims)-1,0,-1):
            dec += (
                [ResBlock(dims[i]) for _ in range(res_blocks[i])] +
                [
                    nn.Upsample(scale_factor=2,mode="nearest"),
                    nn.Conv2d(dims[i],dims[i-1],3,padding=1,padding_mode="reflect"),
                    _gn(dims[i-1]), nn.SiLU()
                ]
            )
        dec += [ResBlock(dims[0]) for _ in range(res_blocks[0])]
        dec += [
            nn.Conv2d(dims[0],in_channels,3,padding=1,padding_mode="reflect"),
            nn.Tanh()
        ]
        self.decoder=nn.Sequential(*dec)

    @jaxtyped(typechecker=beartype)
    def encode(self, images):
        return self.encoder(images)

    @jaxtyped(typechecker=beartype)
    def decode(self, latents):
        return self.decoder(latents)

    def forward(self, images):
        latents=self.encode(images)
        recons=self.decode(latents)
        return recons, AutoencoderResults(latents=latents)

from .hbq import HBQQuantizer

"""
        in_channels=3,
        base_dim=96,
        channel_multipliers=(1,2,4,6,6),
        latent_dim=32,
        res_blocks=(1,2,3,4,4),
        use_attention=True,
"""
@dataclass 
class HBQAutoencoderConfig:
    in_channels: int=3
    base_dim: int=96
    channel_multipliers: tuple[int, ...] = (1,2,4,6,6)
    res_blocks: tuple[int, ...] = (1,2,3,4,4)
    latent_dim: int=32
    quant_dim: int=16
    n_rounds: int=4

class ExampleQuantizingAutoencoder(nn.Module):
    def __init__(self,
                 conf:HBQAutoencoderConfig | dict | None = None, 
                 quantizer:nn.Module | None = None,
                 **kwargs):
        super().__init__()
        
        if conf is None:
            conf = HBQAutoencoderConfig(**kwargs)
        self.config = conf

        self.backbone = ExampleAutoencoder(
            in_channels = conf.in_channels,
            base_dim = conf.base_dim,
            channel_multipliers = conf.channel_multipliers,
            latent_dim = conf.latent_dim, 
            res_blocks = conf.res_blocks,
            use_attention = True,
        )

        self.quantizer = quantizer or HBQQuantizer(n_rounds=conf.n_rounds)

        self.pre_quant = nn.Sequential(
            ResBlock(conf.latent_dim),
            nn.Conv2d(conf.latent_dim,conf.quant_dim,kernel_size=1),
            nn.Tanh(),
        )
        self.post_quant = nn.Sequential(
            nn.Conv2d(conf.quant_dim, conf.latent_dim, kernel_size=1),
            ResBlock(conf.latent_dim),
        )
            
    @jaxtyped(typechecker=beartype)
    def forward(self, images: Float[Tensor,"B C H W"]) -> tuple[torch.Tensor,AutoencoderResults]:
        latents = self.backbone.encode(images)
        z = self.pre_quant(latents)
        q_out, q_aux = self.quantizer(z)
        z = self.post_quant(q_out)
        reconstructions = self.backbone.decode(z)
        return reconstructions,AutoencoderResults(latents=latents, quant_info=q_aux)
