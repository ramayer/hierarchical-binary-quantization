"""
    Thx, claude for writing an example quantizing autoencoder
"""
import torch
from torch import nn

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

class SpatialSelfAttention(nn.Module):
    def __init__(self, channels, num_heads: int=4):
        super().__init__()
        self.norm = _gn(channels)
        actual_heads = max(h for h in range(1,num_heads+1) if channels % h ==0)
        self.attn = nn.MultiheadAttention(channels,actual_heads,batch_first=True)
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        B,C,H,W = x.shape
        normed = self.norm(x)
        tokens = normed.reshape(B,C,H*W).permute(0,2,1) # B, HW, C
        out,_ = self.attn(tokens,tokens,tokens, need_weights = False)
        out = out.permute(0,2,1).reshape(B,C,H,W)
        return x + out

class ExampleAutoencoder(nn.Module):
    def __init__(self,in_channels: int=3, latent_dim: int=256, base_dim:int=64):
        super().__init__()
        d2 = base_dim * 2
        self.enc_stem = nn.Sequential(
            nn.Conv2d(in_channels, base_dim, kernel_size=3, padding=1),
            _gn(base_dim),
            nn.GELU(),
            GroupNormResBlock(base_dim),
        )
        self.enc_down1 = nn.Sequential(
            nn.Conv2d(base_dim, d2, kernel_size=4, stride=2, padding=1),
            _gn(d2),
            nn.GELU(),
            GroupNormResBlock(d2),
            GroupNormResBlock(d2),                        
        )
        self.enc_down2 = nn.Sequential(
            nn.Conv2d(d2, latent_dim, kernel_size=4, stride=2, padding=1),
            _gn(latent_dim),
            nn.GELU(),
            GroupNormResBlock(latent_dim),
            GroupNormResBlock(latent_dim),                        
        )
        self.enc_bottleneck=SpatialSelfAttention(latent_dim,num_heads=4)
        # Quantizer goes here
        self.dec_bottleneck=SpatialSelfAttention(latent_dim,num_heads=4)
        self.dec_up1 = nn.Sequential(
            GroupNormResBlock(latent_dim),
            GroupNormResBlock(latent_dim),
            GroupNormResBlock(latent_dim),
            nn.Upsample(scale_factor=2,mode="nearest"),
            nn.Conv2d(latent_dim,d2,kernel_size=3, padding=1),
            _gn(d2),
            nn.GELU(),
        )
        self.dec_up2 = nn.Sequential(
            GroupNormResBlock(d2),
            GroupNormResBlock(d2),
            GroupNormResBlock(d2),
            nn.Upsample(scale_factor=2,mode="nearest"),
            nn.Conv2d(d2,base_dim,kernel_size=3, padding=1),
            _gn(base_dim),
            nn.GELU(),
        )
        self.dec_out = nn.Sequential(
            GroupNormResBlock(base_dim),
            GroupNormResBlock(base_dim),
            nn.Conv2d(base_dim, in_channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
    def encode(self, images: torch.Tensor) -> torch.Tensor:
        x = self.enc_stem(images)
        x = self.enc_down1(x)
        x = self.enc_down2(x)
        x = self.enc_bottleneck(x)
        return x
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        x = self.dec_bottleneck(latents)
        x = self.dec_up1(x)
        x = self.dec_up2(x)
        x = self.dec_out(x)
        return x
    def forward(self, images: torch.tensor) -> tuple[torch.Tensor,torch.Tensor]:
        latents = self.encode(images)
        reconstructions = self.decode(latents)
        return reconstructions,latents

from .hbq import HBQQuantizer
class ExampleQuantizingAutoencoder(nn.Module):
    def __init__(self,in_channels: int=3, latent_dim: int=256, base_dim:int=64):
        super().__init__()
        self.backbone = ExampleAutoencoder(in_channels,latent_dim,base_dim)
        self.quantizer = HBQQuantizer(n_rounds=1)
    def forward(self, images: torch.tensor) -> tuple[torch.Tensor,torch.Tensor]:
        latents = self.backbone.encode(images)
        q_out, token_ids = self.quantizer(latents)
        reconstructions = self.backbone.decode(q_out)
        return reconstructions,latents, q_out, token_ids

