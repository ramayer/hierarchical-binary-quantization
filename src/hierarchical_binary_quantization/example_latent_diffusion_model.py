import einx
import datetime
import IPython.display as ipd
import math
import numpy as np
import time
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from datetime import timedelta

import cs336_basics.ron_adamw_optimizer as cs336_adamw_optimizer
import cs336_basics.ron_bpe_tokenizer as cs336__bpe_tokenizer
import cs336_basics.ron_causal_multihead_self_attention_with_rope as cs336_causal_multihead_self_attention_with_rope
import cs336_basics.ron_cross_entropy as cs336_cross_entropy
import cs336_basics.ron_data_loader as cs336_data_loader
import cs336_basics.ron_embedding as cs336_embedding
import cs336_basics.ron_linear as cs336_linear
import cs336_basics.ron_multihead_self_attention as cs336_multihead_self_attention
import cs336_basics.ron_rmsnorm as cs336_rmsnorm
import cs336_basics.ron_rope as cs336_rope
import cs336_basics.ron_scaled_dot_product_attention as cs336_scaled_dot_product_attention
import cs336_basics.ron_softmax as cs336_softmax
import cs336_basics.ron_swiglu as cs336_swiglu
import cs336_basics.ron_train_bpe as cs336_train_bpe
import cs336_basics.ron_transformer_lm as cs336_transformer_lm


class RoPE2D(torch.nn.Module):
    """
     Example usage of 2D RoPE (Rotary Positional Embeddings) for a grid of tokens.
     Rope2D.  Like CS336's 1D one, but in 2D.
    """
    def __init__(self, theta: float, d_k: int, *, max_seq_len: int=128, device=None):
        super().__init__()
        assert d_k % 4 == 0, "need d_k/2 even, since each half gets your original RoPE"
        self.rope_row = cs336_rope.RoPE(theta=theta, d_k=d_k // 2, max_seq_len=max_seq_len, device=device)
        self.rope_col = cs336_rope.RoPE(theta=theta, d_k=d_k // 2, max_seq_len=max_seq_len, device=device)

    def forward(self, x, rows, cols):
        x_row_half, x_col_half = einx.id("... (two half) -> two ... half",x, two=2)
        x_row_half = self.rope_row(x_row_half, token_positions=rows)
        x_col_half = self.rope_col(x_col_half, token_positions=cols)
        return einx.id("... half, ... half -> ... (half + half)", x_row_half, x_col_half)


def get_2d_positions(height: int, width: int, device=None):
    """
    For a HxW grid, returns (rows, cols) — each shape(h*w,),
    giving every flattened token's row and column.
    Token order matches how we'll flatten the latent 
    tensor: x.flatten(2) on a (C, H, W) tensor.

    Identical to: 
        rows = einx.id("h -> (h w)", torch.arange(height), w = width)
        cols = einx.id("w -> (h w)", torch.arange(width), h = height)
    but when compiled the einx expressions give
        TorchRuntimeError: Dynamo failed to run FX node ... For even more developer > context, set TORCH_LOGS="+dynamo"
    and 
        See: https://claude.ai/share/fdb18558-b963-4324-b728-bc911a358503
    """
    rows = torch.arange(height, device=device).repeat_interleave(width)  # (h w), h outer
    cols = torch.arange(width, device=device).repeat(height)             # (h w), h outer
    return rows, cols








class SpatialMultiheadSelfAttention(torch.nn.Module):
    """
    2D MultiheadSelfAttention. Like CS336's 1D one, but 2D

    Added qk_norm (QK Norm stablizes attention in many models (ViT-22B, Gemma, and JiT)).
    """
    def __init__(self, d_model, num_heads, theta, qk_norm = True):
        super().__init__()
        dk = dv = d_model // num_heads
        self.num_heads = num_heads
        self.q_proj = nn.Linear(d_model, num_heads * dk)
        self.k_proj = nn.Linear(d_model, num_heads * dk)
        self.v_proj = nn.Linear(d_model, num_heads * dv)
        self.o_proj = nn.Linear(num_heads * dv, d_model)

        self.q_norm = cs336_rmsnorm.RMSNorm(dk) if qk_norm else nn.Identity()
        self.k_norm = cs336_rmsnorm.RMSNorm(dk) if qk_norm else nn.Identity()

        self.rope2d = RoPE2D(d_k=dk, theta=theta)

    def forward(self, x, rows, cols):
        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        q = einx.id("... seq_len (heads d) -> ... heads seq_len d", q, heads=self.num_heads)
        k = einx.id("... seq_len (heads d) -> ... heads seq_len d", k, heads=self.num_heads)
        v = einx.id("... seq_len (heads d) -> ... heads seq_len d", v, heads=self.num_heads)

        # broadcast rows/cols across the heads dim, same trick as the original's token_positions
        heads_rows = einx.id("... s -> ... 1 s", rows)
        heads_cols = einx.id("... s -> ... 1 s", cols)

        q = self.q_norm(q)
        k = self.k_norm(k)

        q = self.rope2d(q, rows=heads_rows, cols=heads_cols)
        k = self.rope2d(k, rows=heads_rows, cols=heads_cols)

        # no mask at all -- every latent-grid token attends to every other token
        #attn_output = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None)
        attn_output = cs336_scaled_dot_product_attention.scaled_dot_product_attention(k=k, q=q, v=v, mask=None)
        attn_output = einx.id("... heads seq d_v -> ... seq (heads d_v)", attn_output)
        return self.o_proj(attn_output)


class SpatialTransformerBlock(torch.nn.Module):
    """
    Like cs336_transformer_block, but uses SpatialMultiheadSelfAttention instead of standard MultiheadSelfAttention.
    """
    def __init__(self, d_model, num_heads, d_ff, theta=150):
        super().__init__()
        self.norm1 = cs336_rmsnorm.RMSNorm(d_model)
        self.attn = SpatialMultiheadSelfAttention(d_model=d_model, num_heads=num_heads,
                                                   theta=theta)
        self.norm2 = cs336_rmsnorm.RMSNorm(d_model)
        self.ffn = cs336_swiglu.SwiGLU(d_model, d_ff)

    def forward(self, x, rows, cols):
        x = x + self.attn(self.norm1(x), rows, cols)
        x = x + self.ffn(self.norm2(x))
        return x

def timestep_embedding(t, dim, theta=10000):
    """Standard sinusoidal embedding, same idea as positional encodings, applied to a scalar 't' instead of a sequence index."""
    half = dim // 2
    freqs = torch.exp(-math.log(theta) * torch.arange(half, device=t.device) / half)
    args = t[:, None].float() * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)  # (batch, dim)


# AdaLN


def modulate(x, shift, scale):
    """
    x: (B, seq_len, d_model)
    shift, scale: (B, d_model) -- one value per channel, per batch item, from t
    """
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class AdaLNSpatialTransformerBlock(torch.nn.Module):
    """
    AdaLN is a standard way of doing time conditioning on diffusion models, 
    but build this one out of CS336's SwiGLU and RMSNorm blocks.
    """
    def __init__(self, d_model, num_heads, d_ff, theta):
        super().__init__()
        self.norm1 = cs336_rmsnorm.RMSNorm(d_model)
        self.attn = SpatialMultiheadSelfAttention(d_model=d_model, num_heads=num_heads,
                                                   theta=theta)
        self.norm2 = cs336_rmsnorm.RMSNorm(d_model)
        self.ffn = cs336_swiglu.SwiGLU(d_model, d_ff)

        self.adaLN_modulation = torch.nn.Linear(d_model, 6 * d_model)
        torch.nn.init.zeros_(self.adaLN_modulation.weight)
        torch.nn.init.zeros_(self.adaLN_modulation.bias)

    def forward(self, x, rows, cols, t_emb):
        shift_attn, scale_attn, gate_attn, shift_ffn, scale_ffn, gate_ffn = \
            self.adaLN_modulation(t_emb).chunk(6, dim=-1)

        attn_in = modulate(self.norm1(x), shift_attn, scale_attn)
        x = x + gate_attn.unsqueeze(1) * self.attn(attn_in, rows, cols)

        ffn_in = modulate(self.norm2(x), shift_ffn, scale_ffn)
        x = x + gate_ffn.unsqueeze(1) * self.ffn(ffn_in)

        return x


###########################################################################################
# Minimal Spatial ViT Implementation

class MinimalSpatialViT(torch.nn.Module):
    """

    ### Minimal Vision Transformer using CS336 blocks.

    Just like the LLM, except:

    * Input tokens come from a quantizing autoencoder's encoder that preprocessed the images instead of BPE.
    * Outputs are tokens for the quantizing autoencoder's decoder instead of BPE tokens.

    Using the Torch Linear layer with a bias makes it train noticibly
    faster than the cs336 one without a bias; though they seem to converge
    on a similar losses eventually.
    """
    def __init__(self, latent_dim=8, d_model=384, num_heads=6,
                 d_ff=1024, depth=6, theta=150.0):
        super().__init__()
        self.latent_dim = latent_dim
        self.d_model = d_model
        #self.x_embedder = cs336_linear.Linear(latent_dim, d_model)
        self.x_embedder = torch.nn.Linear(latent_dim, d_model) # maybe faster with bias
        self.blocks = torch.nn.ModuleList([
            AdaLNSpatialTransformerBlock(d_model, num_heads, d_ff, theta)
            for _ in range(depth)
        ])
        self.final_norm = cs336_rmsnorm.RMSNorm(d_model)
        #self.final_linear = cs336_linear.Linear(d_model, latent_dim)
        self.final_linear = torch.nn.Linear(d_model, latent_dim) # maybe faster with bias
        #rows, cols = get_2d_positions(grid_size)
        #self.register_buffer("rows", rows)
        #self.register_buffer("cols", cols)
        self.config = {
            "latent_dim":latent_dim,
            "d_model":d_model,
            "num_heads":num_heads,
            "d_ff":d_ff,
            "depth":depth,
            "theta":theta,
        }

    def forward(self, x, t, y=None):
        B, C, H, W = x.shape
        rows,cols = get_2d_positions(height=H,width=W)
        tokens = x.flatten(2).transpose(1, 2)      # (B, H*W, latent_dim)
        tokens = self.x_embedder(tokens)             # (B, H*W, d_model)
        t_emb = timestep_embedding(t, self.d_model)   # (B, d_model)
        for block in self.blocks:
            tokens = block(tokens, rows, cols, t_emb)
        tokens = self.final_linear(self.final_norm(tokens))
        return tokens.transpose(1, 2).reshape(B, C, H, W)

######################################################################





class ExampleDiffusionModelTests:

    @staticmethod
    def smoke_test_rope():
        d_k = 4
        one_d_rope = cs336_rope.RoPE(theta=150,d_k=d_k,max_seq_len=1024)(torch.ones(4),token_positions=None)
        print("1D rope",one_d_rope)
        h,w = 2,3
        rows, cols = get_2d_positions(h,w)
        print("rows ",rows)
        print("cols ",cols)
        x = torch.ones(h*w,d_k)
        two_d_rope = RoPE2D(theta=150, d_k = d_k)(x, rows, cols)
        print("2D rope", two_d_rope)
        print(f"{'token':>5} {'(r, c)':>10} {"row_half"!s:<20} col_half")
        for t in range(h*w):
            r,c = rows[t].item(), cols[t].item()
            row_half = [f"{round(v,3):<6}" for v in two_d_rope[t, : d_k//2].tolist()]
            col_half = [f"{round(v,3):<6}" for v in two_d_rope[t, d_k//2 :].tolist()]
            print(f"{t:>5} {(r,c)!s:>10} {row_half!s:<20} {col_half}")
        assert two_d_rope.shape == (h*w, d_k)
        print("Smoke test passed.")

    @staticmethod
    def smoke_test_SpatialMultiheadSelfAttention():
        w,h=24,32
        rows, cols = get_2d_positions(width=w,height=h)
        smsa = SpatialMultiheadSelfAttention(d_model=384, num_heads=4, theta=1000)
        print(smsa.forward(torch.randn(2, w*h, 384), rows, cols)[0:1,0:4,0:4])
        assert smsa.forward(torch.randn(2, w*h, 384), rows, cols).shape == (2, w*h, 384)

    @staticmethod
    def smoke_test_AdaLNSpatialTransformerBlock():
        d_model=100
        num_heads=5
        d_ff=17
        theta=1111
        grid_height = 4
        grid_width = 5
        batch_size = 2

        alnstb = AdaLNSpatialTransformerBlock(d_model, num_heads, d_ff, theta)
        rows, cols = get_2d_positions(grid_width,grid_height)
        x = torch.rand(batch_size,grid_width*grid_height, d_model)
        t = torch.rand(batch_size)
        t_emb = timestep_embedding(t, d_model)  
        out = alnstb.forward(x,rows,cols, t_emb)
        print(torch.allclose(out, x))        # should be True -- zero-init means no-op at init
        assert out.shape == x.shape
        print("Smoke test passed.")

    @staticmethod
    def smoke_test_MinimalSpatialViT():
        BS = 3
        H=16
        W=12
        lj = MinimalSpatialViT()
        total_params = sum(p.numel() for p in lj.parameters())
        print(f"Total parameters: {total_params:,}")
        x = torch.randn(BS, 8, H,W)
        t = torch.zeros(BS, dtype=torch.long)
        labels = torch.zeros(BS, dtype=torch.long)
        print("output: ",lj.forward(x,t,labels).shape)
        assert lj.forward(x,t,labels).shape == (BS, 8, H, W)

    