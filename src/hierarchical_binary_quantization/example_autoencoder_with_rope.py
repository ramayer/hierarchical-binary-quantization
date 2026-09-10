"""
Near drop-in replacement for hierarchical_binary_quantization.example_autoencoder.

Same external interface: HBQAutoencoderConfig, ExampleQuantizingAutoencoder,
AutoencoderResults, the quantizer(z) -> (q_out, q_aux) plug point, encode/decode,
set_grad_checkpointing. The conv towers (ResBlock, _StagedTower, _gn) are
unchanged from the original -- they were never the problem. What's new lives
entirely in the bottleneck:

  - RoPEAttentionBlock: global RoPE attention over the whole bottleneck grid,

  - MaskedTokenPredictor: a self-supervised auxiliary task. Masks a fraction
    of the quantized tokens and tries to recover the pre-quantization latent
    at those positions from the surrounding (still-quantized) context. This
    directly rewards the tokenizer for producing tokens that are mutually
    predictable -- the property a downstream GRN/VAR-style generator needs to
    exploit correlations instead of learning them the hard way from scratch.
    It's optional (use_mae_aux=False disables it) and its loss is exposed as
    AutoencoderResults.mae_loss() so your training loop decides the weighting.
"""

import torch
import einx
import torch.nn.functional as F
from torch import nn, Tensor
from dataclasses import dataclass
from jaxtyping import Float
from beartype import beartype
from jaxtyping import jaxtyped
from torch.utils.checkpoint import checkpoint


# ---------------------------------------------------------------------------
# 2D RoPE primitive (verified separately: relative-position invariant)
# ---------------------------------------------------------------------------

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class RoPE2D(nn.Module):
    def __init__(self, head_dim: int, base: float = 10000.0):
        super().__init__()
        assert head_dim % 4 == 0, (
            "head_dim must be divisible by 4: split in half for x/y, "
            "then each half needs to be even to form rotation pairs"
        )
        self.head_dim = head_dim
        self.half = head_dim // 2
        freqs = 1.0 / (base ** (torch.arange(0, self.half, 2).float() / self.half))
        self.register_buffer("freqs", freqs, persistent=False)

    def get_cos_sin(self, h: int, w: int, device, dtype):
        ys, xs = torch.meshgrid(
            torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij"
        )
        xs = xs.reshape(-1).float()
        ys = ys.reshape(-1).float()
        freqs = self.freqs.to(device)

        angles_x = xs[:, None] * freqs[None, :]
        angles_y = ys[:, None] * freqs[None, :]
        angles_x = torch.cat([angles_x, angles_x], dim=-1)
        angles_y = torch.cat([angles_y, angles_y], dim=-1)

        cos = torch.cat([angles_x.cos(), angles_y.cos()], dim=-1).to(dtype)
        sin = torch.cat([angles_x.sin(), angles_y.sin()], dim=-1).to(dtype)
        return cos, sin


def apply_rope2d(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    d = x.shape[-1]
    half = d // 2
    x_part, y_part = x[..., :half], x[..., half:]
    cos_x, cos_y = cos[..., :half], cos[..., half:]
    sin_x, sin_y = sin[..., :half], sin[..., half:]
    x_rot = x_part * cos_x + rotate_half(x_part) * sin_x
    y_rot = y_part * cos_y + rotate_half(y_part) * sin_y
    return torch.cat([x_rot, y_rot], dim=-1)


# ---------------------------------------------------------------------------
# Unchanged from the original example_autoencoder.py
# ---------------------------------------------------------------------------

def _gn(channels: int) -> nn.GroupNorm:
    g = 8
    while channels % g != 0 and g > 1:
        g //= 2
    return nn.GroupNorm(g, channels)


class ResBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.net = nn.Sequential(
            _gn(c), nn.SiLU(),
            nn.Conv2d(c, c, 3, padding=1, padding_mode="reflect"),
            _gn(c), nn.SiLU(),
            nn.Conv2d(c, c, 3, padding=1, padding_mode="reflect"),
        )

    def forward(self, x):
        return x + self.net(x)


class _StagedTower(nn.Module):
    def __init__(self, stage_module_lists):
        super().__init__()
        self.stages = nn.ModuleList(nn.Sequential(*s) for s in stage_module_lists)
        self.ckpt_stage = [False] * len(self.stages)
        self.already_mentioned_checkpointing = False

    def forward(self, x):
        for stage, do_ckpt in zip(self.stages, self.ckpt_stage):
            nbytes = x.numel() * x.element_size()
            if self.training and x.requires_grad and do_ckpt and nbytes > 60_000_000:
                if not self.already_mentioned_checkpointing:
                    print(f"checkpointing stages with bytes = {nbytes}")
                    self.already_mentioned_checkpointing = True
                x = checkpoint(stage, x, use_reentrant=False)
            else:
                x = stage(x)
        return x

    def set_checkpointing(self, enable: bool = True, stages=None):
        idxs = range(len(self.stages)) if stages is None else stages
        for i in idxs:
            self.ckpt_stage[i] = enable

    def stage_lengths(self):
        return [len(s) for s in self.stages]


# ---------------------------------------------------------------------------
# New: RoPE-based bottleneck attention (local windowed + global)
# ---------------------------------------------------------------------------

class RoPEMultiheadAttention(nn.Module):
    def __init__(self, dim, heads=8, out_dim=None):
        super().__init__()
        self.out_dim = out_dim if out_dim is not None else dim
        assert dim % heads == 0, f"dim={dim} not divisible by heads={heads}"
        assert self.out_dim % heads == 0, f"out_dim={self.out_dim} not divisible by heads={heads}"
        self.heads = heads
        self.head_dim = dim // heads
        self.out_head_dim = self.out_dim // heads
        assert self.head_dim % 4 == 0, (
            f"head_dim={self.head_dim} (dim={dim}/heads={heads}) must be "
            f"divisible by 4 for 2D RoPE -- pick a different head count"
        )
        self.rope = RoPE2D(self.head_dim)
        # Always separate q/k/v projections -- simpler than branching on
        # whether out_dim==dim, at the cost of not matching pre-refactor
        # checkpoints (those need strict=False and a fresh attention start).
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, self.out_dim)
        self.proj = nn.Linear(self.out_dim, self.out_dim)

    def forward(self, x: Tensor, h: int, w: int) -> Tensor:
        # x: (B, N, C), N == h*w, tokens in row-major (y, x) order.
        # Q/K always see the FULL input C (so attention decisions can use
        # protected-channel content); V/output are restricted to out_dim.
        q = einx.id("b n (heads d) -> b heads n d", self.q(x), heads=self.heads)
        k = einx.id("b n (heads d) -> b heads n d", self.k(x), heads=self.heads)
        v = einx.id("b n (heads d) -> b heads n d", self.v(x), heads=self.heads)
        cos, sin = self.rope.get_cos_sin(h, w, x.device, x.dtype)
        q = apply_rope2d(q, cos, sin)
        k = apply_rope2d(k, cos, sin)
        out = F.scaled_dot_product_attention(q, k, v)
        out = einx.id("b heads n d -> b n (heads d)", out)
        return self.proj(out)


class BottleneckAttention(nn.Module):
    """The ONE place 'what does full attention at a bottleneck look like' is
    defined. Used by both the encoder/decoder bottleneck AND the masked-token
    predictor, so they can't independently drift the way window-only vs
    global-only did before. Always global (no windowing): at bottleneck token
    counts, full attention is cheap enough that there's no reason to force a
    narrow window, and long-range binding (matching a hat's color to boots
    across the frame) is exactly the job a window would get in the way of.
    Stacked num_layers deep so the model gets compositional multi-hop reach
    (A informs B informs C), not just one hop.
    """

    def __init__(self, c, heads=8, num_layers=2, protected_channels=0):
        super().__init__()
        self.blocks = nn.Sequential(
            *[RoPEAttentionBlock(c, heads=heads, protected_channels=protected_channels)
              for _ in range(num_layers)]
        )

    def forward(self, x):
        return self.blocks(x)


class RoPEAttentionBlock(nn.Module):
    """Global RoPE attention over the whole H x W grid.

    The residual branch is scaled by a learnable `gate`, initialized to zero
    (LayerScale/ReZero-style). At step zero every block is a pure identity
    pass-through, so training starts from the already-stable conv backbone's
    behavior, with attention phasing in gradually as `gate` moves away from
    zero -- rather than injecting a full-strength, untrained attention
    transform into a deep stack from the very first step.

    protected_channels: leading channels that are a hard identity bypass.
    Attention forms Q/K from the FULL channel width (so it can still be
    informed by protected content) but its WRITE only ever touches the
    remaining (c - protected_channels) channels -- protected channels get an
    exact zero contribution from this block, by construction, guaranteeing
    global attention can never overwrite whatever lives there.
    """

    def __init__(self, c, heads=8, protected_channels=0):
        super().__init__()
        assert 0 <= protected_channels < c
        self.norm = _gn(c)
        self.protected_channels = protected_channels
        write_dim = c - protected_channels
        self.attn = RoPEMultiheadAttention(c, heads, out_dim=write_dim)
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, x: Tensor) -> Tensor:
        b, c, h, w = x.shape
        y = self.norm(x)
        pc = self.protected_channels

        y_flat = einx.id("b c h w -> b (h w) c", y)          # full C for Q/K
        y_flat = self.attn(y_flat, h, w)                      # (B, H*W, write_c)
        write_c = c - pc
        y_write = einx.id("b (h w) c -> b c h w", y_flat, h=h, w=w, c=write_c)

        if pc > 0:
            zeros = torch.zeros(b, pc, h, w, device=x.device, dtype=x.dtype)
            y_full = torch.cat([zeros, y_write], dim=1)
        else:
            y_full = y_write
        return x + self.gate * y_full


# ---------------------------------------------------------------------------
# Encoder / Decoder -- same shape as before, new bottleneck contents
# ---------------------------------------------------------------------------

class Encoder(_StagedTower):
    def __init__(self, in_channels, dims, res_blocks, latent_dim, use_attention,
                 bottleneck_heads, bottleneck_layers, protected_channel_fraction=0.0):
        stages = []
        stem = [nn.Conv2d(in_channels, dims[0], 3, padding=1, padding_mode="reflect"),
                _gn(dims[0]), nn.SiLU(),
                *[ResBlock(dims[0]) for _ in range(res_blocks[0])]]
        stages.append(stem)

        for i in range(len(dims) - 1):
            down = [nn.Conv2d(dims[i], dims[i + 1], 4, stride=2, padding=1),
                    _gn(dims[i + 1]), nn.SiLU(),
                    *[ResBlock(dims[i + 1]) for _ in range(res_blocks[i + 1])]]
            stages.append(down)

        latent_stage = []
        if use_attention:
            pc = round(dims[-1] * protected_channel_fraction)
            latent_stage.append(BottleneckAttention(dims[-1], bottleneck_heads, bottleneck_layers,
                                                      protected_channels=pc))
        latent_stage.append(nn.Conv2d(dims[-1], latent_dim, 1))
        stages.append(latent_stage)

        super().__init__(stages)


class Decoder(_StagedTower):
    def __init__(self, in_channels, dims, res_blocks, latent_dim, use_attention,
                 bottleneck_heads, bottleneck_layers, protected_channel_fraction=0.0):
        stages = []
        latent_stage = [nn.Conv2d(latent_dim, dims[-1], 1), _gn(dims[-1]), nn.SiLU()]
        if use_attention:
            pc = round(dims[-1] * protected_channel_fraction)
            latent_stage.append(BottleneckAttention(dims[-1], bottleneck_heads, bottleneck_layers,
                                                      protected_channels=pc))
        stages.append(latent_stage)

        for i in range(len(dims) - 1, 0, -1):
            up = [*[ResBlock(dims[i]) for _ in range(res_blocks[i])],
                  nn.Upsample(scale_factor=2, mode="nearest"),
                  nn.Conv2d(dims[i], dims[i - 1], 3, padding=1, padding_mode="reflect"),
                  _gn(dims[i - 1]), nn.SiLU()]
            stages.append(up)

        tail = [*[ResBlock(dims[0]) for _ in range(res_blocks[0])],
                nn.Conv2d(dims[0], in_channels, 3, padding=1, padding_mode="reflect"),
                nn.Tanh()]
        stages.append(tail)

        super().__init__(stages)


@dataclass
class AutoencoderResults:
    latents: Tensor
    quant_info: object = None
    q_out: "Tensor | None" = None
    mae_pred: "Tensor | None" = None
    mae_target: "Tensor | None" = None
    mae_mask: "Tensor | None" = None

    def mae_loss(self):
        if self.mae_pred is None:
            return None
        mask = self.mae_mask.expand_as(self.mae_target)
        denom = mask.sum().clamp_min(1.0)
        return ((self.mae_pred - self.mae_target) ** 2 * mask).sum() / denom


class ExampleAutoencoder(nn.Module):
    def __init__(
        self,
        in_channels=3,
        base_dim=96,
        channel_multipliers=(1, 2, 3, 4),
        latent_dim=32,
        res_blocks=(1, 2, 3, 3),
        use_attention=True,
        bottleneck_heads=8,
        bottleneck_layers=2,
        protected_channel_fraction=0.0,
    ):
        super().__init__()
        dims = [base_dim * m for m in channel_multipliers]
        if isinstance(res_blocks, int):
            res_blocks = (res_blocks,) * len(dims)
        self.encoder = Encoder(in_channels, dims, res_blocks, latent_dim, use_attention,
                                bottleneck_heads, bottleneck_layers, protected_channel_fraction)
        self.decoder = Decoder(in_channels, dims, res_blocks, latent_dim, use_attention,
                                bottleneck_heads, bottleneck_layers, protected_channel_fraction)

    def set_grad_checkpointing(self, enable: bool = True):
        self.encoder.set_checkpointing(enable)
        self.decoder.set_checkpointing(enable)

    @jaxtyped(typechecker=beartype)
    def encode(self, images):
        return self.encoder(images)

    @jaxtyped(typechecker=beartype)
    def decode(self, latents):
        return self.decoder(latents)

    def forward(self, images):
        latents = self.encode(images)
        recons = self.decode(latents)
        return recons, AutoencoderResults(latents=latents)


# ---------------------------------------------------------------------------
# New: masked-token auxiliary predictor
# ---------------------------------------------------------------------------

class MaskedTokenPredictor(nn.Module):
    """Same BottleneckAttention as the main model's bottleneck -- deliberately
    NOT its own independently-configured attention stack. The earlier
    windowed-only version had no way to reconcile predictions across window
    boundaries, which is exactly why it checkerboarded at the window grid."""

    def __init__(self, quant_dim, depth=2, heads=2, mask_ratio=0.5):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.mask_token = nn.Parameter(torch.zeros(1, quant_dim, 1, 1))
        nn.init.normal_(self.mask_token, std=0.02)
        self.blocks = BottleneckAttention(quant_dim, heads=heads, num_layers=depth)
        self.predict = nn.Conv2d(quant_dim, quant_dim, kernel_size=1)

    def forward(self, q_out: Tensor, target: Tensor):
        b, c, h, w = q_out.shape
        mask = (torch.rand(b, 1, h, w, device=q_out.device) < self.mask_ratio).float()
        x = q_out * (1 - mask) + self.mask_token * mask
        x = self.blocks(x)
        pred = self.predict(x)
        return pred, target, mask


# ---------------------------------------------------------------------------
# Same config name/fields as the original, plus new (defaulted) fields
# ---------------------------------------------------------------------------

@dataclass
class HBQAutoencoderConfig:
    in_channels: int = 3
    base_dim: int = 96
    channel_multipliers: "tuple[int, ...]" = (1, 2, 3, 4)
    res_blocks: "tuple[int, ...]" = (1, 2, 3, 3)
    latent_dim: int = 32
    quant_dim: int = 8
    n_rounds: int = 4
    # new, all defaulted so existing call sites keep working unchanged
    bottleneck_heads: int = 8
    bottleneck_layers: int = 2         # stacked global attention layers at the bottleneck
    protected_channel_fraction: float = 0.0   # 0.0 = old behavior, fully unaffected
    use_mae_aux: bool = True
    mae_mask_ratio: float = 0.5
    mae_heads: int = 2
    mae_depth: int = 2


class ExampleQuantizingAutoencoderWithRope(nn.Module):
    def __init__(self, conf=None, quantizer=None, **kwargs):
        super().__init__()
        if conf is None:
            conf = HBQAutoencoderConfig(**kwargs)
        if isinstance(conf, dict):
            conf = HBQAutoencoderConfig(**conf)
        self.config = conf

        self.backbone = ExampleAutoencoder(
            in_channels=conf.in_channels,
            base_dim=conf.base_dim,
            channel_multipliers=conf.channel_multipliers,
            latent_dim=conf.latent_dim,
            res_blocks=conf.res_blocks,
            use_attention=True,
            bottleneck_heads=conf.bottleneck_heads,
            bottleneck_layers=conf.bottleneck_layers,
            protected_channel_fraction=conf.protected_channel_fraction,
        )

        if quantizer is not None:
            self.quantizer = quantizer
        else:
            from .hbq import HBQQuantizer  # matches original's import location
            self.quantizer = HBQQuantizer(n_rounds=conf.n_rounds)

        self.pre_quant = nn.Sequential(
            ResBlock(conf.latent_dim),
            nn.Conv2d(conf.latent_dim, conf.quant_dim, kernel_size=1),
            nn.Tanh(),
        )
        self.post_quant = nn.Sequential(
            nn.Conv2d(conf.quant_dim, conf.latent_dim, kernel_size=1),
            ResBlock(conf.latent_dim),
        )

        self.mae = (
            MaskedTokenPredictor(
                conf.quant_dim,
                depth=conf.mae_depth,
                heads=conf.mae_heads,
                mask_ratio=conf.mae_mask_ratio,
            )
            if conf.use_mae_aux else None
        )

    def set_grad_checkpointing(self, enable: bool = True):
        self.backbone.set_grad_checkpointing(enable)

    def forward(self, images: Float[Tensor, "B C H W"]):
        latents = self.backbone.encode(images)
        z = self.pre_quant(latents)
        q_out, q_aux = self.quantizer(z)

        if self.mae is not None:
            mae_pred, mae_target, mae_mask = self.mae(q_out, z.detach())
        else:
            mae_pred = mae_target = mae_mask = None

        z_dec = self.post_quant(q_out)
        reconstructions = self.backbone.decode(z_dec)
        return reconstructions, AutoencoderResults(
            latents=latents, quant_info=q_aux, q_out=q_out,
            mae_pred=mae_pred, mae_target=mae_target, mae_mask=mae_mask,
        )


    @jaxtyped(typechecker=beartype)
    def encode(self, images):
        return self.backbone.encoder(images)

    @jaxtyped(typechecker=beartype)
    def decode(self, latents):
        return self.backbone.decoder(latents)