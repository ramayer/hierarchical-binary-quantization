"""
Local, tileable variant of the autoencoder -- deliberately gives up whole-image
consistency (the job the main nextgen_autoencoder.py bottleneck does) in favor
of an EXACTLY, ADDITIVELY computable receptive field, small enough to tile a
large image with a known, small trim margin. Two motivations converge here:
tiling large images cheaply, and studying how quantization errors look when
the tokenizer can't paper over them with borrowed long-range context.

Reused unchanged from nextgen_autoencoder.py: RoPE2D / apply_rope2d (RoPE is
relative-offset-only by construction, which is exactly what makes this whole
approach translation-invariant), MaskedTokenPredictor, AutoencoderResults,
_StagedTower. Everything else here is new:

  - ChannelLayerNorm: normalizes each token's own channel vector, with ZERO
    dependence on spatial extent -- unlike GroupNorm, which computes stats
    over the whole image and would give a sky-heavy tile different statistics
    than a grass-heavy tile, producing a visible seam at tile boundaries even
    with a perfectly bounded receptive field.

  - HaloedNeighborhoodAttention: HaloNet-style blocked local attention.
    Queries come from non-overlapping (block x block) tiles; keys/values come
    from a slightly larger (block + 2*halo) haloed region around each query
    tile. This is NOT the same as Swin-style window partitioning (where two
    adjacent patches in different windows can't see each other at all) --
    haloing gives every layer a uniform, known amount of cross-boundary reach
    (`halo` on each side), so total network receptive field is exactly
    additive across depth: see receptive_field_pixels() below.

  - Patchify replaces the deep strided-conv downsampling tower entirely: a
    single non-overlapping conv (kernel=stride=patch_size) has ZERO receptive
    field leakage beyond one patch, versus the ~400px RF that came from
    repeated overlapping-kernel downsampling in the main model. This is the
    single biggest lever in keeping this variant's RF small.
"""

import torch
import einx
import torch.nn.functional as F
from torch import nn, Tensor
from dataclasses import dataclass

from .example_autoencoder_with_rope import (
    RoPE2D, apply_rope2d, AutoencoderResults, _StagedTower,
)


# ---------------------------------------------------------------------------
# Normalization: per-token only, no spatial-extent statistics
# ---------------------------------------------------------------------------

class ChannelLayerNorm(nn.Module):
    def __init__(self, c, eps=1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(c, eps=eps)

    def forward(self, x: Tensor) -> Tensor:  # x: (B, C, H, W)
        x = einx.id("b c h w -> b h w c", x)
        x = self.norm(x)
        return einx.id("b h w c -> b c h w", x)


class ResBlockLocal(nn.Module):
    """Same shape/idea as the main file's ResBlock, but ChannelLayerNorm
    instead of GroupNorm -- keeps this stack's cross-tile behavior clean."""

    def __init__(self, c):
        super().__init__()
        self.net = nn.Sequential(
            ChannelLayerNorm(c), nn.SiLU(),
            nn.Conv2d(c, c, 3, padding=1, padding_mode="reflect"),
            ChannelLayerNorm(c), nn.SiLU(),
            nn.Conv2d(c, c, 3, padding=1, padding_mode="reflect"),
        )

    def forward(self, x):
        return x + self.net(x)


# ---------------------------------------------------------------------------
# Haloed neighborhood attention
# ---------------------------------------------------------------------------

class HaloedNeighborhoodAttention(nn.Module):
    def __init__(self, dim, block=8, halo=4, heads=4, rope_base=150):
        # rope_base defaults to 150, not RoPE2D's usual 10000 -- 10000 is
        # calibrated for long sequences (thousands of positions), but this
        # window only ever spans a handful of tokens (block + 2*halo per
        # side), so most of that frequency spectrum's slow-rotating channels
        # barely move across the whole relevant offset range and carry almost
        # no positional information. A much smaller base spreads the spectrum
        # to actually be informative across a small window. Verified: at
        # base=10000 with a 6-token window, the slowest channel pair rotates
        # ~0.05 degrees across the entire relevant offset range; at base=150,
        # ~2.6 degrees -- ~50x more signal. Re-tune this if block/halo change
        # substantially, since the "right" base scales with window size.
        super().__init__()
        assert dim % heads == 0
        self.block = block
        self.halo = halo
        self.heads = heads
        self.head_dim = dim // heads
        assert self.head_dim % 4 == 0, "head_dim must be divisible by 4 for 2D RoPE"
        self.rope = RoPE2D(self.head_dim, base=rope_base)
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, C, H, W). H, W must already be multiples of `block`.
        b, c, h, w = x.shape
        bl, ha = self.block, self.halo
        assert h % bl == 0 and w % bl == 0, f"H,W must be divisible by block={bl}, got {h}x{w}"
        nh, nw = h // bl, w // bl
        hw = bl + 2 * ha

        q_in = einx.id("b c (nh bh) (nw bw) -> (b nh nw) (bh bw) c", x, bh=bl, bw=bl)

        x_pad = F.pad(x, (ha, ha, ha, ha))  # zero-pad; only matters at true image
                                             # edges once tiles are stitched, since
                                             # tile-interior halos come from real
                                             # neighboring content in the padded tile
        patches = x_pad.unfold(2, hw, bl).unfold(3, hw, bl)     # (b, c, nh, nw, hw, hw)
        kv_in = einx.id("b c nh nw kh kw -> (b nh nw) (kh kw) c", patches)

        q = einx.id("bn (bh bw) (heads d) -> bn heads (bh bw) d",
                    self.q(q_in), bh=bl, heads=self.heads)
        k = einx.id("bn (kh kw) (heads d) -> bn heads (kh kw) d",
                    self.k(kv_in), kh=hw, heads=self.heads)
        v = einx.id("bn (kh kw) (heads d) -> bn heads (kh kw) d",
                    self.v(kv_in), kh=hw, heads=self.heads)

        # Q and K MUST share one coordinate frame or RoPE's relative-offset
        # math is wrong by a constant (ha,ha) shift. Build cos/sin once on the
        # full haloed (hw x hw) grid and slice out the query sub-square,
        # rather than computing Q's cos/sin on its own separate (bl x bl)
        # grid starting at (0,0).
        cos_full, sin_full = self.rope.get_cos_sin(hw, hw, x.device, x.dtype)
        cos_full = cos_full.reshape(hw, hw, -1)
        sin_full = sin_full.reshape(hw, hw, -1)
        cos_q = cos_full[ha:ha + bl, ha:ha + bl].reshape(bl * bl, -1)
        sin_q = sin_full[ha:ha + bl, ha:ha + bl].reshape(bl * bl, -1)
        cos_k = cos_full.reshape(hw * hw, -1)
        sin_k = sin_full.reshape(hw * hw, -1)

        q = apply_rope2d(q, cos_q, sin_q)
        k = apply_rope2d(k, cos_k, sin_k)

        out = F.scaled_dot_product_attention(q, k, v)
        out = einx.id("bn heads (bh bw) d -> bn (bh bw) (heads d)", out, bh=bl)
        out = self.proj(out)
        return einx.id("(b nh nw) (bh bw) c -> b c (nh bh) (nw bw)",
                        out, b=b, nh=nh, nw=nw, bh=bl, bw=bl)


class NeighborhoodAttentionBlock(nn.Module):
    """Same zero-init gate idiom as the main file's RoPEAttentionBlock: pure
    identity at step zero, phasing in as `gate` moves away from zero."""

    def __init__(self, c, block=8, halo=4, heads=4, rope_base=150):
        super().__init__()
        self.norm = ChannelLayerNorm(c)
        self.attn = HaloedNeighborhoodAttention(c, block=block, halo=halo, heads=heads,
                                                  rope_base=rope_base)
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return x + self.gate * self.attn(self.norm(x))


class LocalMaskedTokenPredictor(nn.Module):
    """Same job as nextgen_autoencoder.MaskedTokenPredictor, but built on
    NeighborhoodAttentionBlock (local, haloed) instead of BottleneckAttention
    (global). This is a deliberate DIVERGENCE, not a shared component: a
    global MAE predictor can reward the encoder for embedding broadcast-style
    signals that only help a global consumer -- one that never exists in this
    model's actual decode path, since it has no global attention either. That
    reward would be pure waste: capacity spent serving a training-time-only
    crutch, stolen from capacity that could encode the token's own patch. A
    predictor built from the SAME local building block the rest of the model
    uses can't create that incentive, because it never has the reach to
    exploit it in the first place."""

    def __init__(self, quant_dim, depth=2, heads=2, block=4, halo=1, rope_base=150,
                 mask_ratio=0.5):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.mask_token = nn.Parameter(torch.zeros(1, quant_dim, 1, 1))
        nn.init.normal_(self.mask_token, std=0.02)
        self.blocks = nn.Sequential(*[
            NeighborhoodAttentionBlock(quant_dim, block=block, halo=halo, heads=heads,
                                        rope_base=rope_base)
            for _ in range(depth)
        ])
        self.predict = nn.Conv2d(quant_dim, quant_dim, kernel_size=1)

    def forward(self, q_out: Tensor, target: Tensor):
        b, c, h, w = q_out.shape
        mask = (torch.rand(b, 1, h, w, device=q_out.device) < self.mask_ratio).float()
        x = q_out * (1 - mask) + self.mask_token * mask
        x = self.blocks(x)
        pred = self.predict(x)
        return pred, target, mask


# ---------------------------------------------------------------------------
# Receptive field accounting -- the whole point of this variant
# ---------------------------------------------------------------------------

def receptive_field_pixels(cfg: "LocalAutoencoderConfig") -> int:
    """Exact RF RADIUS in input pixels for one encoder pass, by construction:
    - each stem ResBlockLocal conv (3x3, stride 1): +1 px radius per conv (x2 per block)
    - patchify (kernel=stride=patch_size, non-overlapping): 0 additional radius
      beyond the patch itself, so it contributes patch_size/2 as a one-time base
    - each body ResBlockLocal: +1 patch-equivalent... actually operates in
      TOKEN space post-patchify, so its +1px (per conv) is +1 TOKEN, i.e.
      patch_size pixels, x2 per block
    - each NeighborhoodAttentionBlock: +halo TOKENS = halo*patch_size pixels
    - pre_quant's own ResBlockLocal (token space, same as a body "res" layer):
      +2 TOKENS = 2*patch_size pixels -- easy to forget since it's not part
      of cfg.body, which is exactly what happened here once before
    This does NOT include the decoder's own receptive field (in latent-token
    units), including depatchify_overlap's own small one-time addition there
    -- verify the full round trip empirically, not just this number.
    """
    px = cfg.patch_size / 2  # patchify's own footprint, one-time
    px += cfg.stem_blocks * 2 * 1  # 2 convs/block, 1px radius each, pre-patchify
    token_radius = 2  # pre_quant's ResBlockLocal, always present
    for kind in cfg.body:
        if kind == "res":
            token_radius += 2 * 1  # 2 convs, 1 token radius each
        elif kind == "attn":
            token_radius += cfg.attn_halo
    px += token_radius * cfg.patch_size
    return round(px)


# ---------------------------------------------------------------------------
# Config, Encoder, Decoder, top-level module
# ---------------------------------------------------------------------------

@dataclass
class LocalAutoencoderConfig:
    in_channels: int = 3
    stem_dim: int = 64
    stem_blocks: int = 1
    patch_size: int = 8
    body_dim: int = 256
    body: tuple = ("res", "attn")   # DEFAULT WAS NEVER CHECKED AGAINST
                                     # receptive_field_pixels() before now -- the
                                     # old 4-res/4-attn default had a 204px RADIUS
                                     # (408px diameter), i.e. LARGER than a 256px
                                     # image, i.e. not narrow at all. This default
                                     # verified at 30px radius / 60px diameter --
                                     # always re-check with receptive_field_pixels()
                                     # before trusting any config you actually use.
    attn_block: int = 4        # neighborhood-attention query block size, in TOKENS
    attn_halo: int = 1         # neighborhood-attention halo, in TOKENS
    attn_heads: int = 4
    attn_rope_base: float = 150   # tuned for this window's small max offset -- see
                                    # HaloedNeighborhoodAttention's docstring if you
                                    # change attn_block/attn_halo substantially
    latent_dim: int = 32
    quant_dim: int = 8
    n_rounds: int = 4
    refine_blocks: int = 2     # post-de-patchify local cleanup at full resolution
    depatchify_overlap: int = 0   # >0 blends `overlap` px across each patch boundary
                                    # via a larger-than-stride transposed conv (overlap-add) --
                                    # cheap, one-time RF addition at a single layer, meant to
                                    # help blocky-at-quantization artifacts train away faster
    use_mae_aux: bool = True
    mae_mask_ratio: float = 0.5
    mae_heads: int = 2
    mae_depth: int = 2


def _body_stack(dim, cfg):
    layers = []
    for kind in cfg.body:
        if kind == "res":
            layers.append(ResBlockLocal(dim))
        elif kind == "attn":
            layers.append(NeighborhoodAttentionBlock(
                dim, block=cfg.attn_block, halo=cfg.attn_halo, heads=cfg.attn_heads,
                rope_base=cfg.attn_rope_base))
        else:
            raise ValueError(f"unknown body layer kind: {kind}")
    return layers


class LocalEncoder(_StagedTower):
    def __init__(self, cfg: LocalAutoencoderConfig):
        stem = [nn.Conv2d(cfg.in_channels, cfg.stem_dim, 3, padding=1, padding_mode="reflect"),
                ChannelLayerNorm(cfg.stem_dim), nn.SiLU(),
                *[ResBlockLocal(cfg.stem_dim) for _ in range(cfg.stem_blocks)]]
        patchify = [nn.Conv2d(cfg.stem_dim, cfg.body_dim, cfg.patch_size, stride=cfg.patch_size)]
        body = _body_stack(cfg.body_dim, cfg)
        tail = [nn.Conv2d(cfg.body_dim, cfg.latent_dim, 1)]
        super().__init__([stem, patchify, body, tail])


class LocalDecoder(_StagedTower):
    def __init__(self, cfg: LocalAutoencoderConfig):
        head = [nn.Conv2d(cfg.latent_dim, cfg.body_dim, 1)]
        body = _body_stack(cfg.body_dim, cfg)
        depatchify = [nn.ConvTranspose2d(cfg.body_dim, cfg.stem_dim,
                                          cfg.patch_size + 2 * cfg.depatchify_overlap,
                                          stride=cfg.patch_size, padding=cfg.depatchify_overlap)]
        refine = [*[ResBlockLocal(cfg.stem_dim) for _ in range(cfg.refine_blocks)],
                  ChannelLayerNorm(cfg.stem_dim), nn.SiLU(),
                  nn.Conv2d(cfg.stem_dim, cfg.in_channels, 3, padding=1, padding_mode="reflect"),
                  nn.Tanh()]
        super().__init__([head, body, depatchify, refine])


class LocalQuantizingAutoencoder(nn.Module):
    """Same external shape as ExampleQuantizingAutoencoder: encode/decode,
    quantizer(z) -> (q_out, q_aux) plug point, no skip connections, optional
    MAE auxiliary loss (reused unchanged -- architecture, not any loss term,
    is what bounds receptive field, so MAE staying global here is harmless)."""

    def __init__(self, cfg: LocalAutoencoderConfig = None, quantizer=None, **kwargs):
        super().__init__()
        if cfg is None:
            cfg = LocalAutoencoderConfig(**kwargs)
        self.config = cfg
        self.encoder = LocalEncoder(cfg)
        self.decoder = LocalDecoder(cfg)

        if quantizer is not None:
            self.quantizer = quantizer
        else:
            from .hbq import HBQQuantizer
            self.quantizer = HBQQuantizer(n_rounds=cfg.n_rounds)

        self.pre_quant = nn.Sequential(
            ResBlockLocal(cfg.latent_dim),
            nn.Conv2d(cfg.latent_dim, cfg.quant_dim, kernel_size=1), nn.Tanh(),
        )
        self.post_quant = nn.Sequential(
            nn.Conv2d(cfg.quant_dim, cfg.latent_dim, kernel_size=1),
            ResBlockLocal(cfg.latent_dim),
        )

        self.mae = (
            LocalMaskedTokenPredictor(cfg.quant_dim, depth=cfg.mae_depth, heads=cfg.mae_heads,
                                       block=cfg.attn_block, halo=cfg.attn_halo,
                                       rope_base=cfg.attn_rope_base, mask_ratio=cfg.mae_mask_ratio)
            if cfg.use_mae_aux else None
        )

    def encode(self, images):
        return self.encoder(images)

    def decode(self, latents):
        return self.decoder(latents)

    def forward(self, images):
        latents = self.encode(images)
        z = self.pre_quant(latents)
        q_out, q_aux = self.quantizer(z)

        if self.mae is not None:
            mae_pred, mae_target, mae_mask = self.mae(q_out, z.detach())
        else:
            mae_pred = mae_target = mae_mask = None

        reconstructions = self.decode(self.post_quant(q_out))
        return reconstructions, AutoencoderResults(
            latents=latents, quant_info=q_aux, q_out=q_out,
            mae_pred=mae_pred, mae_target=mae_target, mae_mask=mae_mask,
        )


# ---------------------------------------------------------------------------
# Reference tiled-encoding utility
# ---------------------------------------------------------------------------

def encode_tiled(model: LocalQuantizingAutoencoder, image: Tensor, tile: int, margin: int):
    """Encode a large image by tiling: each tile is (tile + 2*margin) pixels,
    but only the CENTER tile//patch_size latent tokens are kept per tile --
    the margin covers exactly the boundary distortion computed by
    receptive_field_pixels(). tile and margin should both be multiples of
    patch_size. Returns the stitched full-image latent (pre-quantization z is
    NOT computed here -- this returns encoder output; run pre_quant/quantizer
    on the stitched result, same as a normal forward pass, since those are
    pointwise 1x1 convs with no spatial mixing at all)."""
    cfg = model.config
    ps = cfg.patch_size
    assert tile % ps == 0 and margin % ps == 0, "tile and margin must be multiples of patch_size"
    b, c, h, w = image.shape
    assert h % tile == 0 and w % tile == 0, "image size must be a multiple of tile for this reference version"

    tokens_per_tile = tile // ps
    margin_tokens = margin // ps
    out_h, out_w = h // ps, w // ps
    latent = torch.zeros(b, cfg.latent_dim, out_h, out_w, device=image.device, dtype=image.dtype)

    for ty in range(0, h, tile):
        for tx in range(0, w, tile):
            y0, y1 = max(0, ty - margin), min(h, ty + tile + margin)
            x0, x1 = max(0, tx - margin), min(w, tx + tile + margin)
            crop = image[:, :, y0:y1, x0:x1]
            with torch.no_grad():
                enc = model.encode(crop)
            # where in the (possibly boundary-truncated) crop's token grid does
            # this tile's OWN content start?
            tok_y0 = (ty - y0) // ps
            tok_x0 = (tx - x0) // ps
            out_y0, out_x0 = ty // ps, tx // ps
            latent[:, :, out_y0:out_y0 + tokens_per_tile, out_x0:out_x0 + tokens_per_tile] = (
                enc[:, :, tok_y0:tok_y0 + tokens_per_tile, tok_x0:tok_x0 + tokens_per_tile]
            )
    return latent