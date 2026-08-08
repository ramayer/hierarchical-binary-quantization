import torch
from torch import nn, Tensor
from jaxtyping import Float, Int, jaxtyped
from beartype import beartype
from dataclasses import dataclass
from torch.utils.checkpoint import checkpoint


@dataclass
class AutoencoderResults:
    latents: Tensor
    quant_info: object = None  # Type depends on the optional encoder.


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


class AttentionBlock(nn.Module):
    def __init__(self, c, heads=8):
        super().__init__()
        self.norm = _gn(c)
        self.attn = nn.MultiheadAttention(c, heads, batch_first=True)

    def forward(self, x):
        b, c, h, w = x.shape
        y = self.norm(x).flatten(2).transpose(1, 2)
        y, _ = self.attn(y, y, y, need_weights=False)
        y = y.transpose(1, 2).reshape(b, c, h, w)
        return x + y


class _StagedTower(nn.Module):
    def __init__(self, stage_module_lists):
        super().__init__()
        self.stages = nn.ModuleList(nn.Sequential(*s) for s in stage_module_lists)
        self.ckpt_stage = [False] * len(self.stages)  # one flag per stage
        self.already_mentioned_checkpointing = False

    def forward(self, x):
        """
        Checkpointing only helps on large activations.
        Bytes input is a reasonable proxy for a large activation.
        """
        for stage, do_ckpt in zip(self.stages, self.ckpt_stage):
            bytes = x.numel() * x.element_size()
            if self.training and x.requires_grad and bytes > 60_000_000:
                if not self.already_mentioned_checkpointing:
                    print(f"checkpointing stages with bytes = {bytes}")
                    self.already_mentioned_checkpointing = True
                x = checkpoint(stage, x, use_reentrant=False)
            else:
                x = stage(x)
        return x

    def set_checkpointing(self, enable: bool = True, stages=None):
        """
        enable: True/False
        stages: which stage indices to affect (default: all stages)
        """
        idxs = range(len(self.stages)) if stages is None else stages
        for i in idxs:
            self.ckpt_stage[i] = enable

    def stage_lengths(self):
        """How many submodules landed in each stage, in order. Use this when
        building the remap for an old flat-Sequential checkpoint -- it always
        matches this exact model instance's config, so there's nothing to
        hand-count or get out of sync."""
        return [len(s) for s in self.stages]


class Encoder(_StagedTower):
    def __init__(self, in_channels, dims, res_blocks, latent_dim, use_attention):
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

        # Attention + projection to latent_dim get their own stage: clean in a
        # network diagram, and attention's O(tokens^2) memory profile differs
        # enough from the conv stages that you may want to checkpoint it
        # independently of its neighbors.
        latent_stage = []
        if use_attention:
            latent_stage.append(AttentionBlock(dims[-1]))
        latent_stage.append(nn.Conv2d(dims[-1], latent_dim, 1))
        stages.append(latent_stage)

        super().__init__(stages)


class Decoder(_StagedTower):
    def __init__(self, in_channels, dims, res_blocks, latent_dim, use_attention):
        stages = []

        latent_stage = [nn.Conv2d(latent_dim, dims[-1], 1), _gn(dims[-1]), nn.SiLU()]
        if use_attention:
            latent_stage.append(AttentionBlock(dims[-1]))
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


class ExampleAutoencoder(nn.Module):
    def __init__(
        self,
        in_channels=3,
        base_dim=96,
        channel_multipliers=(1, 2, 4, 6, 6),
        latent_dim=32,
        res_blocks=(1, 2, 3, 4, 4),
        use_attention=True,
    ):
        super().__init__()
        dims = [base_dim * m for m in channel_multipliers]
        if isinstance(res_blocks, int):
            res_blocks = (res_blocks,) * len(dims)

        self.encoder = Encoder(in_channels, dims, res_blocks, latent_dim, use_attention)
        self.decoder = Decoder(in_channels, dims, res_blocks, latent_dim, use_attention)

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
# Migrating an old flat-nn.Sequential checkpoint to the new staged layout
# ---------------------------------------------------------------------------

from .hbq import HBQQuantizer

@dataclass
class HBQAutoencoderConfig:
    in_channels: int = 3
    base_dim: int = 96
    channel_multipliers: tuple[int, ...] = (1, 2, 4, 6, 6)
    res_blocks: tuple[int, ...] = (1, 2, 3, 4, 4)
    latent_dim: int = 32
    quant_dim: int = 16
    n_rounds: int = 4


class ExampleQuantizingAutoencoder(nn.Module):
    def __init__(self,
                 conf: HBQAutoencoderConfig | dict | None = None,
                 quantizer: nn.Module | None = None,
                 **kwargs):
        super().__init__()

        if conf is None:
            conf = HBQAutoencoderConfig(**kwargs)
        self.config = conf

        self.backbone = ExampleAutoencoder(
            in_channels=conf.in_channels,
            base_dim=conf.base_dim,
            channel_multipliers=conf.channel_multipliers,
            latent_dim=conf.latent_dim,
            res_blocks=conf.res_blocks,
            use_attention=True,
        )

        # Parameter-free / near-free per your note -- deliberately not
        # checkpointed. If that ever changes, watch out for the quantizer's
        # randomized round-count variant: use_reentrant=False's automatic RNG
        # save/restore only covers torch's RNG, not Python's `random` module,
        # so if the round-count draw uses `random` rather than `torch.rand`,
        # checkpointing across it could recompute a different round count
        # during backward and silently corrupt gradients.
        self.quantizer = quantizer or HBQQuantizer(n_rounds=conf.n_rounds)
        self.pre_quant = nn.Sequential(
            ResBlock(conf.latent_dim),
            nn.Conv2d(conf.latent_dim, conf.quant_dim, kernel_size=1),
            nn.Tanh(),
        )
        self.post_quant = nn.Sequential(
            nn.Conv2d(conf.quant_dim, conf.latent_dim, kernel_size=1),
            ResBlock(conf.latent_dim),
        )

    def set_grad_checkpointing(self, enable: bool = True):
        self.backbone.set_grad_checkpointing(enable)

    @jaxtyped(typechecker=beartype)
    def forward(self, images: Float[Tensor, "B C H W"]) -> tuple[torch.Tensor, AutoencoderResults]:
        latents = self.backbone.encode(images)
        z = self.pre_quant(latents)
        q_out, q_aux = self.quantizer(z)
        z = self.post_quant(q_out)
        reconstructions = self.backbone.decode(z)
        return reconstructions, AutoencoderResults(latents=latents, quant_info=q_aux)    

# ---------------------------------------------------------------------------
# Usage sketch: debug-small vs long-large runs, auto-enable on real OOM
# ---------------------------------------------------------------------------
#
#   model = ExampleQuantizingAutoencoder(conf)   # checkpointing off by default
#
#   def train_step(model, batch, optimizer):
#       try:
#           loss = compute_loss(model, batch)
#           loss.backward()
#           optimizer.step()
#       except torch.cuda.OutOfMemoryError:
#           optimizer.zero_grad(set_to_none=True)
#           torch.cuda.empty_cache()
#           model.set_grad_checkpointing(True)
#           print("OOM -- enabling gradient checkpointing and retrying")
#           loss = compute_loss(model, batch)
#           loss.backward()
#           optimizer.step()
#       return loss
#
# Small debug configs never pay the recompute cost; large configs pay it only
# once they actually need the memory, and it adapts to whatever GPU you're on
# without you having to guess a channel_multipliers-based threshold.

#############################################################
# Legacy checkpoint loaders.
# Can be removed after all legacy checkpoints have been converted.
#############################################################

def remap_flat_state_dict(flat_state_dict, stage_lengths):
    """
    Convert a state dict saved from the OLD flat nn.Sequential encoder/decoder
    (keys like '0.weight', '3.bias', ...) into the new stage-indexed layout
    ('stages.1.2.weight', ...).

    stage_lengths: list of ints, one per new stage, in order -- how many
        submodules landed in that stage. Pull this straight from a
        constructed instance of the new model, e.g.:
            stage_lengths = new_model.encoder.stage_lengths()
        so it always matches the config you actually built (dims,
        res_blocks, use_attention) rather than being hand-counted.
    """
    offsets, acc = [], 0
    for n in stage_lengths:
        offsets.append(acc)
        acc += n

    new_sd = {}
    for key, tensor in flat_state_dict.items():
        global_idx_str, rest = key.split(".", 1)
        global_idx = int(global_idx_str)
        for stage_idx, offset in enumerate(offsets):
            length = stage_lengths[stage_idx]
            if offset <= global_idx < offset + length:
                local_idx = global_idx - offset
                new_sd[f"stages.{stage_idx}.{local_idx}.{rest}"] = tensor
                break
        else:
            raise KeyError(
                f"index {global_idx} (from key {key!r}) doesn't fall inside any "
                f"stage -- stage_lengths sums to {acc}, so this index is out of "
                f"range. Did stage_lengths come from a model built with the same "
                f"dims/res_blocks/use_attention as the checkpoint you're loading?"
            )
    return new_sd

def load_legacy_checkpoint(new_model: ExampleAutoencoder, old_encoder_sd, old_decoder_sd):
    """
    old_encoder_sd / old_decoder_sd: state_dict()s saved from the ORIGINAL
    flat-nn.Sequential ExampleAutoencoder, i.e.
        old_encoder_sd = old_model.encoder.state_dict()
        old_decoder_sd = old_model.decoder.state_dict()
    from before this refactor.

    Example:
        new_model = ExampleAutoencoder(base_dim=96, channel_multipliers=(1,2,4,6,6),
                                        latent_dim=32, res_blocks=(1,2,3,4,4),
                                        use_attention=True)
        ckpt = torch.load("old_weights.pt")
        load_legacy_checkpoint(new_model, ckpt["encoder_state_dict"], ckpt["decoder_state_dict"])
    """
    enc_sd = remap_flat_state_dict(old_encoder_sd, new_model.encoder.stage_lengths())
    dec_sd = remap_flat_state_dict(old_decoder_sd, new_model.decoder.stage_lengths())
    # strict=True (the default) -- raises loudly on any missing/unexpected key
    # or shape mismatch, which is your safety net against a bad remap.
    new_model.encoder.load_state_dict(enc_sd)
    new_model.decoder.load_state_dict(dec_sd)
    return new_model

def load_legacy_quant_checkpoint(new_model: ExampleQuantizingAutoencoder,
                                  old_quantizing_encoder_checkpoint,
                                  encoder_prefix: str = "backbone.encoder.",
                                  decoder_prefix: str = "backbone.decoder."):
    """
    old_quantizing_encoder_checkpoint: the full state_dict() from the OLD
    (pre-refactor) ExampleQuantizingAutoencoder, i.e.
        old_quantizing_encoder_checkpoint = old_model.state_dict()

    Its keys look like 'backbone.encoder.0.weight', 'backbone.decoder.3.bias',
    'quantizer....', 'pre_quant....', 'post_quant....' -- only the
    backbone.encoder.* / backbone.decoder.* entries need remapping into the
    new staged layout; everything else (quantizer, pre_quant, post_quant)
    wasn't restructured and passes through unchanged.

    If your saved checkpoint uses different key prefixes for the backbone
    (e.g. it was saved some other way), pass encoder_prefix/decoder_prefix to
    match -- print old_quantizing_encoder_checkpoint.keys() first to check.

    Example:
        new_model = ExampleQuantizingAutoencoder(conf)
        old_sd = torch.load("old_quantizing_model.pt")   # a full state_dict
        load_legacy_quant_checkpoint(new_model, old_sd)
    """
    flat_enc_sd = {
        k[len(encoder_prefix):]: v
        for k, v in old_quantizing_encoder_checkpoint.items()
        if k.startswith(encoder_prefix)
    }
    flat_dec_sd = {
        k[len(decoder_prefix):]: v
        for k, v in old_quantizing_encoder_checkpoint.items()
        if k.startswith(decoder_prefix)
    }
    if not flat_enc_sd or not flat_dec_sd:
        raise KeyError(
            f"Found {len(flat_enc_sd)} keys under prefix {encoder_prefix!r} and "
            f"{len(flat_dec_sd)} under {decoder_prefix!r} -- expected both to be "
            f"non-empty. Check old_quantizing_encoder_checkpoint.keys() and pass "
            f"the correct encoder_prefix/decoder_prefix if your checkpoint layout "
            f"differs from 'backbone.encoder.'/'backbone.decoder.'."
        )

    remapped_enc_sd = remap_flat_state_dict(flat_enc_sd, new_model.backbone.encoder.stage_lengths())
    remapped_dec_sd = remap_flat_state_dict(flat_dec_sd, new_model.backbone.decoder.stage_lengths())

    new_sd = {}
    for k, v in remapped_enc_sd.items():
        new_sd[f"backbone.encoder.{k}"] = v
    for k, v in remapped_dec_sd.items():
        new_sd[f"backbone.decoder.{k}"] = v
    # quantizer / pre_quant / post_quant -- unchanged structure, pass through as-is
    for k, v in old_quantizing_encoder_checkpoint.items():
        if not (k.startswith(encoder_prefix) or k.startswith(decoder_prefix)):
            new_sd[k] = v

    new_model.load_state_dict(new_sd)  # strict=True by default -- raises loudly on any mismatch
    return new_model

