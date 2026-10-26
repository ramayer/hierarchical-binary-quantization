"""measure_receptive_field: gradient-based receptive-field measurement for ANY tensor->tensor callable.

Idea: pick one output position, backprop a random projection of its channels to the input, and
look at |grad| over the input grid.
  * support   : where |grad| > eps * max.  For strictly local models (convs, windowed attention) the
                gradient is EXACTLY zero outside the structural support, so this is the tiling-safe
                bound (weight-independent, but needs live paths -> see activate_gates()).
  * effective : where the gradient *energy* sits (Luo et al. 2016 style). Works for models that do
                not truncate leakage at all (global attention, GroupNorm, ...); tells you whether
                the far-away influence is anything but dust.

Everything is reported in INPUT-grid cells times `scale` (use scale=patch_size to report a decoder,
whose input is the token grid, in pixels). Distances are from the output position's centre, measured
to the outer edge of the farthest cell, so left+right == support width.
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass, field

import torch


@dataclass
class RFResult:
    center: tuple            # output-grid (y, x) that was probed
    ref: tuple               # the corresponding point in input coords (cell units, edge convention)
    left: float; right: float; up: float; down: float          # support extents (scaled)
    eff_left: float; eff_right: float; eff_up: float; eff_down: float   # per-axis central-`energy` interval
    eff_radius: float        # radial (L2) radius holding `energy` of the gradient mass
    touches_border: bool     # support reaches the input edge -> numbers are lower bounds
    is_global: bool          # support spans >= `global_frac` of the input on either axis
    grad_map: torch.Tensor | None = field(default=None, repr=False)   # [H, W] mean |grad| over projections

    @property
    def radius(self):        # worst-case one-sided reach
        return max(self.left, self.right, self.up, self.down)

    @property
    def asymmetry(self):     # |left-right| and |up-down|: nonzero => alignment (block/stride) dependence
        return abs(self.left - self.right), abs(self.up - self.down)


@contextlib.contextmanager
def activate_gates(model: torch.nn.Module, value: float = 1.0, match: str = "gate"):
    """Temporarily set every parameter whose name contains `match` to `value`.
    Zero-init residual gates otherwise hide the attention paths, so a fresh model measures as conv-only."""
    saved = {}
    for name, p in model.named_parameters():
        if match in name:
            saved[name] = p.detach().clone()
            with torch.no_grad():
                p.fill_(value)
    try:
        yield
    finally:
        params = dict(model.named_parameters())
        with torch.no_grad():
            for name, v in saved.items():
                params[name].copy_(v)


def sweep_centers(out_hw, period: int = 1, margin: int = 0, anchor=None):
    """All (y, x) offsets within one `period` x `period` cell block, near `anchor` (default: grid middle,
    rounded down to a multiple of `period`). Use period = attn_block (output-grid cells) to expose
    block-alignment effects; stack several anchors by calling twice."""
    H, W = out_hw
    ay, ax = anchor if anchor is not None else (H // 2, W // 2)
    ay -= ay % period; ax -= ax % period
    assert margin <= ay and margin <= ax and ay + period <= H - margin and ax + period <= W - margin, "grid too small"
    return [(ay + dy, ax + dx) for dy in range(period) for dx in range(period)]


def _extent(mask_1d: torch.Tensor):
    idx = torch.nonzero(mask_1d).flatten()
    return int(idx.min()), int(idx.max()) + 1          # [lo, hi) in cells


def _central_interval(profile: torch.Tensor, energy: float):
    c = torch.cumsum(profile, 0) / profile.sum()
    tail = (1 - energy) / 2
    lo = int((c < tail).sum())
    hi = int((c < 1 - tail).sum()) + 1
    return lo, min(hi, len(profile))


def measure_receptive_field(
    fn,
    in_shape,                   # (1, C, H, W) or (C, H, W)
    centers=None,               # list of output-grid (y, x); default: grid middle. Choose your own sweeps!
    *,
    n_proj: int = 4,            # random channel projections per centre (avoids LayerNorm-style cancellation)
    eps: float = 1e-6,          # support threshold, relative to max |grad|
    energy: float = 0.95,       # mass fraction for the effective radius
    scale: float = 1.0,         # multiply reported distances (e.g. patch_size for decoders)
    global_frac: float = 0.9,
    keep_maps: bool = False,
    seed: int = 0,
    device=None,
    dtype=torch.float32,
):
    """Returns list[RFResult], one per centre. `fn` should be in eval() mode and map [1,C,H,W] -> [1,C',h,w].
    Input is random normal (not zeros) so dead-ReLU regions do not hide paths."""
    shape = tuple(in_shape) if len(in_shape) == 4 else (1, *in_shape)
    H, W = shape[-2:]
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(shape, generator=g, dtype=dtype).to(device) if device else torch.randn(shape, generator=g, dtype=dtype)
    x.requires_grad_(True)
    y = fn(x)
    assert y.dim() == 4 and y.shape[0] == 1, f"fn must return [1,C,h,w], got {tuple(y.shape)}"
    h, w = y.shape[-2:]
    sy, sx = H / h, W / w                                # input cells per output cell (<1 for decoders)
    if centers is None:
        centers = [(h // 2, w // 2)]

    results = []
    for (cy, cx) in centers:
        vec = y[0, :, cy, cx]
        acc = torch.zeros(H, W, dtype=torch.float64)
        for k in range(n_proj):
            proj = torch.randn(vec.shape, generator=g, dtype=vec.dtype).to(vec.device)
            (grad,) = torch.autograd.grad((vec * proj).sum(), x, retain_graph=True)
            acc += grad[0].abs().sum(0).double().cpu()
        acc /= n_proj
        peak = acc.max()
        if peak == 0:
            raise RuntimeError(f"zero gradient at centre {(cy, cx)} -- gates still zero? try activate_gates()")
        sup = acc > eps * peak
        x0, x1 = _extent(sup.any(0)); y0, y1 = _extent(sup.any(1))
        ry, rx = (cy + 0.5) * sy, (cx + 0.5) * sx        # centre in input coords, edge convention
        e = acc / acc.sum()
        ex0, ex1 = _central_interval(e.sum(0), energy); ey0, ey1 = _central_interval(e.sum(1), energy)
        yy, xx = torch.meshgrid(torch.arange(H) + 0.5, torch.arange(W) + 0.5, indexing="ij")
        dist = ((yy - ry) ** 2 + (xx - rx) ** 2).sqrt().flatten()
        order = torch.argsort(dist)
        cum = torch.cumsum(e.flatten()[order], 0)
        eff_r = float(dist[order][min(int((cum < energy).sum()), len(cum) - 1)])
        results.append(RFResult(
            center=(cy, cx), ref=(ry, rx),
            left=(rx - x0) * scale, right=(x1 - rx) * scale, up=(ry - y0) * scale, down=(y1 - ry) * scale,
            eff_left=(rx - ex0) * scale, eff_right=(ex1 - rx) * scale,
            eff_up=(ry - ey0) * scale, eff_down=(ey1 - ry) * scale,
            eff_radius=eff_r * scale,
            touches_border=bool(x0 == 0 or y0 == 0 or x1 == W or y1 == H),
            is_global=bool((x1 - x0) >= global_frac * W or (y1 - y0) >= global_frac * H),
            grad_map=acc.float() if keep_maps else None,
        ))
    return results


def summarize(results, label: str = ""):
    """Print worst-case over the probed centres plus the alignment spread."""
    def rng(f):
        v = [f(r) for r in results]; return min(v), max(v)
    rows = [("support radius", lambda r: r.radius), ("left", lambda r: r.left), ("right", lambda r: r.right),
            ("up", lambda r: r.up), ("down", lambda r: r.down), ("effective radius", lambda r: r.eff_radius)]
    print(f"== {label} ({len(results)} centres) ==")
    for name, f in rows:
        lo, hi = rng(f); print(f"  {name:17s} min {lo:8.1f}   max {hi:8.1f}")
    if any(r.touches_border for r in results): print("  ! some supports touch the input border -> lower bounds")
    if any(r.is_global for r in results): print("  ! global support on some centres")