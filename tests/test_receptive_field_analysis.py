import torch, torch.nn as nn
from hierarchical_binary_quantization.misc.receptive_field_analysis import measure_receptive_field, sweep_centers, summarize, activate_gates

torch.manual_seed(0)

# 1) stride-8 patchify + 7 token-space 3x3 convs  => each token hop = 8px
class Toy(nn.Module):
    def __init__(s, hops):
        super().__init__()
        s.stem = nn.Conv2d(3, 16, 3, padding=1)                       # radius 1 px
        s.pat = nn.Conv2d(16, 32, 8, stride=8)
        s.body = nn.Sequential(*[nn.Sequential(nn.Conv2d(32, 32, 3, padding=1), nn.GELU()) for _ in range(hops)])
    def forward(s, x): return s.body(s.pat(s.stem(x)))

m = Toy(5).eval()
r = measure_receptive_field(m, (1, 3, 512, 512))[0]
# expected width: patch 8 + 2*(8*5) + 2*1 (stem) = 8 + 80 + 2 = 90 -> left+right
print("toy conv:", r.left, r.right, "width", r.left + r.right, "(expect 90)", "touch", r.touches_border, "global", r.is_global)
assert r.left + r.right == 90

# 2) decoder-style (upsampling): ConvTranspose stride 8 from tokens, scale=8 reports ... in token cells
dec = nn.Sequential(nn.Conv2d(32, 32, 3, padding=1), nn.ConvTranspose2d(32, 3, 8, stride=8)).eval()
rd = measure_receptive_field(dec, (1, 32, 32, 32), centers=[(256 // 2, 256 // 2)])[0] if False else \
     measure_receptive_field(dec, (1, 32, 32, 32), centers=[(128, 128)])[0]
print("decoder (token cells):", rd.left, rd.right, rd.up, rd.down, " (expect 1.5 each side-ish: ctr token +-1)")

# 3) global model: mean-pool mixing => global flag, effective radius large
class Glob(nn.Module):
    def __init__(s): super().__init__(); s.c = nn.Conv2d(3, 8, 3, padding=1); s.g = nn.Conv2d(8, 8, 1)
    def forward(s, x):
        h = s.c(x); return h + s.g(h.mean((2, 3), keepdim=True))
rg = measure_receptive_field(Glob().eval(), (1, 3, 64, 64))[0]
print("global:", rg.is_global, "support radius", rg.radius, "eff radius", round(rg.eff_radius, 1))

# 4) alignment sweep on a blocked model: tokens attend only within their 4x4 block (+halo 1)
class Blocked(nn.Module):
    def __init__(s, b=4, h=1):
        super().__init__(); s.b, s.h = b, h; s.v = nn.Conv2d(8, 8, 1); s.gate = nn.Parameter(torch.zeros(1))
    def forward(s, x):
        B, C, H, W = x.shape; b, h = s.b, s.h
        xp = nn.functional.pad(x, (h, h, h, h))
        out = torch.zeros_like(x)
        for by in range(0, H, b):
            for bx in range(0, W, b):
                win = xp[:, :, by:by + b + 2 * h, bx:bx + b + 2 * h]            # block + halo
                ctx = win.mean((2, 3), keepdim=True)                              # stand-in for attention mixing
                out[:, :, by:by + b, bx:bx + b] = x[:, :, by:by + b, bx:bx + b] + s.gate * s.v(ctx)
        return out
bm = Blocked().eval()
c = sweep_centers((16, 16), period=4)
with activate_gates(bm, 1.0):
    rs = measure_receptive_field(lambda t: bm(t), (1, 8, 16, 16), centers=c)
print("blocked (b=4,h=1) lefts/rights per offset:", [(r.left, r.right) for r in rs][:4], "...")
print("  max reach", max(r.radius for r in rs), "(expect b-1+h = 4 worst-case + 0.5 convention)")
r0 = measure_receptive_field(lambda t: bm(t), (1, 8, 16, 16), centers=c[:1])   # gate=0 restored
print("  gate=0 outside context ->", r0[0].left + r0[0].right, "wide (1 cell, identity path only)")
summarize(rs, "blocked sweep")