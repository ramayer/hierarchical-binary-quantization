import torch
from torch.profiler import profile, ProfilerActivity

def profile_net(net, name, batch_size=8, device="cuda"):
    """Results like:
        Self CPU time total: 632.876ms
        Self CUDA time total: 631.603ms
      are healthy comparables to the top of our leaderboard.
    """
    net = net.to(device).eval()
    x = torch.randn(batch_size, 8, 32, 32, device=device)
    t = torch.rand(batch_size, device=device)
    y = torch.zeros(batch_size, dtype=torch.long, device=device)

    for _ in range(3):
        net(x, t, y)
    torch.cuda.synchronize()

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(10):
            net(x, t, y)
        torch.cuda.synchronize()

    print(f"\n=== {name} ===")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))
    return prof

""" usage:
def show_profiling():
    compiled_net = torch.compile(MinimalSpatialViT(grid_size=32, latent_dim=8, d_model=384, num_heads=6, depth=6))
    prof_minimal = profile_net(compiled_net, "MinimalSpatialViT")
    prof_jit     = profile_net(LatentJiT_small(grid_size=32, latent_dim=8), "LatentJiT_small")
    print(prof_minimal)
    print(prof_jit)
"""
