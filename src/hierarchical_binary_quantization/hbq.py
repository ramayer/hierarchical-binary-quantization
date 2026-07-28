"""Hierarchical Binary Quantization

Algorithm from GRN / arXiv 2604.13030 (Han et al., 2026)
"""

import torch
import torch.nn

def hbq(z: torch.Tensor, n_rounds:int):
    quantized = torch.zeros_like(z)
    tokenids = torch.zeros(z.shape,dtype=torch.int,device=z.device)
    for r in range(n_rounds):
        interval = 0.5 ** (r+1)
        high = z > quantized
        tokenids = tokenids * 2 + high.int()
        quantized = quantized + torch.where(high, interval, -interval)
    return quantized, tokenids

class HBQQuantizer(nn.Module):
    def __init__(self, n_rounds:int):
        super().__init__()
        self.n_rounds = n_rounds
    def forward(self, x):  # [B, N, D]
        tx = torch.tanh(x)
        q,tokenids = hbq(tx, self.n_rounds)
        q_with_grad = tx + (q - tx).detach()
        return q_with_grad, tokenids


