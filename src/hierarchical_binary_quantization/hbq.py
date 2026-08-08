"""Hierarchical Binary Quantization

Algorithm from GRN / arXiv 2604.13030 (Han et al., 2026)
"""

import einx
import random
import torch
import torch.nn as nn
from jaxtyping import Float, Int, jaxtyped
from beartype import beartype
from torch import Tensor
from dataclasses import dataclass

@jaxtyped(typechecker=beartype)
def hbq(z: Float[Tensor,"*shape"], n_rounds:int) -> tuple[Float[Tensor,"*shape"],Int[Tensor,"*shape"]]:
    quantized = torch.zeros_like(z)
    bit_codes = torch.zeros(z.shape,dtype=torch.long,device=z.device)
    for r in range(n_rounds):
        interval = 0.5 ** (r+1)
        high = z > quantized
        bit_codes = bit_codes * 2 + high.long()
        quantized = quantized + torch.where(high, interval, -interval)
    return quantized, bit_codes

@jaxtyped(typechecker=beartype)
def bit_codes_to_tokens(
    bit_codes: Int[Tensor,"*B L"],
    n_rounds:int
) -> Int[Tensor,"*B"] | None:
    latent_dim = bit_codes.shape[-1]
    total_bits = latent_dim * n_rounds
    if total_bits > 63:
        #print("Warning: large vocab doesn't lend itself to use as tokens.")
        return None
    shifts = torch.arange(latent_dim, device=bit_codes.device,dtype=torch.long) * n_rounds
    result = (bit_codes << shifts).sum(dim=-1)
    return result

@jaxtyped(typechecker=beartype)
def tokens_to_bit_codes(
    tokens: Int[Tensor,"*B"],
    latent_dim:int,
    n_rounds:int
) -> Int[Tensor,"*B L"]:
    mask = (1 << n_rounds) - 1
    shifts = torch.arange(latent_dim, device=tokens.device, dtype=torch.long)* n_rounds
    result = (tokens.unsqueeze(-1) >> shifts) & mask
    return result

@jaxtyped(typechecker=beartype)
def bit_codes_to_quantized_latent(bit_codes: Int[Tensor,"*B latent_dim"], n_rounds: int) -> Float[Tensor,"*B latent_dim"]:
    q = torch.zeros(bit_codes.shape,dtype=torch.float32, device=bit_codes.device)
    for r in range(n_rounds):
        interval = 0.5 ** (r+1)
        high = ((bit_codes >> (n_rounds - 1 - r))&1).bool()
        q = q + torch.where(high, interval, -interval)
    return q

@dataclass
class QuantizerAuxOutputs:
    quantized: Float[Tensor, "*B L"]
    bit_codes: Int[Tensor,"*B L"]
    tokens: Int[Tensor,"*B"]|None
    n_rounds: int

class HBQQuantizer(nn.Module):
    def __init__(self, n_rounds:int):
        super().__init__()
        self.n_rounds = n_rounds

    @jaxtyped(typechecker=beartype)
    def forward(self, tx:Float[Tensor,"*B"]) -> tuple[Float[Tensor,"*B"],QuantizerAuxOutputs]:
        q,bit_codes = hbq(tx, self.n_rounds)
        q_with_grad = tx + (q - tx).detach()
        tokens = bit_codes_to_tokens(einx.id("B L H W -> B H W L",bit_codes),self.n_rounds) # type: ignore
        return q_with_grad, QuantizerAuxOutputs(q, bit_codes, tokens, self.n_rounds)

class QuantizerRandomizer(nn.Module):
    def __init__(self, quantizers: list[nn.Module]):
        super().__init__()
        self.quantizers = nn.ModuleList(quantizers)

    def forward(self, x: Float[Tensor,"*B"]) -> tuple[Float[Tensor,"*B"],QuantizerAuxOutputs]:
        idx = torch.randint(len(self.quantizers), (1,)).item()
        q = self.quantizers[idx]
        return q(x)

