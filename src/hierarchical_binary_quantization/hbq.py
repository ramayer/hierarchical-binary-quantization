"""Hierarchical Binary Quantization

Algorithm from GRN / arXiv 2604.13030 (Han et al., 2026)
"""

import einx
import random
import torch
import torch.nn as nn
from jaxtyping import Float, Int, jaxtyped, Int8
from beartype import beartype
from torch import Tensor
from dataclasses import dataclass

@jaxtyped(typechecker=beartype)
def hbq(z: Float[Tensor,"*shape"], n_rounds:int) -> tuple[Float[Tensor,"*shape"],Int8[Tensor,"*shape"]]:
    quantized = torch.zeros_like(z)
    bit_codes = torch.zeros(z.shape,dtype=torch.long,device=z.device)
    for r in range(n_rounds):
        interval = 0.5 ** (r+1)
        high = z > quantized
        bit_codes = bit_codes * 2 + high.long()
        quantized = quantized + torch.where(high, interval, -interval)
    if torch.any((bit_codes < -128) | (bit_codes > 127)):
        print("Unexpected to have that many bits")
    bit_codes = bit_codes.to(torch.int8)
    return quantized, bit_codes

@jaxtyped(typechecker=beartype)
def bit_codes_to_tokens(
    bit_codes: Int8[Tensor,"*B L"],
    n_rounds:int
) -> Int[Tensor,"*B"] | None:
    bit_codes = bit_codes.to(torch.int64)
    latent_dim = bit_codes.shape[-1]
    total_bits = latent_dim * n_rounds
    if total_bits > 64:
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
def bit_codes_to_quantized_latent(bit_codes: Int8[Tensor,"*B latent_dim"], n_rounds: int) -> Float[Tensor,"*B latent_dim"]:
    q = torch.zeros(bit_codes.shape,dtype=torch.float32, device=bit_codes.device)
    for r in range(n_rounds):
        interval = 0.5 ** (r+1)
        high = ((bit_codes >> (n_rounds - 1 - r))&1).bool()
        q = q + torch.where(high, interval, -interval)
    return q

@dataclass
class QuantizerAuxOutputs:
    quantized: Float[Tensor, "*B L"]
    bit_codes: Int8[Tensor,"*B L"]
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
        q = self.quantizers[int(idx)]
        return q(x)




import torch
import numpy as np

class HBQEntropyMonitor:
    """
    
    https://share.google/aimode/to3MFwLz0nqQ4CF3S

    Usage: 
        hem = HBQEntropyMonitor()
        hem.print_diagnostics(aux.quant_info.bit_codes)
    
    Output:

        === HBQ BIT PLANE ENTROPY REPORT ===
        Global Bit Utilization: 96.6%
        Columns = Channels (0-15) | Rows = Hierarchical Quantization Rounds
        ---------------------------------------------------------------------------
        Round 1 (Avg 0.99): 1.00 0.99 0.97 0.99 1.00 1.00 0.99 0.99 0.99 0.98 0.99 0.99 0.99 0.98 0.99 0.99
        Round 2 (Avg 0.98): 0.99 1.00 0.93 1.00 1.00 1.00 1.00 0.99 0.99 0.95 0.97 0.97 1.00 0.98 0.96 0.99
        Round 3 (Avg 0.96): 0.94 1.00 0.77 0.98 1.00 1.00 0.93 0.97 1.00 1.00 0.96 0.97 0.99 1.00 0.91 1.00
        Round 4 (Avg 0.93): 1.00 0.94 0.71 0.96 0.89 0.85 0.99 0.96 0.95 0.99 0.93 0.91 0.97 0.92 0.92 0.96
        ---------------------------------------------------------------------------

        Gemini:
        Wow, looking at this output, your Hierarchical Binary Quantization (HBQ) 
        bottleneck is working phenomenally well.
        A 96.6% Global Bit Utilization with an average entropy of 0.93 in Round 4 
        is an outstanding result. It proves that you have absolutely zero representation collapse:
        • There are no dead channels (no columns dropping to zero).
        • The rounds are not short-circuiting; the network is actively using all 4 levels of the tree to pack fine details.
        • Only Channel 2 (Column 3) shows a minor dip down to 0.71 by Round 4, which is completely normal and healthy.

    """
    def __init__(self, quant_dim=16, n_rounds=4):
        """
        Args:
            quant_dim: Number of quantized channels (e.g., 16)
            n_rounds: Number of hierarchical rounds packed into the integer (e.g., 4)
        """
        self.quant_dim = quant_dim
        self.n_rounds = n_rounds
        
    @torch.no_grad()
    def calculate_entropy(self, bit_codes: torch.Tensor):
        """
        Unpacks integer bit codes into binary tensors and calculates Shannon Entropy.
        
        Args:
            bit_codes: Integer tensor containing packed bits.
                       Expected shape: (B, quant_dim, H_latent, W_latent)
        
        Returns:
            entropy_matrix: A numpy array of shape (n_rounds, quant_dim)
            mean_entropy: Global average entropy across all bits
        """
        B, C, H, W = bit_codes.shape
        assert C == self.quant_dim, f"Expected {self.quant_dim} channels, got {C}"
        
        # Flatten spatial and batch dimensions to compute probabilities over all samples
        # Shape: (quant_dim, B * H * W)
        flat_codes = bit_codes.permute(1, 0, 2, 3).reshape(self.quant_dim, -1)
        
        # Initialize an empty tensor to hold the unpacked bits
        # Shape: (n_rounds, quant_dim, B * H * W)
        unpacked_bits = torch.zeros((self.n_rounds, self.quant_dim, flat_codes.shape[1]), 
                                    device=bit_codes.device, dtype=torch.float32)
        
        # Extract each round's bit plane using bitwise shifting and masking
        # Round 1 is typically the lowest bit (LSB) or highest depending on implementation.
        # Here we extract bits from LSB (round 1) upwards.
        for r in range(self.n_rounds):
            # Shift right by 'r' and check if the lowest bit is 1
            unpacked_bits[r] = ((flat_codes >> r) & 1).float()
            
        # Calculate the probability of a bit being "1" for each bit plane
        p1 = unpacked_bits.mean(dim=-1) # Shape: (n_rounds, quant_dim)
        p0 = 1.0 - p1
        
        # Compute Shannon Entropy: -p*log2(p)
        p1 = torch.clamp(p1, min=1e-8, max=1.0 - 1e-8)
        p0 = torch.clamp(p0, min=1e-8, max=1.0 - 1e-8)
        
        entropy = -(p1 * torch.log2(p1) + p0 * torch.log2(p0))
        
        entropy_np = entropy.cpu().numpy()
        mean_entropy = float(entropy_np.mean())
        
        return entropy_np, mean_entropy

    def print_diagnostics(self, bit_codes: torch.Tensor):
        """Prints a scannable grid showing the health of your 64-bit pipeline."""
        entropy_matrix, global_mean = self.calculate_entropy(bit_codes)
        
        print("\n=== HBQ BIT PLANE ENTROPY REPORT ===")
        print(f"Global Bit Utilization: {global_mean * 100:.1f}%\n")
        print("Columns = Channels (0-15) | Rows = Hierarchical Quantization Rounds")
        print("-" * 75)
        
        for r in range(self.n_rounds):
            row_str = " ".join([f"{val:.2f}" for val in entropy_matrix[r]])
            row_mean = entropy_matrix[r].mean()
            print(f"Round {r+1} (Avg {row_mean:.2f}): {row_str}")
            
        print("-" * 75)
        
        # Warning flags for representation collapse
        if entropy_matrix[-1].mean() < 0.1:
            print("⚠️ WARNING: Round 4 has collapsed. Your latent dimensions are too small,")
            print("   or your pre-quant Tanh() is scaling inputs down too heavily.")
        elif entropy_matrix.mean() < 0.5:
            print("⚠️ WARNING: Early rounds have low entropy. The network is under-utilizing")
            print("   its fundamental bit tracks. Check scale initialization.")


    def debug_latent_correlations(self, latent):
        with torch.no_grad():
            z = latent  # (B,C,H,W), BEFORE tanh
            # One row per channel, one column per sample/pixel
            x = z.permute(1, 0, 2, 3).reshape(z.shape[1], -1)
            # Center each channel
            x = x - x.mean(dim=1, keepdim=True)
            # Normalize each channel to unit variance
            x = x / (x.std(dim=1, keepdim=True) + 1e-8)
            # Correlation matrix
            corr = (x @ x.T) / x.shape[1]
            print("Channel correlation:")
            print(corr.cpu().numpy().round(2))
            offdiag = corr - torch.eye(corr.shape[0], device=corr.device)
            mean_abs_corr = offdiag.abs().sum() / (corr.numel() - corr.shape[0])
            print(f"Mean |off-diagonal correlation| = {mean_abs_corr:.3f}")

