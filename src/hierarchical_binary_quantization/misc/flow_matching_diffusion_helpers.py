import math
import torch


### Diffusion Helpers
# * Roughtly following the noise schedules and x-prediction strategies of the JiT paper.
#   * Generate a logit-normal distribution of noise levels
#   * Loss functions that measures de-noising effectiveness.
#   * Functions to take single steps of denoising
#   * Function to start with random noise and loop through denoising steps.

@torch.no_grad()
def forward_sample(net, z, t, t_eps):
    """ Take a step in a denoising direction. """
    x_pred = net(z, t.flatten())
    v_pred = (x_pred - z) / (1 - t).clamp_min(t_eps)
    return v_pred, x_pred

@torch.no_grad()
def euler_step(net, z, t, t_next, t_eps):
    """ https://en.wikipedia.org/wiki/Euler_method """
    v_pred, x_pred = forward_sample(net, z, t, t_eps)
    return z + (t_next - t) * v_pred, x_pred

@torch.no_grad()
def heun_step(net, z, t, t_next, t_eps):
    """ https://en.wikipedia.org/wiki/Heun%27s_method """
    v_t, x_pred = forward_sample(net, z, t, t_eps)
    z_euler = z + (t_next - t) * v_t
    v_t_next, x_pred_next = forward_sample(net, z_euler, t_next, t_eps)
    v_avg = 0.5 * (v_t + v_t_next)
    return z + (t_next - t) * v_avg, (x_pred + x_pred_next)/2


@torch.no_grad()
def generate_simple(net, batch_size, latent_dim, grid_size, steps, t_eps,
             method="heun", noise_scale=1.0, device="cuda",
             save_steps=False):
    """ Generate an image from pure noise by taking multiple steps. """
    z = noise_scale * torch.randn(batch_size, latent_dim, grid_size, grid_size, device=device)
    t_schedule = torch.linspace(0.0, 1.0, steps + 1, device=device)   # (steps+1,) plain values
    stepper = heun_step if method == "heun" else euler_step
    saved_steps=[]
    for i in range(steps - 1):
        z,x_pred = stepper(net, z, t_schedule[i], t_schedule[i + 1], t_eps)
        saved_steps.append(x_pred.cpu().detach())
    result,x_pred = euler_step(net, z, t_schedule[-2], t_schedule[-1], t_eps)
    saved_steps.append(result)
    return (result,saved_steps) if (save_steps) else result

@torch.no_grad()
def generate(net, batch_size, latent_dim, grid_height, grid_width, steps, t_eps,
             method="heun", noise_scale=1.0, device="cuda",
             churn_frac=0.0, t_churn_min=0.0, t_churn_max=0.9, noise_mix=0,
             save_steps=False):
    """ Generate an image from pure noise by taking multiple steps. """
    z = noise_scale * torch.randn(batch_size, latent_dim, grid_height, grid_width, device=device)
    t_schedule = torch.linspace(0.0, 1.0, steps + 1, device=device)
    stepper = heun_step if method == "heun" else euler_step
    saved_steps = []
    for i in range(steps - 1):
        t_cur, t_next = t_schedule[i], t_schedule[i + 1]
        z, x_pred = stepper(net, z, t_cur, t_next, t_eps)

        if churn_frac > 0 and t_churn_min <= t_next.item() <= t_churn_max:
            t_back = (t_next - churn_frac * (t_next - t_cur)).clamp_min(t_eps)

            # what noise is *implied* by the current z and the model's own x_pred --
            # reconstructing this lets us preserve most of the existing trajectory
            e_implied = (z - t_next * x_pred) / (1 - t_next).clamp_min(t_eps)
            e_fresh = torch.randn_like(z) * noise_scale

            # noise_mix=0 -> exact backward move, no new randomness at all
            # noise_mix=1 -> full resample (this was the *only* behavior available before)
            e_mixed = math.sqrt(1 - noise_mix**2) * e_implied + noise_mix * e_fresh

            z = t_back * x_pred + (1 - t_back) * e_mixed

        saved_steps.append(x_pred.cpu().detach())
    result, x_pred = euler_step(net, z, t_schedule[-2], t_schedule[-1], t_eps)
    saved_steps.append(result)
    return (result, saved_steps) if save_steps else result

#####################################################################
# Loss function related functions
#####################################################################
import torch

def generate_flow_matching_samples(net, x, P_mean=-0.8, P_std=0.8, noise_scale=1.0):
    """
    Handles the trajectory mathematics and forward pass.
    Returns unreduced raw tensors for downstream modular losses.
    """
    t_1d = sample_t(x.size(0), P_mean, P_std, device=x.device)
    t_expanded = t_1d[(...,) + (None,) * (x.ndim - 1)] # Human-readable dynamic broadcast
    e = torch.randn_like(x) * noise_scale
    z = t_expanded * x + (1 - t_expanded) * e
    x_pred = net(z, t_1d)
    return x_pred, z, e, t_1d

def compute_v_flow_loss(x, x_pred, z, t_1d, t_eps=0.05):
    """Computes per-sample Flow Matching MSE loss."""
    t_expanded = t_1d[(...,) + (None,) * (x.ndim - 1)]
    dt = (1 - t_expanded).clamp_min(t_eps)
    v_target = (x - z) / dt
    v_pred = (x_pred - z) / dt
    # Flatten everything past the batch dimension and take the mean per sample
    return ((v_target - v_pred) ** 2).flatten(1).mean(dim=1)

def compute_x_l1_loss(x, x_pred):
    """
       Computes per-sample xprediction L1 loss.

       Not at all optimal for flow-matching compared to vflow loss;
       but produces sharper boundaries when used in diffusion models.
       Expect to need more steps to converge compared to vflow loss; but
       to produce finer details.
    """
    return torch.abs(x - x_pred).flatten(1).mean(dim=1)

def compute_lpips_loss(x, x_pred, autoencoder, mask, lpips_fn):
    """
    Per-sample LPIPS loss in pixel space. 
    Purely evaluates whatever samples are marked True in mask,
    because lpips loss on low-t (high noise) samples is undesirable,
    forcing a model to overcommit to details before high level
    structure is established.

    Completely decoupled from v-prediction or flow matching semantics.

    Note that versions trained with LPIPS loss are
    fascinating one-shot-generation artisitic models!!!!!!!!
        https://share.google/aimode/B5ON7BZH8X033JG5a

    """
    lpips_per_sample = torch.zeros(x.size(0), device=x.device)

    if not lpips_fn:
        return lpips_per_sample
    
    # Early exit if no samples passed the external gate
    if not mask.any():
        return lpips_per_sample

    # Target decoding requires no gradients
    with torch.no_grad():
        autoencoder.eval()
        recon_target = autoencoder.decode(autoencoder.post_quant(x[mask]))
    
    # Prediction decoding passes gradients back into the main network
    recon_pred = autoencoder.decode(autoencoder.post_quant(x_pred[mask]))
    
    lpips_results = lpips_fn(recon_pred, recon_target).view(-1)
    lpips_per_sample[mask] = lpips_results
    
    return lpips_per_sample




#===============================
# Log losses per-bin
import torch
from dataclasses import dataclass
@dataclass
class LossBinStats:
    sums: torch.Tensor
    counts: torch.Tensor

def losses_per_t_bin(values: torch.Tensor, t: torch.Tensor, num_bins: int=10) -> LossBinStats:
    bin_indices = (t * num_bins).long().clamp_(0,num_bins-1)
    sums   = torch.zeros(num_bins, device=values.device)
    counts = torch.zeros(num_bins, device=values.device, dtype=torch.long)
    sums   = sums.scatter_add_(0,bin_indices,values.detach())
    counts = counts.scatter_add_(0, bin_indices, torch.ones_like(bin_indices))
    return LossBinStats(sums, counts)

class BinnedLossAccumulator:
    def __init__(self,num_bins, device="cpu", decay=0.9):
        self.decay = decay
        self.sums = torch.zeros(num_bins, device=device)
        self.counts = torch.zeros(num_bins, device=device)
    def update(self, stats:LossBinStats):
        self.sums = self.sums * self.decay + stats.sums.cpu()
        self.counts = self.counts * self.decay + stats.counts.float().cpu()
    def means(self):
        return (self.sums / self.counts.clamp_min(1e-6)).cpu()

#===============================

def sample_t(n, P_mean=-0.8, P_std=0.8, device=None):
    """
    Returns a logit-normal distribution. It's 
    the rectified-flow-world's version of the same 
    idea of EDM's log-normal σ-sampling.

    Determines how much of your compute budget gets spent 
    on which regions of the noise-level spectrum.
    """
    z = torch.randn(n, device=device) * P_std + P_mean
    return torch.sigmoid(z)

#===============================