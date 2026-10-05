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
def generate_simple(net, batch_size, latent_dim, grid_height, grid_width, steps, t_eps,
             method="heun", noise_scale=1.0, device="cuda",
             save_steps=False):
    """ Generate an image from pure noise by taking multiple steps. """
    z = noise_scale * torch.randn(batch_size, latent_dim, grid_height, grid_width, device=device)
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

#######################################################################
# Gemini recommended some differences in the generator
#######################################################################



import math
import torch

@torch.no_grad()
def generate_using_sde(net, batch_size, latent_dim, grid_height, grid_width, steps, t_eps=0.05,
                       noise_scale=1.0, device="cuda", S_churn=0.1, S_min=0.0, S_max=0.9, 
                       save_steps=False):
    """
    Generate an image from pure noise by integrating a Stochastic Differential Equation (SDE)
    constructed from a deterministic Flow Matching vector field.
    
    This function implements an explicit numerical solution to the reverse-time SDE 
    using the classic Euler-Maruyama discretization scheme, combined with stochastic 
    churn mechanics adapted from Karras et al. (Elucidating the Design Space of 
    Diffusion-Based Generative Models, NeurIPS 2022).
    
    https://share.google/aimode/GQu5a6AvrHfCnCG3G
    
    Parameters:
        net: The flow matching neural network, called as net(z, t).
        batch_size (int): Number of parallel samples to generate.
        latent_dim (int): Number of channels in the latent map.
        grid_height/width (int): Spatial dimensions of the latent map.
        steps (int): Total number of discrete integration intervals.
        t_eps (float): Small safety clamping threshold to avoid singularities near t=1.
        S_churn (float): Controls the total amount of stochasticity injected per step.
                         Set to 0.0 to collapse the path back into a deterministic Euler ODE.
        S_min / S_max (float): The active time interval [S_min, S_max] where stochastic noise
                               injection is allowed. Excellent for preventing early structural
                               chaos or late pixel-space clipping.
    """
    # 1. Initialize our trajectory at t=0 (Pure Gaussian Noise)
    z = noise_scale * torch.randn(batch_size, latent_dim, grid_height, grid_width, device=device)
    
    # 2. Construct the time schedule. We step from t=0.0 (Noise) to t=1.0 (Clean Data)
    t_schedule = torch.linspace(0.0, 1.0, steps + 1, device=device)
    saved_steps = []
    
    for i in range(steps):
        t_cur = t_schedule[i]
        t_next = t_schedule[i + 1]
        dt = t_next - t_cur
        
        # Determine if we should inject stochasticity at the current timestep
        # Ref: Karras et al. (2022) algorithmic framework for stochastic churn
        gamma = 0.0
        if S_min <= t_cur.item() <= S_max:
            # Scale gamma based on the step size to keep total variance invariant to step count
            gamma = min(S_churn / steps, math.sqrt(2) - 1)
            
        if gamma > 0:
            # Calculate an explicitly "inflated" temporary timestep
            t_hat = t_cur + gamma * t_cur
            
            # Compute the proportional noise injection magnitude
            # This maintains the exact variance schedule implied by our linear paths
            sigma_fresh = math.sqrt(t_hat**2 - t_cur**2) * noise_scale
            e_fresh = torch.randn_like(z)
            
            # Step 1 of Euler-Maruyama: Ancestral variance inflation
            z = z + sigma_fresh * e_fresh
            t_cur = t_hat
            dt = t_next - t_cur
            
        # 3. Request our core Flow Matching velocity prediction
        t_batch = torch.full((batch_size,), t_cur.item(), device=device)
        x_pred = net(z, t_batch)
        
        # Convert the model's unreduced x_pred back into a trajectory velocity vector field
        # v = (x - z) / (1 - t)
        v_pred = (x_pred - z) / (1 - t_cur).clamp_min(t_eps)
        
        # Step 2 of Euler-Maruyama: Continuous path update
        # z_{t+1} = z_t + dt * f(z_t, t) + (stochastic corrections embedded via dt)
        z = z + dt * v_pred
        
        if save_steps:
            saved_steps.append(x_pred.cpu().detach())
            
    return (z, saved_steps) if save_steps else z

@torch.no_grad()
def generate_using_stochastic_heun(net, batch_size, latent_dim, grid_height, grid_width, steps, t_eps=0.05,
                                   noise_scale=1.0, device="cuda", S_churn=0.1, S_min=0.0, S_max=0.9, 
                                   save_steps=False):
    """
    Generate an image from pure noise using a 2nd-Order Stochastic Heun Sampler.
    
    This function implements the official stochastic predictor-corrector framework 
    detailed in 'Elucidating the Design Space of Diffusion-Based Generative Models' 
    (Karras et al., NeurIPS 2022, Algorithm 2). 
    
    It injects stochastic noise at the beginning of each step interval, then uses 
    Heun's 2nd-order method to accurately integrate across that specific noise state.
    """
    z = noise_scale * torch.randn(batch_size, latent_dim, grid_height, grid_width, device=device)
    t_schedule = torch.linspace(0.0, 1.0, steps + 1, device=device)
    saved_steps = []
    
    for i in range(steps):
        t_cur = t_schedule[i]
        t_next = t_schedule[i + 1]
        
        # --- 1. STOCHASTIC CHURN STEP ---
        # Inject fresh noise to push the state onto a slightly noisier manifold position (t_hat)
        gamma = 0.0
        if S_min <= t_cur.item() <= S_max:
            gamma = min(S_churn / steps, math.sqrt(2) - 1)
            
        t_hat = t_cur + gamma * t_cur
        
        if gamma > 0:
            sigma_fresh = math.sqrt(t_hat**2 - t_cur**2) * noise_scale
            e_fresh = torch.randn_like(z)
            z = z + sigma_fresh * e_fresh
            t_cur = t_hat # Update current time to our noise-inflated point
            
        dt = t_next - t_cur
        
        # --- 2. DETERMINISTIC HEUN PREDICTOR STEP ---
        t_batch_cur = torch.full((batch_size,), t_cur.item(), device=device)
        x_pred_cur = net(z, t_batch_cur)
        v_cur = (x_pred_cur - z) / (1 - t_cur).clamp_min(t_eps)
        
        # Euler predictive step to find our intermediate state (z_prime)
        z_prime = z + dt * v_cur
        
        if t_next >= 1.0:
            # If we are hitting the final data step, collapse cleanly to Euler
            z = z_prime
            if save_steps:
                saved_steps.append(x_pred_cur.cpu().detach())
            continue
            
        # --- 3. DETERMINISTIC HEUN CORRECTOR STEP ---
        t_batch_next = torch.full((batch_size,), t_next.item(), device=device)
        x_pred_next = net(z_prime, t_batch_next)
        v_next = (x_pred_next - z_prime) / (1 - t_next).clamp_min(t_eps)
        
        # 2nd-order trapezoidal integration using the average of both velocities
        v_avg = 0.5 * (v_cur + v_next)
        z = z + dt * v_avg
        
        if save_steps:
            # Average the one-shot predictions for the logging step
            saved_steps.append(((x_pred_cur + x_pred_next) / 2.0).cpu().detach())
            
    return (z, saved_steps) if save_steps else z

###################################################################################

#lds,ldl = get_latent_dataset_and_dataloader(dataset,shuffle=True, batch_size=1)

def generate_using_direct_x_prediction(
        lj,autoencoder,
        x0 = torch.zeros(1,16,64,48),
        t_schedule = torch.arange(0.0,1.0,0.1),
        seed_noise_weight = 1,
        churn_noise_weight = 0,
        save_steps = False,
        ):
    """
        No Euler or Heun required.
        These models are trained for x-prediction,
        so we can just repeatededly predict x.

        Note that when trained on v-space MSE loss
        this converges more slowly than the Euler 
        and Heun samplers (that assume minimal curved
        trajectories).

        But when given a model trained on x-prediction
        with a pixel-space loss function like lpips
        this can converge faster(!)
    """
    device = next(lj.parameters()).device
    x0 = x0.to(device)
    x_pred = x0.clone()
    #print("Original")
    #decoded = autoencoder.decode(autoencoder.post_quant(x0))
    #ipd.display(ih.tensor_to_pil(decoded[0]))
    ew0 = seed_noise_weight
    ew1 = churn_noise_weight
    e0 = torch.randn_like(x0) 
    saved_steps = []
    for t_val in t_schedule:
        e1 = torch.randn_like(x0)
        e = (e0*ew0 + e1*ew1) / (ew0*ew0+ew1*ew1)**0.5
        t = torch.full((x0.size(0),), t_val, device=device)
        z = t.view(-1,1,1,1) * x_pred + (1 - t.view(-1,1,1,1)) * e
        with torch.no_grad():
            x_pred = lj(z, t)
        saved_steps.append(z)
        saved_steps.append(x_pred)
        #noised = ih.tensor_to_pil(autoencoder.decode(autoencoder.post_quant(z))[0])
        #denoised = ih.tensor_to_pil(autoencoder.decode(autoencoder.post_quant(x_pred))[0])
        #print(t_val)
        #ipd.display(ipd.HTML(ih.html_for_images([noised,denoised],f"time {t}")))
    return (x_pred, saved_steps) if save_steps else x_pred
