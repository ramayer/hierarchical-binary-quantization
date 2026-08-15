
import copy
import math
import torch

class EDMEMAHelper:
    """
        Optimized EMA Helper tailored for NVIDIA Elucidated Diffusion (EDM).
        
        Decay time measured in 1000s of images processed.
        Better than epoch based, because for some of our datasets we want to 
        train 100 epochs, while others we'll never be able to afford 0.002 epochs.
    """
    def __init__(self, model, step=1, batch_size=512, ema_halflife_kimg=500):
        # Create a deep copy and freeze it
        self.ema_model = copy.deepcopy(model).eval()
        for p in self.ema_model.parameters():
            p.requires_grad = False
            
        self.step = step
        self.batch_size = batch_size
        self.ema_halflife_kimg = ema_halflife_kimg
        
        # Pre-cache parameter lists to avoid state_dict() overhead in the loop
        self.ema_params = list(self.ema_model.parameters())
        self.model_params = list(model.parameters())
        
        self.ema_buffers = list(self.ema_model.buffers())
        self.model_buffers = list(model.buffers())

    def get_ema_decay(self):
        """
        NVIDIA EDM profile: Computes step-wise decay based on image half-life.
        Adapts to your batch size.
        """
        # Current total images processed (in thousands)
        current_kimg = (self.step * self.batch_size) / 1000.0
        
        # Ramp up the halflife target based on progress
        # This replaces your (1 - 1/step) with a scale matching millions of images
        current_halflife = min(self.ema_halflife_kimg, current_kimg * 2) 
        
        # Calculate the exact step-wise decay rate
        decay = math.exp(math.log(0.5) / (current_halflife * 1000 / self.batch_size))
        return decay

    @torch.no_grad()
    def update(self, model):
        decay = self.get_ema_decay()

        # 1. Update Parameters (Smooth Moving Average)
        for ema_p, model_p in zip(self.ema_params, self.model_params):
            ema_p.mul_(decay).add_(model_p, alpha=1 - decay)
            
        # 2. Update Buffers (Direct copy for tracking stats like BatchNorm)
        for ema_b, model_b in zip(self.ema_buffers, self.model_buffers):
            ema_b.copy_(model_b)
            
        self.step += 1

    def get_model(self):
        return self.ema_model


class CPUOffloadedEMAHelper:
    """Faster on some models, fails on some."""
    def __init__(self, model, step=1, batch_size=512, ema_halflife_kimg=500):
        """
        EMA Helper that stores the EMA weights completely on the CPU,
        freeing up maximum GPU VRAM for training.  From Gemini

        Faster on some models, fails on some.

        Claude's review below:

        Note: One thing worth a quick check before you commit to an overnight 
        run on the compiled version: your CPUOffloadedEMAHelper does 
        copy.deepcopy(model) to build its CPU-resident EMA copy. Deepcopy-ing 
        a torch.compile-wrapped module can behave oddly (compiled state, 
        guards, and the Triton kernel cache are CUDA-specific, and the 
        whole point of your EMA copy is that it lives on CPU) — it's 
        the kind of thing that either works fine or breaks in a way 
        you'd rather discover now than three hours into an unattended run.
        ...
        Good call switching, and I think I can tell you exactly what broke — 
        I pulled up your ema_helper.py to look at CPUOffloadedEMAHelper.update() 
        directly:

            gpu_p_copied_to_cpu = model_p.to('cpu', non_blocking=True)
            ema_p.mul_(decay).add_(gpu_p_copied_to_cpu, alpha=1 - decay)

        This is a classic non_blocking=True footgun. That flag only actually 
        gives you an async transfer if the destination tensor is pinned
        memory — but model_p.to('cpu', ...) allocates a brand-new, 
        unpinned CPU tensor every call. Without a pinned destination, 
        PyTorch can't guarantee the GPU→CPU DMA copy has actually 
        finished by the time the very next line reads from it. 
        The __init__ code pins self.ema_model's own parameters (ema_p), 
        but that's the wrong tensor — it never pins the staging tensor 
        that non_blocking=True actually needs pinned. So the very next 
        line, ema_p.mul_(decay).add_(gpu_p_copied_to_cpu, ...), can 
        race the copy and read from memory that hasn't been written 
        yet — garbage, stale, or zero-filled data, silently, no 
        error. Feeding zeros into ema_p.add_() step after step would 
        plausibly drift your EMA weights toward exactly the kind of 
        degenerate all-black output you saw.
        """
        self.step = step
        self.batch_size = batch_size
        self.ema_halflife_kimg = ema_halflife_kimg
        
        # 1. Create the EMA model clone directly on the CPU
        self.ema_model = copy.deepcopy(model).cpu().eval()
        for p in self.ema_model.parameters():
            p.requires_grad = False
            
        # 2. Pin the CPU memory so the GPU can stream data to it instantly
        for p in self.ema_model.parameters():
            p.pin_memory()
        for b in self.ema_model.buffers():
            b.pin_memory()

        # Cache parameter references
        self.ema_params = list(self.ema_model.parameters())
        self.ema_buffers = list(self.ema_model.buffers())

    def get_ema_decay(self):
        current_kimg = (self.step * self.batch_size) / 1000.0
        current_halflife = min(self.ema_halflife_kimg, current_kimg * 2) 
        return math.exp(math.log(0.5) / (current_halflife * 1000 / self.batch_size))

    @torch.no_grad()
    def update(self, model):
        decay = self.get_ema_decay()
        # Iterate through the active GPU model parameters
        for ema_p, model_p in zip(self.ema_params, model.parameters()):
            # non_blocking=True streams the GPU tensor to the CPU asynchronously
            gpu_p_copied_to_cpu = model_p.to('cpu', non_blocking=True)
            # Do the EMA math strictly on the CPU
            ema_p.mul_(decay).add_(gpu_p_copied_to_cpu, alpha=1 - decay)
            
        # Do the same async streaming for tracking buffers (BatchNorm, etc.)
        for ema_b, model_b in zip(self.ema_buffers, model.buffers()):
            gpu_b_copied_to_cpu = model_b.to('cpu', non_blocking=True)
            ema_b.copy_(gpu_b_copied_to_cpu)
            
        self.step += 1

    def get_model(self):
        return self.ema_model

