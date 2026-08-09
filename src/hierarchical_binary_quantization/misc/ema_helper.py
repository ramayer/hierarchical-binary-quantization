
import copy
import math
import torch

class CPUOffloadedEMAHelper:
    def __init__(self, model, step=1, batch_size=512, ema_halflife_kimg=500):
        """
        EMA Helper that stores the EMA weights completely on the CPU,
        freeing up maximum GPU VRAM for training.
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
        """
        Returns the model. Note: If you want to sample/infer from this,
        you will need to do `ema_helper.get_model().to('cuda')`.
        """
        return self.ema_model

