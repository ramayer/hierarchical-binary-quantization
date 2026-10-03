import contextlib
import random
import torch
import numpy as np

@contextlib.contextmanager
def torch_random_seed(seed: int):
    """
    Sets all random seeds inside the context block, 
    and completely re-randomizes them upon exit.

    https://share.google/aimode/ExcEcFi0bURvvk9Dp
    """
    # 1. Set the fixed seed for the context block
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    try:
        yield
    finally:
        # 2. Generate a new high-entropy system random seed
        new_seed = random.SystemRandom().randint(0, 2**32 - 1)
        
        # 3. Re-randomize everything so your training loop continues normally
        random.seed(new_seed)
        np.random.seed(new_seed)
        torch.manual_seed(new_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(new_seed)
