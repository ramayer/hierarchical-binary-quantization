import torch
import os
from datetime import datetime
import dataclasses

"""
NOTE - ITS IMPORTANT this works in multiple situations.

* with both the diffusion models and the SR models
* both saving the optimizer state and not
* with the live-training models, and the EMA models

"""

def save_checkpoint(path, model, optimizer, tag="", metadata = {}):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
        "model_class": model.__class__.__name__,
        "model_repr": str(model),  # optional: full repr for reference
        "metadata": metadata,
        "config":model.config,
    }

    if dataclasses.is_dataclass(model.config):
        checkpoint['config'] = dataclasses.asdict(model.config)

    if path is None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        cls_name = model.__class__.__name__
        path = f"checkpoints/{cls_name}_{timestamp}_{tag}.pth"

    dir_name = os.path.dirname(path)
    if dir_name != "":
        os.makedirs(dir_name, exist_ok=True)

    torch.save(checkpoint, path)
    print(f"✅ Saved checkpoint: {path}")
    return path

def check_checkpoint(model, optimizer, path, map_location=None, strict=True):
    checkpoint = torch.load(path, map_location=map_location or "cpu")
    print(f"🔄 Loading checkpoint from {path}")
    print(f"    Model class: {checkpoint.get('model_class', '?')}")
    print(checkpoint["config"])
    
def load_checkpoint(model, optimizer, path, map_location=None, strict=True):
    checkpoint = torch.load(path, map_location=map_location or "cpu")
    print(f"🔄 Loading checkpoint from {path}")
    print(f"    Model class: {checkpoint.get('model_class', '?')}")
    print(checkpoint["config"])
    model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return model, optimizer, checkpoint.get('metadata',{})

def show_model_info(model_edm):
    # Totals
    results = []
    total_params = sum(p.numel() for p in model_edm.parameters())
    trainable_params = sum(p.numel() for p in model_edm.parameters() if p.requires_grad)
    results.append(model_edm.__class__.__name__)
    results.append(f"Total parameters: {total_params:,}")
    results.append(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)\n")
    
    # Breakdown by top-level module (first name segment)
    by_module = {}
    for name, p in model_edm.named_parameters():
        top = name.split('.')[0]
        tot = p.numel()
        by_module.setdefault(top, [0, 0])
        by_module[top][0] += tot
        if p.requires_grad:
            by_module[top][1] += tot
    
    # Print sorted breakdown
    results.append("Parameter breakdown by top-level module:")
    for mod, (tot, train) in sorted(by_module.items(), key=lambda x: x[1][0], reverse=True):
        pct = train / tot * 100 if tot else 0.0
        results.append(f"{mod:35} total: {tot:12,}   trainable: {train:12,}   trainable%: {pct:6.2f}")
    
    # Show largest individual parameter tensors for quick inspection
    results.append("\nTop 10 largest parameter tensors:")
    largest = sorted(model_edm.named_parameters(), key=lambda x: x[1].numel(), reverse=True)[:10]
    for name, p in largest:
        results.append(f"{name:60} shape: {tuple(p.shape)} params: {p.numel():12,}  {'train' if p.requires_grad else 'frozen'}")
    return "\n".join(results)

