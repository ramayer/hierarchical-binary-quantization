import base64
import io
from typing import List
from PIL import Image, Image as PILImage
import torch
from torch import Tensor
import torchvision.transforms as transforms
import numpy as np

def tensor_to_pil(img):
    """
    Converts a CHW or HWC tensor in [-1,1] (or [0,1]) to a PIL.Image.
    """
    img = img.detach().cpu()
    if img.shape[0] == 3:
        img = img.permute(1, 2, 0)
    if img.min() < 0:
        img = img / 2 + 0.5
    img = img.clamp(0, 1)
    img = img.numpy()
    img = (img * 255).astype(np.uint8)
    return Image.fromarray(img)

def sr_to_pil_legacy(sr_tensor: Tensor) -> PILImage.Image:
    """sr_tensor: [3,H,W] or [B,3,H,W] float in [-1,1] returns: PIL Image (RGB)"""
    if sr_tensor.dim() == 4:
        sr_tensor = sr_tensor[0] # first in batch
    sr_tensor = ((sr_tensor.clamp(-1, 1) + 1) * 127.5).to(torch.uint8)
    sr_np = sr_tensor.permute(1,2,0).cpu().numpy()
    return Image.fromarray(sr_np)

def sr_to_pil(sr_tensor: Tensor) -> PILImage.Image:
    to_pil = transforms.ToPILImage()
    sr_tensor = (sr_tensor + 1) / 2  # scale from [-1,1] to [0,1]
    sr_tensor = sr_tensor.clamp(0, 1)
    return to_pil(sr_tensor)

def pil_to_data_url(pil_img: PILImage.Image) -> str:
    buffered = io.BytesIO()
    pil_img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{img_str}"

def html_for_images(pil_images: List[PILImage.Image], min_height: int = 64, title: str = "") -> str:
    data_urls = [pil_to_data_url(img) for img in pil_images]
    html = ""
    if title: 
        html += f"<h4>{title}</h4>"
    html += f"""<div style="display: flex; flex-wrap: wrap; gap: 2px;">"""
    for url in data_urls:
        html += f"""
        <div style="flex: 0 0 auto;">
            <img src="{url}" style="min-width: {min_height}px;"/>
        </div>
        """
    html += "</div>"
    html += "<style> img {image-rendering: pixelated;}</style>"
    return html

def scale_to_minus_one_to_one(x):
    return x * 2. - 1.

def imgs_to_sr_tensors(imgs, LR=64):
    lr_transform = transforms.Compose([
            transforms.Resize((LR, LR)),
            transforms.ToTensor(),
            transforms.Lambda(scale_to_minus_one_to_one),
    ])
    return torch.stack([lr_transform(img) for img in imgs])
