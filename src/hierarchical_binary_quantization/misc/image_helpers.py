import base64
import io
import numpy as np
import os
import torch
import torchvision.transforms as transforms
from typing import List
from PIL import Image, Image as PILImage
from torch import Tensor
from tqdm import tqdm
from hierarchical_binary_quantization.misc.flow_matching_diffusion_helpers import generate_using_stochastic_heun

# from warnings import deprecated # when we're on newer python

def tensor_to_pil(img):
    """
    Converts a CHW or HWC tensor in [-1,1] (or [0,1]) to a PIL.Image.
    """
    img = img.detach().cpu()
    if img.shape[0] == 3:
        img = img.permute(1, 2, 0)
    if img.min() < 0:
        img = img / 2 + 0.5
    else:
        print("Warning - this project defaults to +/- 1 for most tensors")
    img = img.clamp(0, 1)
    img = img.numpy()
    img = (img * 255).astype(np.uint8)
    return Image.fromarray(img)

#@deprecated("Use tensor_to_pil instead")
def sr_to_pil_legacy(sr_tensor: Tensor) -> PILImage.Image:
    """sr_tensor: [3,H,W] or [B,3,H,W] float in [-1,1] returns: PIL Image (RGB)"""
    if sr_tensor.dim() == 4:
        sr_tensor = sr_tensor[0] # first in batch
    sr_tensor = ((sr_tensor.clamp(-1, 1) + 1) * 127.5).to(torch.uint8)
    sr_np = sr_tensor.permute(1,2,0).cpu().numpy()
    return Image.fromarray(sr_np)

#@deprecated("Use tensor_to_pil instead")
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

def q_out_to_rgb(q_out):
    from sklearn.decomposition import PCA
    B, C, H, W = q_out.shape
    rgb_images = []
    for b in range(B):
        # (C,H,W) -> (H*W,C)
        x = q_out[b].permute(1, 2, 0).reshape(-1, C)
        x = x.detach().cpu().numpy()
        rgb = PCA(n_components=3, whiten=True).fit_transform(x)
        rgb -= rgb.min(axis=0, keepdims=True)
        rgb /= rgb.max(axis=0, keepdims=True) + 1e-8
        rgb = torch.from_numpy(rgb.reshape(H, W, 3)).float()
        rgb = rgb * 2 - 1
        rgb_images.append(rgb)
    return torch.stack(rgb_images)

def rgb_to_ycbcr(x):
    """
    x: (B,3,H,W) in [-1,1]
    returns Y,Cb,Cr in approximately [0,1]
    Fully differentiable.
    """
    x = (x + 1.0) * 0.5
    r = x[:, 0:1]
    g = x[:, 1:2]
    b = x[:, 2:3]
    # BT.601
    y  = 0.299000 * r + 0.587000 * g + 0.114000 * b
    cb = 0.5 + (-0.168736 * r - 0.331264 * g + 0.500000 * b)
    cr = 0.5 + ( 0.500000 * r - 0.418688 * g - 0.081312 * b)
    return torch.cat([y, cb, cr], dim=1)




def generate_gallery(ldm,autoencoder,n=10000,
                     output_dir="outputs/sample_images",seeds=None,
                     grid_width=384//8,grid_height=512//8):
    """Generate multiple images using the provided LDM and autoencoder, with keyboard controls to select images."""

    html_headers = """
        <style>

            .i {
            display: flex;
            flex-direction: column;
            padding: 10px;
            background: #aaaaaa;
            border: 3px solid #ccc;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.2s ease;
            margin: 2px;
            }

            .i:hover {
            border-color: #999;
            }

            .i:focus-visible {
            outline: 1px solid #00ffff;
            outline-offset: 2px;
            }

            .i[aria-pressed="true"] {
            border-color: #ff7700;
            box-shadow: 0 0 10px rgba(127, 255, 127, 0.3);
            background: #ffff00;
            }

            .i img {
            width: 100%;
            height: auto;
            border-radius: 4px;
            }

            .ic {
                display:flex;
                flex-wrap: wrap;
            }
            body {
                background: #888888;
            }
            .caption {
            margin-top: 8px;
            font-size: 14px;
            }

            .i img {
                max-width: 150px;
                height: auto;
            }

            .preview-overlay {
                display: none; /* Hidden by default */
                position: fixed;
                z-index: 9999;
                /*width: 400px;  Adjust preview box width */
                /*height: 300px; Adjust preview box height */
                background: #000;
                border: 4px solid #ff007f; /* Matching your neon theme */
                box-shadow: 0 10px 30px rgba(0,0,0,0.5);
                pointer-events: none; /* Prevents the overlay from intercepting mouse events */
            }

            .preview-overlay img {
                width: 100%;
                height: 100%;
                object-fit: cover;
            }

        </style>
        <div id="image-preview-overlay" class="preview-overlay">
        <img id="preview-img" src="" alt="Full size preview">
        </div>
        <div id="selected_images"></div>
        <script>
            function toggleSelection(buttonElement) {
            const isPressed = buttonElement.getAttribute('aria-pressed') === 'true';
            buttonElement.setAttribute('aria-pressed', !isPressed);
            const itemId = buttonElement.getAttribute('data-id');
            const isNowSelected = !isPressed;
            onItemSelectionChange(itemId, isNowSelected);
            }

            function onItemSelectionChange(iid, isSelected) {
                console.log(`Item ID: ${iid} | Selected: ${isSelected}`);
                if (isSelected) {
                    var selected = document.getElementById('selected_images');
                    selected.innerHTML += ' ' + iid;
                } else {
                    var selected = document.getElementById('selected_images');
                    selected.innerHTML = selected.innerHTML.replace(' ' + iid, '');
                }
            }

            const previewOverlay = document.getElementById('image-preview-overlay');
            const previewImg = document.getElementById('preview-img');

            function showPreview(buttonElement) {
                console.log('Preview shown');

                // Grab the image source inside the hovered button
                const imgSource = buttonElement.querySelector('img').src;
                
                // Update the preview image source (swap with a high-res URL if you have one)
                previewImg.src = imgSource;
                previewOverlay.style.display = 'block';
                }

                function movePreview(event) {
                const mouseX = event.clientX;
                const mouseY = event.clientY;
                
                // Get current viewport dimensions
                const windowWidth = window.innerWidth;
                const windowHeight = window.innerHeight;
                
                // Determine horizontal opposite
                if (mouseX < windowWidth / 2) {
                    // Mouse is on the LEFT -> Place preview on the RIGHT
                    previewOverlay.style.left = 'auto';
                    previewOverlay.style.right = '20px';
                } else {
                    // Mouse is on the RIGHT -> Place preview on the LEFT
                    previewOverlay.style.right = 'auto';
                    previewOverlay.style.left = '20px';
                }
                
                // Determine vertical opposite
                if (mouseY < windowHeight / 2) {
                    // Mouse is on the TOP -> Place preview on the BOTTOM
                    previewOverlay.style.top = 'auto';
                    previewOverlay.style.bottom = '20px';
                } else {
                    // Mouse is on the BOTTOM -> Place preview on the TOP
                    previewOverlay.style.bottom = 'auto';
                    previewOverlay.style.top = '20px';
                }
            }

            function hidePreview() {
                previewOverlay.style.display = 'none';
                previewImg.src = '';
            }



        </script>
        """

    with open(f"{output_dir}/index.html", "w") as f:
        f.write(html_headers)
        f.write("""
        <div class='ic'>
        """)

    try:
        if seeds is None:
            seeds = list(range(n))
        for seed in tqdm(seeds):
            if not os.path.exists(f"{output_dir}/{seed}.webp"):
                torch.manual_seed(seed)
                l1 = generate_using_stochastic_heun(ldm,1,16,
                                                    grid_width=grid_width,grid_height=grid_height,
                                                    steps=25,t_eps=0.01,S_churn=0.05, noise_scale=0.95)

                i1 = autoencoder.decode(autoencoder.post_quant(l1))
                p1 = [tensor_to_pil(i) for i in i1]
                p1[0].save(f"{output_dir}/{seed}.webp",quality=70)
            with open(f"{output_dir}/index.html", "a") as f:
                f.write(f"""
                 <button class='i' onclick='toggleSelection(this)' data-id='{seed}'  onmouseenter='showPreview(this)' onmousemove='movePreview(event)' onmouseleave='hidePreview()' onfocus="showPreview(this)" >
                 {seed}<br />
                 <img src='{seed}.webp'/>
                 </button>
                 """)
    except KeyboardInterrupt as e:
        print("Generation interrupted by user.")
    finally:
        with open(f"{output_dir}/index.html", "a") as f:
            f.write("</div>")