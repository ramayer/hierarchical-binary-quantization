# Scale to [-1, 1] (diffusion models usually expect this)
def scale_to_minus_one_to_one(x):
    return x * 2. - 1.
LR=64
HR=256
# from torch.utils.data import Dataset
# import torch.nn.functional as F
# from torch.utils.data import DataLoader
# from torchvision import datasets, transforms

# =========================
# Simple LR/HR Dataset (hello-world friendly)
# =========================

from dataclasses import dataclass
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import v2
from torchvision.transforms.v2 import functional as TF
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_pil_image


# -------------------------
# Utils
# -------------------------

def to_minus_one_one(x):
    return x * 2.0 - 1.0

def to_zero_one(x):
    return (x + 1.0) * 0.5


# -------------------------
# Simple augmentation config
# -------------------------

@dataclass
class SimpleAugmentConfig:
    hflip: bool = True

    # Crop bias (pixels)
    crop: bool = False
    max_top_crop: int = 4
    max_bottom_crop: int = 24

    # Color jitter
    color_jitter: bool = True


# -------------------------
# Simple square crop, top-biased
# -------------------------

def top_biased_square_crop(img: torch.Tensor, cfg: SimpleAugmentConfig):
    """
    img: [3, H, W], H == W
    """
    _, H, W = img.shape
    assert H == W

    top = torch.randint(0, cfg.max_top_crop + 1, (1,)).item()
    bottom = torch.randint(0, cfg.max_bottom_crop + 1, (1,)).item()

    total_crop = top + bottom
    if total_crop >= H:
        return img

    new_size = int(H - total_crop)

    # Horizontal crop: center-biased
    max_left = W - new_size
    center = max_left // 2
    jitter = torch.randint(-center // 2, center // 2 + 1, (1,)).item()
    left = max(0, min(max_left, center + jitter))

    return img[:, top:top+new_size, left:left+new_size]

def skin_preserving_color_jitter(img: torch.Tensor, xform) -> torch.Tensor:
    """
    img: [3,H,W] in [-1,1]
    """
    img01 = (img + 1) * 0.5
    r, g, b = img01

    # RGB → YCbCr (ITU-R BT.601-ish)
    y  = 0.299 * r + 0.587 * g + 0.114 * b
    cb = 0.564 * (b - y)
    cr = 0.713 * (r - y)

    # Skin mask
    skin = (
        (cr > 0.05) & (cr < 0.25) &
        (cb > -0.15) & (cb < 0.05) &
        (y > 0.2)
    )

    img02 = xform(img01)
    img = torch.where(skin, img01, img02)
    out = img * 2 - 1
    return torch.clamp(out, -1.0, 1.0)


# -------------------------
# Dataset
# -------------------------

class AugmentedHRLRDataset(Dataset):
    def __init__(self, root, HR, LR, aug: SimpleAugmentConfig | None = None, hflip=None):
        self.HR = HR
        self.LR = LR
        self.aug = aug or SimpleAugmentConfig()
        if hflip is not None:
            self.aug.hflip=hflip

        self.base = datasets.ImageFolder(
            root=root,
            transform=v2.Compose([
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Lambda(to_minus_one_one),
            ])
        )

        self.color = v2.ColorJitter(
            brightness=0.1,
            hue=0.5
        )

    def __len__(self):
        return len(self.base)

    def _resize_if_needed(self, img, size):
        if img.shape[1] == size:
            return img
        out =  F.interpolate(
            img.unsqueeze(0),
            size=(size, size),
            mode="bicubic",
            align_corners=False,
            antialias=True
        ).squeeze(0)
        return torch.clamp(out, -1.0, 1.0)

    def __getitem__(self, idx):
        orig, _ = self.base[idx]

        hr = orig.clone()
        #print(f"in AugmentedHRLRDataset a {hr.shape}, {orig.shape}")

        # Flip
        if self.aug.hflip and torch.rand(1) < 0.5:
            hr = torch.flip(hr, dims=[2])
        #print(f"in AugmentedHRLRDataset b {hr.shape}, {orig.shape}")

        # Crop
        if self.aug.crop:
            hr = top_biased_square_crop(hr, self.aug)
            #print(f"in AugmentedHRLRDataset c {hr.shape}, {orig.shape}")

        # Resize to HR
        hr = self._resize_if_needed(hr, self.HR)
        #print(f"in AugmentedHRLRDataset d {hr.shape}, {orig.shape}")

        # Color jitter (expects [0,1])
        if self.aug.color_jitter:
            hr = skin_preserving_color_jitter(hr, self.color)

        # LR derived from HR
        lr = self._resize_if_needed(hr.clone(), self.LR)
        #print(f"in AugmentedHRLRDataset {hr.shape}, {lr.shape}, {orig.shape}")

        return hr, lr, orig

class TwoImageDebugDataset(Dataset):
    """
    Minimal 2-image diagnostic dataset: one all-black background with a
    grey dot, one all-white background with the same grey dot -- same
    size, same position, same color -- so background color is the ONLY
    thing that differs between the two images. Useful for isolating
    whether grey backgrounds come from genuine training/architecture
    averaging vs. a deterministic sampler's inability to express a
    bimodal marginal.

    No augmentation applied -- flip/crop/color-jitter would reintroduce
    variability you're specifically trying to eliminate here. hr/lr use
    the same bicubic antialiased downsize path as AugmentedHRLRDataset,
    so the statistics the model sees match normal training. length lets
    a DataLoader form full batches by cycling between the two images
    (alternating on even/odd index).
    """

    def __init__(self, HR, LR, length=256, dot_radius_frac=0.12,
                 dot_value=0.0, channels=3):
        self.HR = HR
        self.LR = LR
        self.length = length

        self.hr_images = [
            self._make_image(HR, bg_value=-1.0, dot_value=dot_value,
                              dot_radius_frac=dot_radius_frac, channels=channels),
            self._make_image(HR, bg_value=1.0, dot_value=dot_value,
                              dot_radius_frac=dot_radius_frac, channels=channels),
        ]
        self.lr_images = [self._resize(hr, LR) for hr in self.hr_images]

    @staticmethod
    def _make_image(size, bg_value, dot_value, dot_radius_frac, channels):
        img = torch.full((channels, size, size), bg_value, dtype=torch.float32)
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, size), torch.linspace(-1, 1, size), indexing="ij"
        )
        dist = (xx ** 2 + yy ** 2).sqrt()
        mask = dist <= dot_radius_frac
        img[:, mask] = dot_value
        return torch.clamp(img, -1.0, 1.0)

    @staticmethod
    def _resize(img, size):
        if img.shape[-1] == size:
            return img
        out = F.interpolate(
            img.unsqueeze(0), size=(size, size), mode="bicubic",
            align_corners=False, antialias=True
        ).squeeze(0)
        return torch.clamp(out, -1.0, 1.0)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        which = idx % 2
        hr = self.hr_images[which]
        lr = self.lr_images[which]
        orig = hr.clone()  # unused for training; arbitrary resolution field
        return hr, lr, orig


def show_transform_effect(loader,n_images=1, n_augs=4, seed=None):
    """
     usage: show_transform_effect(train_loader)
    """
    if seed is not None:
        torch.manual_seed(seed)
        import random
        random.seed(seed)
    dataset = loader.dataset
    transform = dataset.transform
    idxs=torch.randperm(len(dataset))[:n_images].tolist()
    fig,axes = plt.subplots(n_images,n_augs+1,figsize=(4*n_augs+1,4*n_images),squeeze=False)
    for row,idx in enumerate(idxs):
        dataset.transform=None
        orig = dataset[idx]
        dataset.transform = transform
        ax=axes[row][0]
        ax.imshow(orig)
        w,h = TF.get_image_size(orig)
        ax.set_title(f"orig {w}x{h}",fontsize=10)
        ax.axis("off")
        for col in range(n_augs):
            aug=transform(orig)
            ax = axes[row][col+1]
            ax.imshow(aug.permute(1,2,0)/2+0.5)
            ax.set_title("aug")
            ax.axis("off")
            
