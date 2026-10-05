# from torch.utils.data import Dataset
# import torch.nn.functional as F
# from torch.utils.data import DataLoader
# from torchvision import datasets, transforms

# =========================
# Simple LR/HR Dataset (hello-world friendly)
# =========================

from dataclasses import dataclass
import io
import os
import random
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import v2
from torchvision.transforms.v2 import functional as TF
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm


# Scale to [-1, 1] (diffusion models usually expect this)
def scale_to_minus_one_to_one(x):
    return x * 2. - 1.
LR=64
HR=256


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

    # # Skin mask
    # skin_color = (
    #     (cr > 0.05) & (cr < 0.25) &
    #     (cb > -0.15) & (cb < 0.05) &
    #     (y > 0.2)
    # )
    # Expanded profile to capture shadows, varied lighting, and diverse skin tones
    skin_color = (
        (cr > 0.015) & (cr < 0.25) &
        (cb > -0.22) & (cb < 0.05) &
        (y > 0.10)
    )
    saturation = torch.sqrt(cb**2 + cr**2)
    # Rule out hyper-saturated fruit/objects
    # (Adjust 0.28 down if fruit still leaks through, or up if it cuts real skin)
    skin = skin_color & (saturation < 0.1)

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
            
########################################################
## Newer approach
########################################################

import hashlib
import io
import os
import sqlite3
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as transforms


class BlobDataset(Dataset):
    """
    Arbitray binary blob dataset.
    Works as well on images as latent embedding space tensors.
    """
    TABLE = "blobs"

    def __init__(self, filename, mode="r", where=None):
        self.filename = str(filename)
        self.mode = mode
        self.where = where

        self.conn = None
        self._pid = None
        self._rowids = []
        self._columns = None

        if mode == "w":
            self._connect()
            if self.conn is None:
                print("failed to connect")
                return
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS blobs (
                    hash INTEGER NOT NULL UNIQUE,
                    path TEXT,
                    data BLOB NOT NULL
                )
            """)
            self.conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_blobs_hash
                ON blobs(hash)
            """)
            self.conn.commit()

        elif mode == "r":
            self._connect()
            self._columns = self._get_columns()
            self._rowids = self._query_rowids()

        else:
            raise ValueError("mode must be 'r' or 'w'")

    def _connect(self):
        pid = os.getpid()

        if self.conn is None or self._pid != pid:
            if self.conn is not None:
                self.conn.close()

            if self.mode == "r":
                self.conn = sqlite3.connect(
                    f"file:{self.filename}?mode=ro",
                    uri=True,
                )
            else:
                self.conn = sqlite3.connect(self.filename)

            self._pid = pid

        return self.conn

    def _get_columns(self):
        cursor = self._connect().execute(
            f"SELECT * FROM {self.TABLE} LIMIT 0"
        )
        columns = [column[0] for column in cursor.description]

        for required in ("hash", "data"):
            if required not in columns:
                raise ValueError(
                    f"{self.TABLE} must contain '{required}' column"
                )

        return columns

    def _query_rowids(self):
        sql = f"SELECT rowid FROM {self.TABLE}"

        if self.where:
            sql += f" WHERE {self.where}"

        return [row[0] for row in self._connect().execute(sql)]

    def __len__(self):
        return len(self._rowids)

    def __getitem__(self, index):
        rowid = self._rowids[index]

        row = self._connect().execute(
            f"SELECT * FROM {self.TABLE} WHERE rowid = ?",
            (rowid,),
        ).fetchone()

        if row is None:
            raise IndexError(index)

        return self._make_result(row)

    def __iter__(self):
        conn = self._connect()

        for rowid in self._rowids:
            row = conn.execute(
                f"SELECT * FROM {self.TABLE} WHERE rowid = ?",
                (rowid,),
            ).fetchone()

            if row is not None:
                yield self._make_result(row)

    def _make_result(self, row):
        values = dict(zip(self._columns, row)) # type:ignore
        hash_ = values.pop("hash")
        data = values.pop("data")

        return hash_, data, values

    def write(self, hash_, data, columns=None):
        if self.mode != "w":
            raise RuntimeError("dataset is not writable")

        columns = columns or {}

        names = ["hash", "data", *columns.keys()]
        placeholders = ", ".join("?" for _ in names)

        self._connect().execute(
            f"""
            INSERT INTO {self.TABLE} ({", ".join(names)})
            VALUES ({placeholders})
            ON CONFLICT(hash) DO NOTHING
            """,
            [hash_, data, *columns.values()],
        )

    def commit(self):
        if self.mode != "w":
            raise RuntimeError("dataset is not writable")

        self._connect().commit()

    def close(self):
        if self.conn is not None:
            self.conn.close()
            self.conn = None
            self._pid = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["conn"] = None
        state["_pid"] = None
        return state

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def hash_bytes(data):
    i =  int.from_bytes(
        hashlib.sha256(data).digest()[:8],
        byteorder="big",
        signed=False,
    )
    return i & 0x7fffffffffffffff


def load_image_directory(dataset, directory, 
                         transform=None, 
                         pats = ('*.jpg','*.jpeg','*.webp','*.avif','*.gif')):
    directory = Path(directory)
    paths = []
    for pat in pats:
        paths.extend(directory.rglob(pat))
    for path in tqdm(paths):
        if not path.is_file():
            continue
        original = path.read_bytes()
        hash_ = hash_bytes(original)
        with Image.open(io.BytesIO(original)) as image:
            if transform is not None:
                image = transform(image)
            buffer = io.BytesIO()
            image.save(buffer, format="WEBP")
        dataset.write(
            hash_,
            buffer.getvalue(),
            {"path": str(path.relative_to(directory))},
        )
    dataset.commit()

def make_w_h_dataset(src = '../../edm_diffusion_vibe_coding/data/fantasy',
                     dst = 'tmp_new_ds.sqlite3',
                     w = 384,
                     h = 512,
):
    def resize_with_pil(img:Image.Image,w=w,h=h):
        img.thumbnail(size=(w,h),resample=Image.Resampling.LANCZOS)
        return img
    transform = transforms.Compose([
        resize_with_pil,
    ])
    with BlobDataset(dst, "w") as ds:
        load_image_directory(ds, src, transform=transform)


class PadToRectangle:
    def __init__(self, width: int, height: int, background_color=(0,0,0)):
        self.target_height = height
        self.target_width = width
        self.background_color = background_color

    def __call__(self, img: Image.Image) -> Image.Image:
        img_w, img_h = img.size
        padded_img = Image.new(img.mode, (self.target_width, self.target_height), self.background_color)
        paste_x = max(0, (self.target_width - img_w) // 2)
        paste_y = max(0, (self.target_height - img_h) // 2)
        padded_img.paste(img, (paste_x, paste_y))
        return padded_img



import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

class ImageDataset(Dataset):
    """
        Treat a blob dataset as an image dataset with arbitrary transforms

        transform = transforms.Compose([
            ih.PadToRectangle(384,512),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        imgds = ImageDataset(bds, transform)
        next(iter(imgds))
        imgds[10]
        from hierarchical_binary_quantization.misc.image_helpers import tensor_to_pil
        tensor_to_pil(imgds[10])
    """
    def __init__(self, blob_ds, transform=None):
        self.transform = transform
        self.blob_ds = blob_ds

    def __len__(self):
        return len(self.blob_ds)

    def __getitem__(self, idx):
        id, data, metadata = self.blob_ds[idx]
        img = Image.open(io.BytesIO(data)).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return id, img, metadata



# For mapping between byte-based datasets and tesnor based datasets

def tensor_to_bytes(t):
    buffer = io.BytesIO()
    torch.save(t.to('cpu'), buffer)
    tensor_bytes = buffer.getvalue()
    return tensor_bytes

def bytes_to_tensor(tensor_bytes):
    buffer = io.BytesIO(tensor_bytes)
    restored_tensor = torch.load(buffer, weights_only=True, map_location='cpu')
    return restored_tensor

###############################################################################
# Latent/embedding space datasets
###############################################################################

deprecated="""
import torchvision as tv
import einx
from tqdm import tqdm

def image_dataset_to_quantized_latent_dataset(
        autoencoder, 
        width,height,
        dataset_name=None,
        src_img_path=None,dst_latent_path=None,
        device="cuda"
    ):

    src_img_path = src_img_path or f"data/dbs/{dataset_name}_{width}x{height}.sqlite3"
    dst_latent_path = dst_latent_path or f"data/dbs/quantized_latents_for_{dataset_name}_{width}x{height}.sqlite3"

    if os.path.exists(dst_latent_path):
        print(f"Warning: {dst_latent_path} already existed. Skipping")
        return None

    transform = tv.transforms.Compose([
        PadToRectangle(width=width,height=height,background_color=(255,255,255)),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    img_blob_ds = BlobDataset(src_img_path, mode="r")
    img_ds = ImageDataset(img_blob_ds, transform=transform)
    with torch.inference_mode():
        autoencoder.to(device)
        with BlobDataset(dst_latent_path, "w") as derived:
            for hash_, img_tensor, columns in tqdm(img_ds):
                batch = einx.id("C H W -> 1 C H W", img_tensor)
                raw_latents = autoencoder.encode(batch.to(device))
                pre_quant_latents = autoencoder.pre_quant(raw_latents)
                q_out, q_aux = autoencoder.quantizer(pre_quant_latents)
                bit_codes = q_aux.bit_codes[0].to("cpu") # [C, H, W]
                derived.write(
                    hash_,
                    tensor_to_bytes(bit_codes),
                    columns,
                )
            derived.commit()
    return dst_latent_path

import hierarchical_binary_quantization.hbq as hbq

class QuantizedLatentDataset(Dataset):
    def __init__(self, blob_ds):
        self.blob_ds = blob_ds
    def __len__(self):
        return len(self.blob_ds)
    def __getitem__(self, idx):
        id, data, metadata = self.blob_ds[idx]
        bit_codes = bytes_to_tensor(data)
        qlatents = hbq.bit_codes_to_quantized_latent(bit_codes,4)
        return id, qlatents, metadata
"""

######################################
# Newer, torchvision v2
######################################
import random
import torchvision.transforms.v2 as v2
import torch
import math
import random
import torch
import torchvision.transforms.v2 as T
from torchvision.transforms.v2 import functional as F

import math
import random
import torch
import torchvision.transforms.v2 as T
from torchvision.transforms.v2 import functional as F

class ScalePadCrop(torch.nn.Module):
    def __init__(
        self,
        width=384,
        height=512,
        *,
        p_full_context=0.7,
        cx_beta=(4, 4),
        cy_beta=(2, 5),
        random_pad=True,
        fill=None,
        deterministic=False,  # <-- Set True for structured data like Pokémon cards
    ):
        super().__init__()
        self.width = width
        self.height = height
        self.p_full_context = p_full_context if not deterministic else 1.0
        self.cx_beta = cx_beta
        self.cy_beta = cy_beta
        self.random_pad = random_pad if not deterministic else False
        self.fill = fill
        self.deterministic = deterministic

    @staticmethod
    def _beta(beta):
        return random.betavariate(*beta)

    def forward(self, img):
        orig_h, orig_w = F.get_size(img)

        # ---------------------------------------------------------------------
        # STEP 1: Sizing Path
        # ---------------------------------------------------------------------
        max_fit_scale = min(self.width / orig_w, self.height / orig_h)

        if max_fit_scale >= 1.0:
            # Already fits inside target bounds: protect native resolution
            scale = 1.0
        elif self.deterministic or (random.random() < self.p_full_context):
            # PATH A: Full-Context View (Always taken if deterministic)
            scale = max_fit_scale
        else:
            # PATH B: Zoomed Crop View (Log-Uniform Distribution)
            log_min = math.log(max_fit_scale)
            log_max = math.log(1.0)
            scale = math.exp(random.uniform(log_min, log_max))

        new_w = max(1, int(orig_w * scale))
        new_h = max(1, int(orig_w * scale) if orig_w == orig_h else int(orig_h * scale)) # Keep aspect ratio
        img = F.resize(img, (new_h, new_w), interpolation=T.InterpolationMode.BILINEAR)

        # Update dimensions after resizing
        h, w = F.get_size(img)

        # ---------------------------------------------------------------------
        # STEP 2: Padding (Centered if deterministic)
        # ---------------------------------------------------------------------
        padw = max(self.width - w, 0)
        padh = max(self.height - h, 0)

        if padw or padh:
            if self.random_pad:
                padleft = random.randint(0, padw)
                padtop = random.randint(0, padh)
            else:
                # Evenly split padding to lock the image to the exact center
                padleft = padw // 2
                padtop = padh // 2

            padright = padw - padleft
            padbottom = padh - padtop

            current_fill = (
                img.mean(dim=(-2, -1)).tolist() 
                if self.fill is None and isinstance(img, torch.Tensor)
                else (self.fill if self.fill is not None else 0)
            )

            img = F.pad(img, [padleft, padtop, padright, padbottom], fill=current_fill)
            h, w = F.get_size(img)

        # ---------------------------------------------------------------------
        # STEP 3: Cropping (Centered if deterministic)
        # ---------------------------------------------------------------------
        min_cx = self.width / 2
        max_cx = w - self.width / 2
        min_cy = self.height / 2
        max_cy = h - self.height / 2

        if self.deterministic:
            # Extract perfectly from the spatial dead-center
            cx = min_cx + 0.5 * (max_cx - min_cx) if max_cx > min_cx else min_cx
            cy = min_cy + 0.5 * (max_cy - min_cy) if max_cy > min_cy else min_cy
        else:
            cx = min_cx + self._beta(self.cx_beta) * (max_cx - min_cx) if max_cx > min_cx else min_cx
            cy = min_cy + self._beta(self.cy_beta) * (max_cy - min_cy) if max_cy > min_cy else min_cy

        left = round(cx - self.width / 2)
        top = round(cy - self.height / 2)

        return F.crop(img, top=top, left=left, height=self.height, width=self.width)

class SkinPreservingColorJitter(torch.nn.Module):
    def __init__(self, xform=None):
        super().__init__()
        self.xform = xform or v2.ColorJitter(
            brightness=0.1,
            hue=0.5
        )

    def forward(self,img):
        return skin_preserving_color_jitter(img, self.xform)

### Nice presets

import hierarchical_binary_quantization.misc.image_helpers as ih
import hierarchical_binary_quantization.misc.dataset_helpers as dh
from torchvision.transforms import v2
import random
import torch


def get_augmentation_preset(
        width=384,  height=512,
        augmentation_name = "basic"
    ):
    W,H = width,height
    if augmentation_name == "basic":
        return  v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True), 
            dh.ScalePadCrop(height=H,width=W, random_pad = False, deterministic=True, fill=(1,1,1)),
            v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5),),
        ])
    if augmentation_name == "full_body_portrait":
        # Emphasizes the top half of a body to emphasize facial features
        return v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True), 
            dh.ScalePadCrop(height=H,width=W, cy_beta=(1,10), cx_beta=(6,6),p_full_context=0.5),
            v2.RandomApply([dh.SkinPreservingColorJitter()], p=0.5),
            v2.RandomHorizontalFlip(),
            v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5),),
        ])
    if augmentation_name == "autoencoder_training":
        if W < 384 or H < 384:
            print("Warning, the autoencoder trains better with larger images zoomed to multiple scales")
        return v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True), 
            dh.ScalePadCrop(height=H,width=W, cy_beta=(1,1), cx_beta=(2,2),p_full_context=0),
            v2.RandomHorizontalFlip(),
            v2.RandomVerticalFlip(),
            v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5),),
        ])
    raise ValueError("unexpected augmentation name")

def augmented_image_dataset(
        dataset_name = "fantasy",
        augmentation_name = "full_body_portrait",
        width=384, height=512,
        *,
        base_path="data/dbs",
        src_width=416, src_height=544,
    ):
    if src_width is None and src_height is None:
        src_width,src_height = width,height
    dbpath = f"{base_path}/{dataset_name}_{src_width}x{src_height}.sqlite3"
    augmented_transform = get_augmentation_preset(
        width=width,height=height,
        augmentation_name=augmentation_name
    )
    bds = dh.BlobDataset(dbpath)
    lds = dh.ImageDataset(bds, augmented_transform)
    return lds



        
