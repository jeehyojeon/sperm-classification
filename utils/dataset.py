import os
import csv
import torch
from pathlib import Path
from PIL import Image
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode, functional as F
from torch.utils.data import Dataset

class SquarePad:
    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        max_wh = max(w, h)
        pad_left   = (max_wh - w) // 2
        pad_top    = (max_wh - h) // 2
        pad_right  = max_wh - w - pad_left
        pad_bottom = max_wh - h - pad_top
        return F.pad(img, (pad_left, pad_top, pad_right, pad_bottom), fill=255) # white padding

class SpermDataset(Dataset):
    """
    Sperm normality dataset loader.
    Supports loading images and labels from relative paths.
    """
    def __init__(self, image_dir, csv_path, img_size=224, use_imagenet_norm=True):
        self.image_dir = Path(image_dir)
        
        # Load labels
        self.label_map = {}
        if os.path.exists(csv_path):
            with open(csv_path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    fname = os.path.basename(row["filename"]).strip().lower()
                    cls_raw = str(row["class"]).strip().lower()
                    # normal->1, abnormal->0
                    if cls_raw in ("0", "false", "neg", "abnormal"):
                        y = 0
                    elif cls_raw in ("1", "true", "pos", "normal"):
                        y = 1
                    else:
                        try:
                            y = 1 if int(float(cls_raw)) >= 1 else 0
                        except ValueError:
                            y = 0
                    self.label_map[fname] = y

        self.samples = []
        if self.image_dir.exists():
            for f in sorted(self.image_dir.iterdir()):
                if f.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                    if f.name.lower() in self.label_map:
                        self.samples.append(f)

        # Transforms
        pad_resize = [SquarePad(), T.Resize((img_size, img_size), interpolation=InterpolationMode.BILINEAR)]
        if use_imagenet_norm:
            self.transform = T.Compose(pad_resize + [
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = T.Compose(pad_resize + [T.ToTensor()])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        x = self.transform(img)
        y = torch.tensor(self.label_map[img_path.name.lower()], dtype=torch.float32)
        return x, y
