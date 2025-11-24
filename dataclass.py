import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image
from pathlib import Path


class YoloDetectionDataset(Dataset):
    """YOLO-format dataset with reference image support for few-shot detection"""

    def __init__(self, img_dir, lbl_dir, transform=None, ref_root=None, max_refs=3):
        self.img_dir = Path(img_dir)
        self.lbl_dir = Path(lbl_dir)
        self.transform = transform
        self.ref_root = Path(ref_root) if ref_root else None
        self.max_refs = max_refs

        if self.img_dir.exists():
            self.images = sorted(self.img_dir.glob("*.jpg"))
            if not self.images:
                print(f"[WARNING] No .jpg files found in {self.img_dir}")
        else:
            print(f"[ERROR] Image directory does not exist: {self.img_dir}")
            self.images = []

        print(f"[DATASET] Found {len(self.images)} images")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if idx >= len(self.images):
            raise IndexError(f"Index {idx} out of range for {len(self.images)} images")

        img_path = self.images[idx]
        label_path = self.lbl_dir / f"{img_path.stem}.txt"

        try:
            image = Image.open(img_path).convert("RGB")
            w, h = image.size
        except Exception as e:
            print(f"[ERROR] Failed to load image {img_path}: {e}")
            image = Image.new('RGB', (320, 320), color='black')
            w, h = 320, 320

        boxes, labels = [], []
        if label_path.exists():
            try:
                for line in open(label_path):
                    parts = line.strip().split()
                    if len(parts) == 5:
                        cls, xc, yc, bw, bh = map(float, parts)
                        x1, y1 = (xc - bw / 2) * w, (yc - bh / 2) * h
                        x2, y2 = (xc + bw / 2) * w, (yc + bh / 2) * h
                        boxes.append([x1, y1, x2, y2])
                        labels.append(int(cls))
            except Exception as e:
                print(f"[WARNING] Failed to parse label {label_path}: {e}")

        if self.transform:
            image = self.transform(image)

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
        }

        ref_images = []
        if self.ref_root and self.ref_root.exists():
            # Extract video ID from filename (e.g., "Backpack_0_003483.jpg" -> "Backpack_0")
            vid_id = "_".join(img_path.stem.split("_")[:-1])
            ref_dir = self.ref_root / vid_id / "object_images" 
            if ref_dir.exists():
                all_refs = sorted(ref_dir.glob("*.jpg"))
                for ref_path in all_refs[:self.max_refs]:
                    try:
                        ref_img = Image.open(ref_path).convert("RGB")
                        if self.transform:
                            ref_img = self.transform(ref_img)
                        ref_images.append(ref_img)
                    except Exception as e:
                        print(f"[WARNING] Failed to load reference image {ref_path}: {e}")
            else:
                ref_dir_fallback = self.ref_root / vid_id
                if ref_dir_fallback.exists():
                    all_refs = sorted(ref_dir_fallback.glob("*.jpg"))
                    for ref_path in all_refs[:self.max_refs]:
                        try:
                            ref_img = Image.open(ref_path).convert("RGB")
                            if self.transform:
                                ref_img = self.transform(ref_img)
                            ref_images.append(ref_img)
                        except Exception as e:
                            print(f"[WARNING] Failed to load reference image {ref_path}: {e}")

        while len(ref_images) < self.max_refs:
            if len(ref_images) == 0:
                ref_images.append(torch.zeros_like(image))
            else:
                ref_images.append(ref_images[0])

        ref_images = ref_images[:self.max_refs]
        ref_stack = torch.stack(ref_images, dim=0)

        return {
            "frame": image,
            "targets": target,
            "ref_images": ref_stack,
        }


def collate_fn(batch):
    """Custom collate function for batch processing"""
    return {
        "frame": torch.stack([b["frame"] for b in batch]),
        "targets": [b["targets"] for b in batch],
        "ref_images": torch.stack([b["ref_images"] for b in batch]),
    }


def get_transform():
    """Get the data transformation pipeline"""
    transform = T.Compose([
        T.ToTensor(),
        T.Resize((320, 320)),
        T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.8, 1.2)),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform