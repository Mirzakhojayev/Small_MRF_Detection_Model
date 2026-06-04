import os
import logging
import torch
from torch.utils.data import Dataset
from torchvision import tv_tensors
import torchvision.transforms.v2 as T
from PIL import Image

logger = logging.getLogger(__name__)


class DamageDataset(Dataset):
    """
    Reads image paths from a split .txt file (train.txt / val.txt / test.txt).
    Loads YOLO-format labels (class cx cy w h, normalized) from a labels/ folder.
    Returns: (PIL.Image, boxes Tensor [N,4] xyxy abs, labels Tensor [N,] long)
    """

    def __init__(self, split_txt: str):
        with open(split_txt, 'r') as f:
            self.image_paths = [l.strip() for l in f if l.strip()]
        self.root = os.path.dirname(split_txt)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        basename = os.path.basename(self.image_paths[idx])
        img_path = os.path.normpath(
            os.path.join(self.root, 'images', basename))

        img = Image.open(img_path).convert('RGB')
        W, H = img.size

        stem = os.path.splitext(basename)[0]
        label_path = os.path.join(self.root, 'labels', f"{stem}.txt")

        boxes, labels = [], []
        if os.path.exists(label_path):
            for line in open(label_path):
                parts = line.strip().split()
                if len(parts) == 5:
                    cls = int(float(parts[0]))
                    cx, cy, w, h = map(float, parts[1:])
                    boxes.append(
                        [(cx-w/2)*W, (cy-h/2)*H, (cx+w/2)*W, (cy+h/2)*H])
                    labels.append(cls)
        else:
            logger.warning(f"Missing label: {label_path}")

        boxes = torch.tensor(
            boxes,  dtype=torch.float32) if boxes else torch.zeros((0, 4))
        labels = torch.tensor(labels, dtype=torch.long) if labels else torch.zeros(
            (0,), dtype=torch.long)
        return img, boxes, labels


def _build_train_transform(config: dict) -> T.Compose:
    """
    6 augmentations, all bbox-aware via torchvision.transforms.v2:
      1. RandomHorizontalFlip
      2. ColorJitter  (brightness, contrast, saturation, hue)
      3. RandomAffine (rotation + scale + translate)
      4. GaussianBlur
      5. RandomGrayscale
      6. RandomErasing  (cutout / occlusion simulation)
         + Resize to 640x640 and ToImage/ToDtype for tensor conversion
    """
    aug = config['data'].get('augmentation', {})
    sc = aug.get('scale', [0.5, 1.5])

    return T.Compose([
        T.Resize((640, 640)),
        T.RandomHorizontalFlip(p=aug.get('flip_lr', 0.5)),
        T.ColorJitter(
            brightness=aug.get('hsv_v', 0.4),
            contrast=0.3,
            saturation=aug.get('hsv_s', 0.7),
            hue=aug.get('hsv_h', 0.015),
        ),
        T.RandomAffine(
            degrees=aug.get('rotation_max_deg', 10),
            translate=(0.1, 0.1),
            scale=(sc[0], sc[1]),
            fill=114,
        ),
        T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
        T.RandomGrayscale(p=0.05),
        T.ToImage(),
        # [0, 255] uint8 → [0, 1] float32
        T.ToDtype(torch.float32, scale=True),
        T.RandomErasing(
            p=aug.get('cutout_prob', 0.3),
            scale=(0.01, 0.05),
            ratio=(0.5, 2.0),
            value=114/255.0,
        ),
    ])


def _build_val_transform() -> T.Compose:
    return T.Compose([
        T.Resize((640, 640)),
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
    ])


class AugmentationWrapper(Dataset):
    """
    Wraps DamageDataset and applies torchvision.transforms.v2 augmentations.
    Bounding boxes are passed as tv_tensors.BoundingBoxes so all geometric
    transforms (flip, affine, resize) update them automatically.
    mosaic_prob / mixup_prob are kept as attributes for train.py phase-3 control.
    """

    def __init__(self, dataset: Dataset, config: dict, is_train: bool = True):
        self.dataset = dataset
        self.is_train = is_train
        aug = config['data'].get('augmentation', {})
        self.mosaic_prob = aug.get('mosaic', 1.0) if is_train else 0.0
        self.mixup_prob = aug.get('mixup',  0.1) if is_train else 0.0
        self.transform = _build_train_transform(
            config) if is_train else _build_val_transform()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # PIL image, float32 tensor, long tensor
        img, boxes, labels = self.dataset[idx]
        W, H = img.size

        if boxes.numel() > 0:
            bb = tv_tensors.BoundingBoxes(
                boxes, format='XYXY', canvas_size=(H, W)
            )
        else:
            bb = tv_tensors.BoundingBoxes(
                torch.zeros((0, 4)), format='XYXY', canvas_size=(H, W)
            )

        img_t, bb_t = self.transform(img, bb)

        out_boxes = bb_t.data if hasattr(bb_t, 'data') else bb_t
        if out_boxes.numel() > 0:
            w_b = out_boxes[:, 2] - out_boxes[:, 0]
            h_b = out_boxes[:, 3] - out_boxes[:, 1]
            valid = (w_b >= 1.0) & (h_b >= 1.0)
            out_boxes = out_boxes[valid]
            labels = labels[valid] if labels.numel() > 0 else labels
        else:
            out_boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,),   dtype=torch.long)

        return img_t, out_boxes.float(), labels.long()
