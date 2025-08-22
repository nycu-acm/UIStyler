import os
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from einops import repeat
import numpy as np
import torch

import os
from torch.utils.data import Dataset
from PIL import Image
import torch.nn.functional as F
from torch.nn.modules.utils import _pair

class ImageDataset(Dataset):
    def __init__(self, images_dir, transform=None, mask_transform=None):
        self.images_dir = os.path.join(images_dir, "imgs")
        self.masks_dir = os.path.join(images_dir, "masks")
        self.image_files = [os.path.join(self.images_dir, f)
                            for f in os.listdir(self.images_dir)
                            if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        self.transform = transform
        self.mask_transform = mask_transform

        self.classes = sorted(set([os.path.basename(f).split("_")[0] for f in self.image_files]))
        self.class_to_id = {cls: idx for idx, cls in enumerate(self.classes)}
        self.num_classes = len(self.classes)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        output = {}

        img_path = self.image_files[idx]
        img_name = os.path.basename(img_path)
        image = Image.open(img_path).convert("RGB")
        image_gray = image.convert("L")

        # --- MASK ---
        mask_path = os.path.join(self.masks_dir, img_name)
        mask = Image.open(mask_path).convert("L")
        if self.mask_transform is not None:
            mask = self.mask_transform(mask)
        mask = (mask > 0.5).long()
        
        if self.transform is not None:
            img = self.transform(image)
            img_gray = self.transform(image_gray)
        else:
            img = image
            img_gray = image_gray

        output['img'] = img
        output['img_gray'] = img_gray.repeat(3, 1, 1) if isinstance(img_gray, torch.Tensor) else img_gray
        output['mask'] = mask.squeeze(0)
        output['path'] = img_path
        output["cls_label"] = self.class_to_id[img_name.split("_")[0]]

        return output

##### Transformation #####
class ResizeImage:
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation
    def __call__(self, img):
        return img.resize((self.size, self.size), self.interpolation)

def mask_transform(resize_size=32):
    return transforms.Compose([
        ResizeImage(resize_size, interpolation=Image.NEAREST),
        transforms.ToTensor()
    ])

def train_transform(resize_size=256):
    return transforms.Compose([
            ResizeImage(resize_size),
            transforms.ToTensor()
        ])

def test_transform(resize_size=256):
    return transforms.Compose([
            ResizeImage(resize_size),
            transforms.ToTensor()
        ])

