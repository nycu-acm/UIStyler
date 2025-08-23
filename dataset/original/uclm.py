import os
import random
import argparse
from pathlib import Path
from PIL import Image
import numpy as np
import shutil

random.seed(42)

# ================== UTILS ==================
def create_dirs(base_dir):
    # Create imgs/ and masks/ subfolders
    os.makedirs(os.path.join(base_dir, 'imgs'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'masks'), exist_ok=True)

def convert_mask_to_binary_white(mask_path):
    # Convert color mask to binary (255 for target, 0 for background)
    mask = np.array(Image.open(mask_path))
    binary_mask = np.zeros(mask.shape[:2], dtype=np.uint8)
    if mask.ndim == 3:
        # Red or green pixels are considered positive
        binary_mask[np.all(mask == [255, 0, 0], axis=-1)] = 255  # Malignant
        binary_mask[np.all(mask == [0, 255, 0], axis=-1)] = 255  # Benign
    return Image.fromarray(binary_mask)

def resize_and_save(src_path, dst_path, size=(256, 256), is_mask=False):
    # Resize using NEAREST for masks to preserve labels
    if is_mask:
        mask = convert_mask_to_binary_white(src_path)
        mask = mask.resize(size, resample=Image.NEAREST)
        mask.save(dst_path)
    else:
        img = Image.open(src_path)
        img = img.resize(size, resample=Image.BILINEAR)
        img.save(dst_path)

def classify_mask(mask_path):
    # Determine class from the color mask; return None if not target
    if not os.path.exists(mask_path):
        return None  # Skip normal / missing
    mask = np.array(Image.open(mask_path))
    if mask.ndim == 3:
        if np.any(np.all(mask == [255, 0, 0], axis=-1)):
            return 'malignant'
        elif np.any(np.all(mask == [0, 255, 0], axis=-1)):
            return 'benign'
    return None

# ================== CORE ==================
def process_dataset(image_dir, mask_dir, output_dir, split_ratio=0.7, img_size=(256, 256)):
    # Collect original images/masks
    image_files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith('.png')])
    mask_files = sorted([f for f in os.listdir(mask_dir) if f.lower().endswith('.png')])

    # Print original counts
    print(f"✅ Total images: {len(image_files)}")
    print(f"✅ Total masks: {len(mask_files)}")

    # Filter and assign class based on mask colors
    filtered_images = []
    class_map = {}
    for filename in image_files:
        mask_path = os.path.join(mask_dir, filename)
        class_name = classify_mask(mask_path)
        if class_name is not None:
            filtered_images.append(filename)
            class_map[filename] = class_name

    # Prepare output structure
    os.makedirs(output_dir, exist_ok=True)
    dataset_tag = Path(output_dir).name
    full_dir = os.path.join(output_dir, 'full')
    create_dirs(full_dir)
    full_list = []

    class_counters = {'benign': 0, 'malignant': 0}

    # Step 1: Build FULL split
    for filename in filtered_images:
        img_path = os.path.join(image_dir, filename)
        mask_path = os.path.join(mask_dir, filename)
        class_name = class_map[filename]
        index = class_counters[class_name]
        new_name = f"{class_name}_{index:04d}.png"
        class_counters[class_name] += 1

        resize_and_save(img_path, os.path.join(full_dir, 'imgs', new_name), size=img_size, is_mask=False)
        resize_and_save(mask_path, os.path.join(full_dir, 'masks', new_name), size=img_size, is_mask=True)
        full_list.append(new_name)

    # Write full.txt
    with open(os.path.join(output_dir, 'full.txt'), 'w') as f:
        for item in full_list:
            f.write(f"{dataset_tag}/full/imgs/{item}\n")

    # Step 2: Train/Valid split
    random.shuffle(full_list)
    split_idx = int(len(full_list) * split_ratio)
    train_files = full_list[:split_idx]
    valid_files = full_list[split_idx:]

    for subset_name, subset_files in [('train', train_files), ('valid', valid_files)]:
        subset_dir = os.path.join(output_dir, subset_name)
        create_dirs(subset_dir)
        with open(os.path.join(output_dir, f"{subset_name}.txt"), 'w') as f:
            for fn in subset_files:
                shutil.copy(os.path.join(full_dir, 'imgs', fn), os.path.join(subset_dir, 'imgs', fn))
                shutil.copy(os.path.join(full_dir, 'masks', fn), os.path.join(subset_dir, 'masks', fn))
                f.write(f"{dataset_tag}/{subset_name}/imgs/{fn}\n")

    # Print requested stats
    train_len = len(train_files)
    valid_len = len(valid_files)
    total_len = len(full_list)
    print(f"Total images in train: {train_len}")
    print(f"Total images in valid: {valid_len}")
    print(f"Total images in dataset: {total_len}")
    print("✅ Process completed! No duplicated names between train/valid.")

# ================== CLI ==================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process UCLM-like dataset by converting color masks to binary and splitting into train/valid."
    )
    parser.add_argument("--image_dir", type=str, default="BUS-UCLM/images",
                        help="Path to original images (.png).")
    parser.add_argument("--mask_dir", type=str, default="BUS-UCLM/masks",
                        help="Path to masks (.png) with SAME filenames as images.")
    parser.add_argument("--output_dir", type=str, default="../UCLM",
                        help="Path to save processed dataset (creates full/, train/, valid/).")
    parser.add_argument("--split_ratio", type=float, default=0.7,
                        help="Train ratio; the rest goes to valid. Default: 0.7")
    parser.add_argument("--img_size", type=int, nargs=2, default=[256, 256],
                        help="Resize target size, e.g. --img_size 256 256")

    args = parser.parse_args()
    random.seed(42)

    print("************** Processing UCLM dataset **************")
    process_dataset(
        image_dir=args.image_dir,
        mask_dir=args.mask_dir,
        output_dir=args.output_dir,
        split_ratio=args.split_ratio,
        img_size=tuple(args.img_size),
    )
