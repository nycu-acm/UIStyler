import os
import csv
import shutil
import random
import argparse
from pathlib import Path
from PIL import Image

random.seed(42)

# ================== UTILS ==================
def create_dirs(base_dir):
    os.makedirs(os.path.join(base_dir, 'imgs'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'masks'), exist_ok=True)

def resize_and_save(src_path, dst_path, size=(256, 256), is_mask=False):
    img = Image.open(src_path)
    resample_method = Image.NEAREST if is_mask else Image.BILINEAR
    img = img.resize(size, resample=resample_method)
    img.save(dst_path)

def load_id_to_label(csv_path):
    """Read bus_data.csv and map ID → label ('benign' or 'malignant')."""
    id2label = {}
    with open(csv_path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_id = row.get('ID', '').strip()
            label = (row.get('Pathology') or '').strip().lower()
            if img_id and label in ('benign', 'malignant'):
                id2label[img_id] = label
    return id2label

def prefixed_filename(label, filename):
    """Insert label as prefix before filename (avoid double-prefix)."""
    if filename.lower().startswith(f"{label}_"):
        return filename.replace(" ", "_")
    return f"{label}_{filename.replace(' ', '_')}"

def mask_name_from_image(filename):
    """
    Convert image filename (bus_0001-l.png) → mask filename (mask_0001-l.png).
    """
    base = filename.replace("bus_", "mask_")
    return base

# ================== CORE ==================
def process_busbra_dataset(image_dir, mask_dir, csv_path, output_dir,
                           split_ratio=0.7, img_size=(256, 256)):
    # Load mapping from CSV
    id2label = load_id_to_label(csv_path)

    # Collect files
    image_files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith('.png')])
    mask_files = set([f for f in os.listdir(mask_dir) if f.lower().endswith('.png')])

    print(f"✅ Total images in images: {len(image_files)}")
    print(f"✅ Total images in masks: {len(mask_files)}")

    dataset_tag = Path(output_dir).name
    full_dir = os.path.join(output_dir, 'full')
    create_dirs(full_dir)
    full_list = []

    for filename in image_files:
        img_path = os.path.join(image_dir, filename)
        base_id = Path(filename).stem  # e.g., bus_0001-l
        label = id2label.get(base_id, None)

        if label is None:
            print(f"⚠️ WARNING: No label in CSV for ID '{base_id}'. Skipping {filename}...")
            continue

        # Map to corresponding mask name
        mask_filename = mask_name_from_image(filename)
        mask_path = os.path.join(mask_dir, mask_filename)
        if not os.path.exists(mask_path):
            print(f"⚠️ WARNING: No mask file for {filename} (expected {mask_filename}). Skipping...")
            continue

        # New name with label prefix
        new_name = prefixed_filename(label, filename)

        # Save resized image and mask under new_name
        resize_and_save(img_path, os.path.join(full_dir, 'imgs', new_name),
                        size=img_size, is_mask=False)
        resize_and_save(mask_path, os.path.join(full_dir, 'masks', new_name),
                        size=img_size, is_mask=True)
        full_list.append(new_name)

    # Write full.txt
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'full.txt'), 'w') as f:
        for item in full_list:
            f.write(f"{dataset_tag}/full/imgs/{item}\n")

    # Split train/valid
    random.shuffle(full_list)
    split_idx = int(len(full_list) * split_ratio)
    train_files = full_list[:split_idx]
    valid_files = full_list[split_idx:]

    for subset_name, subset_files in [('train', train_files), ('valid', valid_files)]:
        subset_dir = os.path.join(output_dir, subset_name)
        create_dirs(subset_dir)
        with open(os.path.join(output_dir, f"{subset_name}.txt"), 'w') as f:
            for fn in subset_files:
                shutil.copy(os.path.join(full_dir, 'imgs', fn),
                            os.path.join(subset_dir, 'imgs', fn))
                shutil.copy(os.path.join(full_dir, 'masks', fn),
                            os.path.join(subset_dir, 'masks', fn))
                f.write(f"{dataset_tag}/{subset_name}/imgs/{fn}\n")

    print(f"Total images in train: {len(train_files)}")
    print(f"Total images in valid: {len(valid_files)}")
    print(f"Total images in dataset: {len(full_list)}")
    print("✅ Process completed! Masks renamed and aligned with images.")

# ================== CLI ==================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process BUSBRA dataset with masks renamed to match images and prefixed labels."
    )
    parser.add_argument("--image_dir", type=str, default="BUSBRA/Images")
    parser.add_argument("--mask_dir", type=str, default="BUSBRA/Masks")
    parser.add_argument("--csv_path", type=str, default="BUSBRA/bus_data.csv")
    parser.add_argument("--output_dir", type=str, default="../BUSBRA")
    parser.add_argument("--split_ratio", type=float, default=0.7)
    parser.add_argument("--img_size", type=int, nargs=2, default=[256, 256])

    args = parser.parse_args()
    random.seed(42)

    print("************** Processing BUSBRA dataset **************")
    process_busbra_dataset(
        image_dir=args.image_dir,
        mask_dir=args.mask_dir,
        csv_path=args.csv_path,
        output_dir=args.output_dir,
        split_ratio=args.split_ratio,
        img_size=tuple(args.img_size),
    )
