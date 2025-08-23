import os
import re
import shutil
import random
import argparse
from pathlib import Path
from PIL import Image, ImageChops

# ================== UTILS ==================
def natural_sort_key(s):
    # Natural sort key so numbers in filenames are ordered numerically
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'([0-9]+)', s)]

def create_dirs(base_dir):
    # Create imgs/ and masks/ subfolders
    os.makedirs(os.path.join(base_dir, 'imgs'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'masks'), exist_ok=True)

def resize_and_save(src_path, dst_path, size=(256, 256), is_mask=False):
    # Resize image with appropriate resampling
    img = Image.open(src_path)
    resample_method = Image.NEAREST if is_mask else Image.BILINEAR
    img = img.resize(size, resample=resample_method)
    img.save(dst_path)

def merge_all_masks(class_dir, filename):
    """
    Merge all masks corresponding to an image (logical OR).
    Example: 'benign (4).png' will look for 'benign (4)_mask.png', 'benign (4)_mask_1.png', etc.
    """
    base_name = filename.replace('.png', '')
    mask_files = [f for f in os.listdir(class_dir)
                  if f.startswith(base_name + '_mask') and f.endswith('.png')]

    if not mask_files:
        print(f"⚠️ WARNING: No mask found for image {filename}.")
        return None

    mask_files = sorted(mask_files, key=natural_sort_key)
    merged_mask = None
    for mask_file in mask_files:
        mask = Image.open(os.path.join(class_dir, mask_file)).convert('L')
        mask = mask.point(lambda p: 255 if p > 0 else 0)  # Force binary [0, 255]
        if merged_mask is None:
            merged_mask = mask
        else:
            # Pixel-wise OR via lighter
            merged_mask = ImageChops.lighter(merged_mask, mask)
    return merged_mask

def process_class_to_full(class_dir, class_name, full_dir, full_list, img_size):
    # Collect image files (exclude mask files)
    all_files = [f for f in os.listdir(class_dir) if f.endswith('.png') and '_mask' not in f]
    all_files = sorted(all_files, key=natural_sort_key)

    for filename in all_files:
        img_src = os.path.join(class_dir, filename)
        merged_mask = merge_all_masks(class_dir, filename)
        if merged_mask is None:
            print(f"⏩ Skipping {filename} due to missing mask.")
            continue

        new_name = filename.replace(" ", "_")
        # Save resized image
        resize_and_save(img_src, os.path.join(full_dir, 'imgs', new_name), size=img_size, is_mask=False)
        # Save resized merged mask
        merged_mask = merged_mask.resize(img_size, resample=Image.NEAREST)
        merged_mask.save(os.path.join(full_dir, 'masks', new_name))
        full_list.append(new_name)

def split_from_full(full_dir, output_dir, full_list, split_ratio=0.8):
    # Create train/valid directories
    train_dir = os.path.join(output_dir, 'train')
    valid_dir = os.path.join(output_dir, 'valid')
    create_dirs(train_dir)
    create_dirs(valid_dir)

    # Shuffle and split
    random.shuffle(full_list)
    split_idx = int(len(full_list) * split_ratio)
    train_files = full_list[:split_idx]
    valid_files = full_list[split_idx:]

    train_list, valid_list = [], []
    # Copy files to train
    for filename in train_files:
        shutil.copy(os.path.join(full_dir, 'imgs', filename), os.path.join(train_dir, 'imgs', filename))
        shutil.copy(os.path.join(full_dir, 'masks', filename), os.path.join(train_dir, 'masks', filename))
        train_list.append(f"BUSI/train/imgs/{filename}")

    # Copy files to valid
    for filename in valid_files:
        shutil.copy(os.path.join(full_dir, 'imgs', filename), os.path.join(valid_dir, 'imgs', filename))
        shutil.copy(os.path.join(full_dir, 'masks', filename), os.path.join(valid_dir, 'masks', filename))
        valid_list.append(f"BUSI/valid/imgs/{filename}")

    return train_list, valid_list

def count_original_images_and_masks(original_dir, class_names):
    """
    Count original images (no '_mask') and masks (with '_mask') under class folders.
    """
    num_imgs = 0
    num_masks = 0
    for class_name in class_names:
        class_dir = os.path.join(original_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        for fname in os.listdir(class_dir):
            if fname.endswith(".png"):
                if "_mask" in fname:
                    num_masks += 1
                else:
                    num_imgs += 1
    return num_imgs, num_masks

def split_and_process_dataset(original_dir, output_dir, split_ratio=0.7, img_size=(256, 256)):
    # Define class names (BUSI commonly uses benign/malignant)
    class_names = ['benign', 'malignant']

    # Count original images/masks before processing
    orig_imgs, orig_masks = count_original_images_and_masks(original_dir, class_names)
    print(f"✅ Total image files (no '_mask'): {orig_imgs}")
    print(f"✅ Total mask files (with '_mask'): {orig_masks}")

    # Prepare full directory
    full_dir = os.path.join(output_dir, 'full')
    create_dirs(full_dir)
    full_list = []

    # Process each class into the full/ folder
    for class_name in class_names:
        class_dir = os.path.join(original_dir, class_name)
        process_class_to_full(class_dir, class_name, full_dir, full_list, img_size)
        print(f"✅ Processed class to full: {class_name}")

    # Split from full into train/valid
    train_list, valid_list = split_from_full(full_dir, output_dir, full_list, split_ratio=split_ratio)

    # Write index files
    with open(os.path.join(output_dir, 'train.txt'), 'w') as f:
        for item in train_list:
            f.write(item + '\n')

    with open(os.path.join(output_dir, 'valid.txt'), 'w') as f:
        for item in valid_list:
            f.write(item + '\n')

    with open(os.path.join(output_dir, 'full.txt'), 'w') as f:
        for item in full_list:
            f.write(f"BUSI/full/imgs/{item}\n")

    # Print split stats
    print(f"Total images in train: {len(train_list)}")
    print(f"Total images in valid: {len(valid_list)}")
    print(f"Total images in dataset: {len(full_list)}")

    print("✅ Saved train.txt, valid.txt, full.txt and split dataset successfully!")

# ================== MAIN ==================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process BUSI dataset into train/valid splits with masks")
    parser.add_argument("--original_dir", type=str, default="Dataset_BUSI_with_GT",
                        help="Path to original BUSI dataset root (contains class folders)")
    parser.add_argument("--output_dir", type=str, default="../BUSI",
                        help="Path to save processed dataset")
    parser.add_argument("--split_ratio", type=float, default=0.7,
                        help="Train split ratio (default=0.7)")
    parser.add_argument("--img_size", type=int, nargs=2, default=[256, 256],
                        help="Resize image size, e.g. --img_size 256 256")

    args = parser.parse_args()
    random.seed(42)

    print("************** Processing BUSI dataset **************")
    split_and_process_dataset(
        args.original_dir,
        args.output_dir,
        split_ratio=args.split_ratio,
        img_size=tuple(args.img_size)
    )
