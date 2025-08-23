import os
import argparse
import random
import shutil
from pathlib import Path
from PIL import Image

random.seed(42)

# ============== UTILS ==============
def create_dirs(base_dir: Path):
    # Create imgs/ and masks/ subfolders
    (base_dir / "imgs").mkdir(parents=True, exist_ok=True)
    (base_dir / "masks").mkdir(parents=True, exist_ok=True)

def resize_and_save(src_path: Path, dst_path: Path, size=(256, 256), is_mask=False):
    # Resize with appropriate resampling
    img = Image.open(src_path)
    resample_method = Image.NEAREST if is_mask else Image.BILINEAR
    img = img.resize(size, resample=resample_method)
    img.save(dst_path)

def index_masks(mask_dir: Path):
    """
    Build an index from a canonical stem -> mask filename.
    Canonical stem removes leading 'mask_' if present.
    e.g., mask file 'mask_0001-l.png' -> key '0001-l'
          mask file '0001-l.png'      -> key '0001-l'
    """
    idx = {}
    collisions = 0
    for f in os.listdir(mask_dir):
        if not f.lower().endswith(".png"):
            continue
        stem = Path(f).stem
        if stem.startswith("mask_"):
            key = stem[len("mask_"):]
        else:
            key = stem
        if key in idx and idx[key] != f:
            collisions += 1
        idx[key] = f
    if collisions > 0:
        print(f"⚠️ WARNING: {collisions} mask key collisions detected (using last seen).")
    return idx

def candidate_keys_for_image(image_name: str):
    """
    Derive candidate keys to match a mask for a given image file.
    The primary key is the image stem itself.
    If the image stem starts with 'benign_' or 'malignant_' (unlikely in inputs),
    we also try removing that prefix.
    """
    stem = Path(image_name).stem
    keys = [stem]
    for prefix in ("benign_", "malignant_"):
        if stem.startswith(prefix):
            keys.append(stem[len(prefix):])
    return keys

# ============== CORE ==============
def build_file_lists(input_root: Path, benign_dirname="Benign", malignant_dirname="Malignant"):
    """
    Collect image files from Benign/ and Malignant/ subfolders.
    Returns a list of tuples: (class_label, filename, full_path)
    """
    items = []
    counts = {"benign": 0, "malignant": 0}

    for label, sub in (("benign", benign_dirname), ("malignant", malignant_dirname)):
        src_dir = input_root / sub
        if not src_dir.exists():
            print(f"⚠️ WARNING: Missing folder: {src_dir}")
            continue
        for f in os.listdir(src_dir):
            if f.lower().endswith(".png"):
                items.append((label, f, src_dir / f))
                counts[label] += 1
    return items, counts

def process_dataset_b2(
    input_root: Path,
    output_dir: Path,
    masks_subdir="all_masks",
    train_ratio=0.7,
    img_size=(256, 256),
    benign_dirname="Benign",
    malignant_dirname="Malignant",
):
    # Build image list from class folders
    image_items, class_counts = build_file_lists(input_root, benign_dirname, malignant_dirname)
    mask_dir = input_root / masks_subdir

    # Print original counts
    benign_n = class_counts.get("benign", 0)
    malignant_n = class_counts.get("malignant", 0)
    print(f"Total images (Benign): {benign_n}")
    print(f"Total images (Malignant): {malignant_n}")

    # Index masks
    mask_index = index_masks(mask_dir)
    print(f"Total masks in '{masks_subdir}': {len(mask_index)}")

    # Prepare output structure
    outdir = output_dir
    train_dir = outdir / "train"
    valid_dir = outdir / "valid"
    full_dir = outdir / "full"
    for folder in (train_dir, valid_dir, full_dir):
        create_dirs(folder)

    train_txt = outdir / "train.txt"
    valid_txt = outdir / "valid.txt"
    full_txt = outdir / "full.txt"
    dataset_tag = outdir.name  # e.g., 'UDIAT'

    # Build list of output filenames after prefixing label
    kept = []           # list of output filenames (with class prefix)
    skipped_missing = 0 # count skipped due to missing masks

    for label, fname, img_path in image_items:
        # Find corresponding mask file by candidate keys
        matched_mask_file = None
        for key in candidate_keys_for_image(fname):
            if key in mask_index:
                matched_mask_file = mask_index[key]
                break

        if matched_mask_file is None:
            skipped_missing += 1
            print(f"⚠️ WARNING: No mask for image '{fname}' (keys tried: {candidate_keys_for_image(fname)}). Skipping...")
            continue

        mask_path = mask_dir / matched_mask_file

        # Output name with label prefix (e.g., benign_123.png). Keep original filename content.
        out_name = f"{label}_{fname.replace(' ', '_')}"
        # Resize + save into FULL
        resize_and_save(img_path, full_dir / "imgs" / out_name, size=img_size, is_mask=False)
        resize_and_save(mask_path, full_dir / "masks" / out_name, size=img_size, is_mask=True)
        kept.append(out_name)

    # Write full.txt
    outdir.mkdir(parents=True, exist_ok=True)
    with open(full_txt, "w") as f_full:
        for name in sorted(kept):
            f_full.write(f"{dataset_tag}/full/imgs/{name}\n")

    # Split into train/valid
    random.shuffle(kept)
    split_idx = int(len(kept) * train_ratio)
    train_files = sorted(kept[:split_idx])
    valid_files = sorted(kept[split_idx:])

    # Copy to TRAIN and VALID, write txt files
    with open(train_txt, "w") as f_train:
        for name in train_files:
            shutil.copy(full_dir / "imgs" / name, train_dir / "imgs" / name)
            shutil.copy(full_dir / "masks" / name, train_dir / "masks" / name)
            f_train.write(f"{dataset_tag}/train/imgs/{name}\n")

    with open(valid_txt, "w") as f_valid:
        for name in valid_files:
            shutil.copy(full_dir / "imgs" / name, valid_dir / "imgs" / name)
            shutil.copy(full_dir / "masks" / name, valid_dir / "masks" / name)
            f_valid.write(f"{dataset_tag}/valid/imgs/{name}\n")

    # Prints
    print(f"Skipped due to missing mask: {skipped_missing}")
    print(f"Total images in full: {len(kept)}")
    print(f"Total images in train: {len(train_files)}")
    print(f"Total images in valid: {len(valid_files)}")
    print("✅ Done! Filenames were prefixed with class and masks were aligned accordingly.")

# ============== CLI ==============
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process DatasetB2 with Benign/Malignant/all_masks structure into UDIAT-style splits."
    )
    parser.add_argument("--input_root", type=str, default="DatasetB2",
                        help="Root folder containing Benign/, Malignant/, and all_masks/ subfolders.")
    parser.add_argument("--output_dir", type=str, default="../UDIAT",
                        help="Output root (will create full/, train/, valid/).")
    parser.add_argument("--train_ratio", type=float, default=0.7,
                        help="Train split ratio (default=0.7).")
    parser.add_argument("--img_size", type=int, nargs=2, default=[256, 256],
                        help="Resize target, e.g., --img_size 256 256")
    parser.add_argument("--benign_dirname", type=str, default="Benign",
                        help="Subfolder name for benign images (default: Benign).")
    parser.add_argument("--malignant_dirname", type=str, default="Malignant",
                        help="Subfolder name for malignant images (default: Malignant).")
    parser.add_argument("--masks_dirname", type=str, default="all_masks",
                        help="Subfolder name for masks (default: all_masks).")

    args = parser.parse_args()
    random.seed(42)

    print("************** Processing UDIAT dataset **************")
    process_dataset_b2(
        input_root=Path(args.input_root),
        output_dir=Path(args.output_dir),
        masks_subdir=args.masks_dirname,
        train_ratio=args.train_ratio,
        img_size=tuple(args.img_size),
        benign_dirname=args.benign_dirname,
        malignant_dirname=args.malignant_dirname,
    )
