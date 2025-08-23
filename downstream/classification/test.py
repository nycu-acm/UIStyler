import argparse
import os
import re

import torch
from torch.utils.data import DataLoader

import timm
from tqdm import tqdm
from termcolor import colored

from utils.dataloader import ImageList
from utils.preprocess import val_transform
from utils.utils import validate

parser = argparse.ArgumentParser(description="Classification")
# Data parameters
parser.add_argument("--test_dir", type=str, default="./checkpoints")

parser.add_argument("--source", type=str, default="BUSBRA")
parser.add_argument("--target", type=str, default="BUSI")
parser.add_argument("--resize_size", type=int, default=256, help="Resize size")
parser.add_argument("--crop_size", type=int, default=224, help="Crop size")
# Model parameters
parser.add_argument("--model", type=str, default="vit")
parser.add_argument("--cls_checkpoints", type=str, default="./downstream/classification/checkpoints_final",
                    help="Pretrained model")
# Training parameters
parser.add_argument("--bz", type=int, default=16, help="Batch size")
# Device parameters
parser.add_argument("--device", default="cuda:7", help="Device")

args = parser.parse_args()

# ===== Utilities =====
def parse_best_steps(log_path: str):
    if not os.path.isfile(log_path):
        raise FileNotFoundError(f"log.txt not found at: {log_path}")

    pattern = re.compile(
        r"Best accuracy at\s+(\d+):\s*([0-9.]+)%\s+Best AUC at\s+(\d+):\s*([0-9.]+)",
        re.IGNORECASE
    )
    acc_step = auc_step = None

    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = pattern.search(line)
            if m:
                acc_step = int(m.group(1))
                # acc_val = float(m.group(2))  # not used, but kept if you want to print
                auc_step = int(m.group(3))
                # auc_val = float(m.group(4))

    if acc_step is None or auc_step is None:
        raise ValueError(
            "Could not find a line like 'Best accuracy at XXXX: .. Best AUC at YYYY: ..' "
            "in the last occurrence of log.txt."
        )
    return acc_step, auc_step

def step_to_folder_suffix(step: int) -> str:
    return f"{step/1000:.1f}k"

# ===== Parse log.txt and derive result folders =====
test_dir = os.path.join(args.test_dir, f"cp_{args.source}2{args.target}")
log_path = os.path.join(test_dir, "log.txt")
acc_step, auc_step = parse_best_steps(log_path)

acc_suffix = step_to_folder_suffix(acc_step) 
auc_suffix  = step_to_folder_suffix(auc_step)

acc_results_dir = os.path.join(test_dir, "results", f"results_{acc_suffix}")
auc_results_dir = os.path.join(test_dir, "results", f"results_{auc_suffix}")

print(colored(f"Found best markers in log.txt → accuracy@{acc_step} ({acc_suffix}), AUC@{auc_step} ({auc_suffix})", "yellow", force_color=True))
print(colored(f"→ Will evaluate:", "yellow", force_color=True))
print(colored(f"   1) {acc_results_dir}", "yellow", force_color=True))
print(colored(f"   2) {auc_results_dir}", "yellow", force_color=True))

# ===== Prepare model once =====
device = torch.device(args.device)
print(colored(f"Loading model: {args.model}", color="red", force_color=True))
model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=2)

# Load your checkpoints (assumes head is included in checkpoint)
cp_dir = os.path.join(args.cls_checkpoints, f"vit_Breast-{args.target}/best.pth")
print(colored(f"Loading pretrained weights from {cp_dir}", color="blue", force_color=True))
state = torch.load(cp_dir, map_location=device)
model.load_state_dict(state)
model = model.to(device)
model.eval()

# ===== Common transform =====
resize_size = args.resize_size
crop_size = args.crop_size
transform_w = val_transform(resize_size, crop_size)

def evaluate_one_folder(folder_path, acc=True):
    if not os.path.isdir(folder_path):
        raise FileNotFoundError(f"Test folder does not exist: {folder_path}")

    print(colored(f"\nLoading dataset from {folder_path}", "blue", force_color=True))

    test_dataset = ImageList(folder_path, transform_w=transform_w)
    test_loader = DataLoader(
        test_dataset, batch_size=args.bz, shuffle=False, num_workers=4, drop_last=False
    )

    val_acc, AUC = validate(model, test_loader, device)
    print(colored(f"Validation accuracy: {val_acc * 100:.2f}", "red", force_color=True)) if acc is True \
        else print(colored(f"Validation AUC: {AUC * 100:.2f}", "green", force_color=True))

# ===== Evaluate at the two target folders =====
evaluate_one_folder(acc_results_dir, acc=True)
evaluate_one_folder(auc_results_dir, acc=False)
