import argparse
import os
from tqdm import tqdm
import numpy as np
import torch_fidelity

def parse_args():
    parser = argparse.ArgumentParser(description="Find best KID score across iterations")
    parser.add_argument("--source", default="BUSBRA")
    parser.add_argument("--target", default="BUSI")
    parser.add_argument("--checkpoint_dir", default="./checkpoints")
    parser.add_argument("--dataset_dir", default="./dataset")
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--kid-subset-size", type=int, default=49)
    parser.add_argument("--kid-subsets", type=int, default=10)
    parser.add_argument("--start-k", type=float, default=1.0)
    parser.add_argument("--end-k", type=float, default=100.0)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()

def main():
    args = parse_args()

    # Set GPU visibility
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    task = f"{args.source}2{args.target}"

    # Paths for source and target images
    source_root_path = os.path.join(args.checkpoint_dir, f"cp_{task}", "results")
    target_path = os.path.join(args.dataset_dir, f"{args.target}", "train", "imgs")

    print(f"source_root_path: {source_root_path}")
    print(f"target_path: {target_path}")

    eval_args = {
        "kid": True,
        "kid_subset_size": args.kid_subset_size,
        "kid_subsets": args.kid_subsets,
        "verbose": args.verbose,
        "gpu": args.gpu,
    }

    best_KID = 1000.0
    iter_KID = 0.0

    # Loop over results_{k}k folders
    for i in tqdm(np.arange(args.start_k, args.end_k, 1.0)):
        k_folder = f"results_{i}k"
        source_path = os.path.join(source_root_path, k_folder)

        if not os.path.isdir(source_path):
            continue  # skip if folder does not exist

        # Compute KID metric
        metric_dict_AB = torch_fidelity.calculate_metrics(
            input1=source_path,
            input2=target_path,
            **eval_args
        )
        KID = metric_dict_AB["kernel_inception_distance_mean"] * 100.0

        # Keep track of best KID
        if KID < best_KID:
            best_KID = KID
            iter_KID = i

    if best_KID < 1000.0:
        print(f"- Best KID at {iter_KID}: {best_KID:.2f}")
    else:
        print("No valid results found.")

if __name__ == "__main__":
    main()
