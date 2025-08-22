import os
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import timm
from tqdm import tqdm
from termcolor import colored

from log_utils.utils import ReDirectSTD
from utils.dataloader import ImageList
from utils.preprocess import train_transform, val_transform
from utils.utils import validate

from torch.optim.lr_scheduler import ExponentialLR

import random
import numpy as np

from utils.optimizer import inv_lr_scheduler

def main():
    """Train, validate, and test the classification model."""
    parser = argparse.ArgumentParser(description="Classification")
    # Data parameters
    parser.add_argument("--train_dir", type=str, default="../../dataset/BUSBRA/train")
    parser.add_argument("--val_dir", type=str, default="../../dataset/BUSBRA/valid")
    parser.add_argument("--test_dir", type=str, default="../../dataset/BUSI/valid")
    parser.add_argument("--resize_size", type=int, default=256,
                        help="Resize size")
    parser.add_argument("--crop_size", type=int, default=224,
                        help="Crop size")
    # Model parameters
    parser.add_argument("--model", type=str, default="vit")
    # Training parameters    
    parser.add_argument("--bz", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="LR")
    parser.add_argument("--epochs", type=int, default=50, help="Epochs")
    # Save directory parameters
    parser.add_argument("--save_dir", type=str,
                        default="checkpoints3/",
                        help="Save dir")
    parser.add_argument("--seed", type=int, default=1234)

    # Device parameters
    parser.add_argument("--device", default="cuda:6", help="Device")

    args = parser.parse_args()

    # Set random seed
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Prepare save directory and logging
    save_dir = f"{args.save_dir}{args.model}_{os.path.basename(args.train_dir.replace('/train',''))}"
    os.makedirs(save_dir, exist_ok=True)
    best_model_path = os.path.join(save_dir, "best.pth")
    log_dir = os.path.join(save_dir, "logs.txt")
    ReDirectSTD(log_dir, "stdout", True)

    # Set device for training
    device = torch.device(args.device)

    # Load datasets using custom dataloader classes
    print(colored(f"Loading train datasets from {args.train_dir}", color="blue", force_color=True))
    print(colored(f"Loading valid datasets from {args.val_dir}", color="blue", force_color=True))
    print(colored(f"Loading test datasets from {args.test_dir}", color="blue", force_color=True))
    resize_size = args.resize_size
    crop_size = args.crop_size
    train_dataset = ImageList(args.train_dir, transform_w=train_transform(resize_size, crop_size))
    val_dataset = ImageList(args.val_dir, transform_w=val_transform(resize_size, crop_size))
    test_dataset = ImageList(args.test_dir, transform_w=val_transform(resize_size, crop_size))
    num_classes = train_dataset.num_classes

    print(colored(f"Length Dataset {len(train_dataset)}", color="yellow", force_color=True))
    print("Unique classes found:", train_dataset.classes)
    print("Num classes:", train_dataset.num_classes)

    train_loader = DataLoader(train_dataset, batch_size=args.bz, shuffle=True, num_workers=4, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.bz, shuffle=False, num_workers=4, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=args.bz, shuffle=False, num_workers=4, drop_last=False)

    ##### Loading Model #####
    print(colored(f"Loading model: {args.model}", color="red",
                  force_color=True))
    if args.model == 'resnet34':
        model = timm.create_model('timm/resnet34', pretrained=True, num_classes=num_classes)
    elif args.model == 'resnet50':
        model = timm.create_model('timm/resnet50', pretrained=True, num_classes=num_classes)
    elif args.model == 'resnet101':
        model = timm.create_model('timm/resnet101', pretrained=True, num_classes=num_classes)
    elif args.model == 'vit':
        model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=num_classes)
    else:
        raise ValueError(f"Model {args.model} not supported")

    print(model.default_cfg)

    model.to(device)
    model.train()

    # Loss function and optimizer    
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0005, nesterov=True)
    # scheduler = ExponentialLR(optimizer, gamma=0.9)

    best_valid_acc, best_valid_AUC = 0.0, 0.0
    best_test_acc, best_test_AUC = 0.0, 0.0
    epochs = args.epochs
    max_iters = len(train_loader)

    param_lr = []
    for param_group in optimizer.param_groups:
        param_lr.append(param_group["lr"])

    iters = 0
    # Training loop
    for epoch in tqdm(range(epochs), desc="Training epochs"):
        for step, batch_train in enumerate(train_loader):
            ##### Scheduler #####
            optimizer = inv_lr_scheduler(
            param_lr, optimizer, iters, init_lr=args.lr
            )
            lr = optimizer.param_groups[0]["lr"]

            train_w = batch_train["img_w"].to(device)
            train_labels = batch_train["target"].to(device)
            
            outputs = model(train_w)
            loss = criterion(outputs, train_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iters += 1
            if step % 20 == 0 or step == max_iters - 1:
                print(
                    "Epoch {} Iters: ({}/{}) \t Loss Sup = {:<10.6f} \t "
                    "learning rate = {:<10.6f}".format(
                        epoch, step, max_iters, loss.item(), lr
                    )
                )
        # scheduler.step()

        model.eval() 
        val_acc, val_AUC = validate(model, val_loader, device)
        print(f"Validation accuracy: {val_acc * 100:.2f}% \t AUC: {val_AUC:.2f} at epoch {epoch}")

        test_acc, test_AUC = validate(model, test_loader, device)
        print(colored(f"Test Accuracy: {test_acc * 100:.2f} \t AUC: {test_AUC * 100:.2f}", color="green", force_color=True))

        if val_AUC > best_valid_AUC:
            best_valid_AUC = val_AUC

        if val_acc > best_valid_acc:
            best_valid_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved best model with val acc: {best_valid_acc * 100:.2f}")
            
            counter = 0
        else:
            counter += 1

        log_str = (
            "===================================================================\n"
            f"Best Validation Accuracy: {best_valid_acc * 100:.2f}% \t Best AUC: {best_valid_AUC * 100:.2f}\n"
            "==================================================================="
        )        
        print(colored(log_str, color="red", force_color=True))
        
        model.train()
        ##### Early Stopping #####
        # if counter > 5:
        #     break

if __name__ == "__main__":
    main()
