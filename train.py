import argparse
import os
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from tqdm import tqdm
from models.transformer import SourceEncoder, TargetEncoder, StyleTrans
import models.network as network 
from models.vgg import vgg
from models.decoder import decoder

from torchvision.utils import save_image

from utils.dataloader import ImageDataset, train_transform, test_transform, mask_transform
from utils.scheduler import adjust_learning_rate, warmup_learning_rate

from termcolor import colored
from log_utils.utils import ReDirectSTD

from torch.utils.data import DataLoader
from utils.utils import set_seed
import shutil

from classification.utils import validate
from classification.dataloader import ImageList
from classification.preprocess import val_transform
import timm

from models.prompt_generator import PromptGenerator

import numpy
import random

##### Parser #####
parser = argparse.ArgumentParser(description='Medical Image Style Transfer')

### Dataset ###
parser.add_argument('--root_dir', type=str, default='./dataset')
parser.add_argument('--source', type=str, default='BUSBRA',
                    help='source domain')
parser.add_argument('--target', type=str, default='BUSI',
                    help='target domain')
parser.add_argument('--img_size', type=int, default=256)
parser.add_argument('--patch_size', type=int, default=8)

### Training parameters ###
parser.add_argument('--max_iters', type=int, default=50000)
parser.add_argument('--lr', type=float, default=5e-4)
parser.add_argument('--lr_decay', type=float, default=1e-5)
parser.add_argument('--batch_size', dest='bz', type=int, default=8)

### VGG model ###
parser.add_argument('--vgg_weights', type=str, default='./vgg_normalised.pth')
parser.add_argument('--cls_weights', type=str, default='./downstream/classification/checkpoints_final')

### Output ###
parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--log_interval', type=int, default=20)
parser.add_argument('--save_interval', type=int, default=100)
parser.add_argument('--test_interval', type=int, default=1000)

parser.add_argument('--device', type=str, default='cuda:0')

parser.add_argument('--mask', dest='mask', action='store_true', help='Use mask for training (default)')

args = parser.parse_args()
###########################

##### Set random seed #####
set_seed(args.seed)

### Config log file ###
args.save_dir = f"checkpoints/cp_{args.source}2{args.target}"

os.makedirs(args.save_dir, exist_ok=True)
os.makedirs(args.save_dir+"/test", exist_ok=True)

log_path = os.path.join(args.save_dir, "log" + ".txt")
re = ReDirectSTD(log_path, "stdout", True)

log_tensorboard = log_path.replace(".txt","")
if os.path.exists(log_tensorboard):
    shutil.rmtree(log_tensorboard)
writer = SummaryWriter(log_dir=log_tensorboard)

result_path = os.path.join(args.save_dir, 'results')
os.makedirs(result_path, exist_ok=True)

##### Data loading #####
source_dir = os.path.join(args.root_dir, args.source, "train")
target_dir = os.path.join(args.root_dir, args.target, "train")
source_test_dir = os.path.join(args.root_dir, args.source, "valid")

log_str = 'Dataset Information: %s to %s' % (args.source, args.target)
print(colored(log_str, 'green', force_color=True))
log_str = f"Source: {source_dir} Target: {target_dir}"
print(colored(log_str, 'yellow', force_color=True))
log_str = f"Source test: {source_test_dir}"
print(colored(log_str, 'yellow', force_color=True))

### Random dataloader ####
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(args.seed)

### Training Dataset ###
source_dataset = ImageDataset(source_dir, transform=train_transform(args.img_size), mask_transform=mask_transform(args.img_size//args.patch_size))
target_dataset = ImageDataset(target_dir, transform=train_transform(args.img_size), mask_transform=mask_transform(args.img_size//args.patch_size))
source_loader = DataLoader(source_dataset, batch_size=args.bz, num_workers=4, shuffle=True, drop_last=True,
                            worker_init_fn=seed_worker, generator=g)
target_loader = DataLoader(target_dataset, batch_size=args.bz, num_workers=4, shuffle=True, drop_last=True,
                            worker_init_fn=seed_worker, generator=g)

log_str = "Source dataset length: %d, Target dataset length: %d" % (len(source_dataset), len(target_dataset))
print(colored(log_str, 'blue', force_color=True))

### Get Len (total_imgs / batch_size) ###
len_source = len(source_loader)
len_target = len(target_loader)

### Test Dataset ###
source_test_dataset = ImageDataset(source_test_dir, transform=test_transform(args.img_size), mask_transform=mask_transform(args.img_size//args.patch_size))
target_test_dataset = ImageDataset(target_dir, transform=test_transform(args.img_size), mask_transform=mask_transform(args.img_size//args.patch_size))
source_test_loader = DataLoader(source_test_dataset, batch_size=args.bz, num_workers=4, shuffle=False, drop_last=False,
                                worker_init_fn=seed_worker, generator=g)
target_test_loader = DataLoader(target_test_dataset, batch_size=args.bz, num_workers=4, shuffle=False, drop_last=True,
                                worker_init_fn=seed_worker, generator=g)
len_test_source = len(source_test_loader)
len_test_target = len(target_test_loader)
print(colored(f"Source test dataset length: {len(source_test_dataset)}, Target test dataset length: {len(target_test_dataset)}", 'cyan', force_color=True))

##### Model loading #####
device = torch.device(args.device)
### Load VGG model ###
print(colored("Loading VGG model...", 'red', force_color=True))
vgg.load_state_dict(torch.load(args.vgg_weights, map_location=device))
vgg = nn.Sequential(*list(vgg.children())[:44])

### Classification Downstream ###
cls_model = timm.create_model('timm/vit_base_patch16_224', pretrained=True, num_classes=2)
cls_weight_path = os.path.join(args.cls_weights, f"vit_{args.target}/best.pth")
print(colored(f"Loading Classification model {cls_weight_path}", 'red', force_color=True))
cls_model.load_state_dict(torch.load(cls_weight_path, map_location=device))
cls_model = cls_model.to(device)
cls_model.eval()

### Style Transfer Network ###
source_encoder = SourceEncoder(img_size=args.img_size, patch_size=args.patch_size, num_layers=3)
target_encoder = TargetEncoder(img_size=args.img_size, patch_size=args.patch_size, num_layers=3)
ST_module = StyleTrans(num_layers=3)
CPG_module = PromptGenerator(num_class=2)

network = network.Network(source_encoder, target_encoder, ST_module, CPG_module, decoder, vgg, device, args.mask)

network.train()
network.to(device)
#########################
optimizer = torch.optim.Adam([ 
                            {'params': network.source_encoder.parameters()},
                            {'params': network.target_encoder.parameters()},
                            {'params': network.ST_module.parameters()},
                            {'params': network.CPG_module.parameters()},
                            {'params': network.decoder.parameters()}
                            ], lr=args.lr)

best_acc, best_AUC = 0.0, 0.0

##### Training #####
for step in tqdm(range(args.max_iters)):
    if step < 1e4:
        warmup_learning_rate(optimizer, step, args.lr)
    else:
        adjust_learning_rate(optimizer, step, args.lr_decay)

    ### Load train data ###
    if step % len_source == 0:
        iter_source = iter(source_loader)
    if step % len_target == 0:
        iter_target = iter(target_loader)

    batch_source = next(iter_source)
    batch_target = next(iter_target)
    ########################
    Ist, loss_c, loss_s, loss_prompt = network(batch_source, batch_target)
    
    loss = loss_c + loss_s + loss_prompt
       
    optimizer.zero_grad()
    loss.sum().backward()
    optimizer.step()

    ### Print log ###
    if step % args.log_interval == 0 or step == args.max_iters - 1:
        print("Train Step: [{}/{}] Loss: {:.4f} - Content: {:.4f} - Style: {:.4f} - Prompt: {:.4f}".format(
            step, args.max_iters, 
            loss.sum().cpu().detach().numpy(), 
            loss_c.sum().cpu().detach().numpy(), 
            loss_s.sum().cpu().detach().numpy(),
            loss_prompt.sum().cpu().detach().numpy()
        ))
        
    if step % args.save_interval == 0 or step == args.max_iters - 1:
        source_imgs = batch_source["img"].to(device)
        target_imgs = batch_target["img"].to(device)

        ##### Save image #####
        output_name = '{:s}/test/{:s}{:s}'.format(
                        args.save_dir, str(step),".jpg"
                    )
        out = torch.cat((source_imgs, Ist),0)
        out = torch.cat((target_imgs, out),0)
        save_image(out, output_name)

    writer.add_scalar('loss_content', loss_c.item(), step + 1)
    writer.add_scalar('loss_style', loss_s.item(), step + 1)
    writer.add_scalar('loss_prompt', loss_prompt.item(), step + 1)
    writer.add_scalar('total_loss', loss.item(), step + 1)

    if (step + 1) % args.test_interval == 0:
        network.eval()

        ### Save results ###
        path_save = f"{result_path}/results_{(step+1)/1000}k"
        os.makedirs(path_save, exist_ok=True)
        
        iter_test_target = iter(target_test_loader)
        for i, batch_test_source in tqdm(enumerate(source_test_loader)):
            ### Get data ###
            # Source #
            source_test_paths = batch_test_source["path"]

            # Target #
            if i % len_test_target == 0:
                iter_test_target = iter(target_test_loader)

            batch_test_target = next(iter_test_target)
            ################

            ##### inference #####
            with torch.no_grad():                
                Ist, *_ = network(batch_test_source, batch_test_target)

            for idx, img in enumerate(Ist):
                img_name = source_test_paths[idx].split('/')[-1]                  

                output_name = os.path.join(path_save, img_name)
                save_image(img.cpu(), output_name)

        network.train()

        #### Testing ####
        log_str = f"Testing at {path_save}"
        print(colored(log_str, color="magenta", force_color=True))

        test_dataset = ImageList(path_save, transform_w=val_transform(256, 224)) # args.target_dir
        test_loader = DataLoader(test_dataset, batch_size=args.bz, shuffle=False, drop_last=False)
        val_acc, val_AUC = validate(cls_model, test_loader, device)
        
        if val_AUC >= best_AUC:
            best_step_AUC = step + 1
            best_AUC = val_AUC
            log_str = f"Best AUC: {best_AUC * 100:.2f}%"
            print(colored(log_str, color="red", force_color=True))

            torch.save(network.state_dict(), f"{args.save_dir}/best_network_AUC.pth")

        if val_acc >= best_acc:
            best_step = step + 1
            best_acc = val_acc
            log_str = f"Best accuracy: {best_acc * 100:.2f}%"
            print(colored(log_str, color="red", force_color=True))

            torch.save(network.state_dict(), f"{args.save_dir}/best_network_acc.pth")

        print(f"Validation accuracy: {val_acc * 100:.2f}% \t AUC: {val_AUC * 100:.2f}")
        log_str = f"Best accuracy at {best_step}: {best_acc * 100:.2f}% \t Best AUC at {best_step_AUC}: {best_AUC * 100:.2f}"
        print(colored(log_str, color="red", force_color=True))
                                                    
writer.close()
