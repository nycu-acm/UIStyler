from ast import arg
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import argparse
from pickle import FALSE, TRUE
from statistics import mode
from tkinter import image_names
import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import time
import random
from utils.config import get_config
from utils.evaluation import get_eval
from importlib import import_module

from torch.nn.modules.loss import CrossEntropyLoss
from monai.losses import DiceCELoss
from einops import rearrange
from models.model_dict import get_model
from utils.data_us import JointTransform2D, ImageToImage2D_Test
from utils.loss_functions.sam_loss import get_criterion
from utils.generate_prompts import get_click_prompt
from thop import profile
from tqdm import tqdm

def main():

    #  =========================================== parameters setting ==================================================
    parser = argparse.ArgumentParser(description='Networks')
    parser.add_argument('--modelname', default='SAMUS', type=str, help='type of model, e.g., SAM, SAMFull, SAMHead, MSA, SAMed, SAMUS...')
    parser.add_argument('-encoder_input_size', type=int, default=256, help='the image size of the encoder input, 1024 in SAM and MSA, 512 in SAMed, 256 in SAMUS') 
    parser.add_argument('-low_image_size', type=int, default=128, help='the image embedding size, 256 in SAM and MSA, 128 in SAMed and SAMUS') 
    
    parser.add_argument('--vit_name', type=str, default='vit_b', help='select the vit model for the image encoder of sam')
    parser.add_argument('--sam_ckpt', type=str, default='./downstream/segmentation/checkpoints2/sam_vit_b_01ec64.pth', help='Pretrained checkpoint of SAM')
    parser.add_argument('--batch_size', type=int, default=8, help='batch_size per gpu') # 8 # SAMed is 12 bs with 2n_gpu and lr is 0.005
    parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
    parser.add_argument('--base_lr', type=float, default=0.0001, help='segmentation network learning rate, 0.005 for SAMed, 0.0001 for MSA') #0.0006
    parser.add_argument('--warmup', type=bool, default=False, help='If activated, warp up the learning from a lower lr to the base_lr') # True
    parser.add_argument('--warmup_period', type=int, default=250, help='Warp up iterations, only valid whrn warmup is activated')
    parser.add_argument('-keep_log', type=bool, default=False, help='keep the loss&lr&dice during training or not')

    parser.add_argument('--root_dir', type=str, default='./checkpoints')
    parser.add_argument('--source_name', type=str, default='BUSBRA')
    parser.add_argument('--task', default='BUSI', help='task or dataset name')
    
    args = parser.parse_args()
    opt = get_config(args.task)  # please configure your hyper-parameter
    print("task", args.task, "checkpoints:", opt.load_path)

    opt.mode = "train"
    opt.visual = False
    opt.modelname = args.modelname
    device = torch.device(opt.device)

     #  =============================================================== add the seed to make sure the results are reproducible ==============================================================

    seed_value = 300 # the number of seed
    np.random.seed(seed_value)  # set random seed for numpy
    random.seed(seed_value)  # set random seed for python
    os.environ['PYTHONHASHSEED'] = str(seed_value)  # avoid hash random
    torch.manual_seed(seed_value)  # set random seed for CPU
    torch.cuda.manual_seed(seed_value)  # set random seed for one GPU
    torch.cuda.manual_seed_all(seed_value)  # set random seed for all GPU
    torch.backends.cudnn.deterministic = True  # set random seed for convolution

    #  =========================================================================== model and data preparation ============================================================================
    
    # register the sam model
    opt.batch_size = args.batch_size * args.n_gpu

    opt.data_path = os.path.join(args.root_dir, f"cp_{args.source_name}2{args.task}", "results")
    source_test_dir = f"./dataset/{args.source_name}/valid.txt"
    print("test_dir", opt.data_path)

    tf_val = JointTransform2D(img_size=args.encoder_input_size, low_img_size=args.low_image_size, ori_size=opt.img_size, crop=opt.crop, p_flip=0, color_jitter_params=None, long_mask=True)
    
    dataloader_list = []

    for i in np.arange(1.0, 50.0):
        results_dir = os.path.join(opt.data_path, f"results_{i}k")
        val_dataset = ImageToImage2D_Test(results_dir, source_test_dir, tf_val, img_size=args.encoder_input_size, class_id=1)  # return image, mask, and filename
        valloader = DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=8, pin_memory=True)
        dataloader_list.append(valloader)

    if args.modelname=="SAMed":
        opt.classes=2
    model = get_model(args.modelname, args=args, opt=opt)
    model.to(device)
    model.train()

    checkpoint = torch.load(opt.load_path)
    #------when the load model is saved under multiple GPU
    new_state_dict = {}
    for k,v in checkpoint.items():
        if k[:7] == 'module.':
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    model.load_state_dict(new_state_dict)
    
    criterion = get_criterion(modelname=args.modelname, opt=opt)
#  ========================================================================= begin to evaluate the model ============================================================================

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total_params: {}".format(pytorch_total_params))
    input = torch.randn(1, 1, args.encoder_input_size, args.encoder_input_size).cuda()
    points = (torch.tensor([[[1, 2]]]).float().cuda(), torch.tensor([[1]]).float().cuda())
    flops, params = profile(model, inputs=(input, points), )
    print('Gflops:', flops/1000000000, 'params:', params)

    model.eval()

    ###### Evaluate ######
    best_dice = 0
    best_iou = 0
    best_dice_iter = 0
    best_iou_iter = 0

    for idx, loader in tqdm(enumerate(dataloader_list), total=len(dataloader_list)):
        dices, mean_dice, mean_hdis, mean_ious, val_losses = get_eval(loader, model, criterion=criterion, opt=opt, args=args)

        mapped_idx = idx+1
        if mean_dice > best_dice:
            best_dice = mean_dice
            best_dice_iter = mapped_idx
        if mean_ious > best_iou:
            best_iou = mean_ious
            best_iou_iter = mapped_idx

    print(f"mean dice: {best_dice:.2f} at {best_dice_iter} \t mean iou: {best_iou:.2f} at {best_iou_iter}")

if __name__ == '__main__':
    main()
