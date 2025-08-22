# viz.py: Grad-CAM Visualization for SAM-based Segmentation
#!/usr/bin/env python3
import os
import argparse
import random
import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import cv2

# =============================================================================
# Utilities imports
# =============================================================================
from utils.config import get_config
from utils.data_us import JointTransform2D, ImageToImage2D_Test
from models.model_dict import get_model
from utils.generate_prompts import get_click_prompt
from utils.loss_functions.sam_loss import get_criterion
from utils.evaluation import visual_segmentation_sets_with_pt

# =============================================================================
# Helper: Set random seeds for reproducibility
# =============================================================================
def set_seed(seed: int = 300):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

# =============================================================================
# Utility: Get a module by its attr path (dot-separated)
# =============================================================================
def get_module_by_name(model: torch.nn.Module, name: str) -> torch.nn.Module:
    parts = name.split('.')
    mod = model
    for p in parts:
        # list index
        if p.isdigit() and isinstance(mod, (list, tuple)):
            mod = mod[int(p)]
        elif hasattr(mod, p):
            mod = getattr(mod, p)
        else:
            raise ValueError(f"Module has no attribute '{p}' in path '{name}'")
    return mod

# =============================================================================
# Grad-CAM implementation
# =============================================================================
class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_handles = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.hook_handles.append(self.target_layer.register_forward_hook(forward_hook))
        # use full backward hook to capture all gradients
        full_bkw = lambda module, grad_in, grad_out: backward_hook(module, grad_in, grad_out)
        self.hook_handles.append(self.target_layer.register_full_backward_hook(full_bkw))

    def remove_hooks(self):
        for handle in self.hook_handles:
            handle.remove()

    def __call__(self, input_tensor: torch.Tensor, prompt, class_idx: int = None) -> np.ndarray:
        # Forward/backward pass
        with torch.enable_grad():
            output = self.model(input_tensor, prompt)
            mask_logits = output.get('masks', output) if isinstance(output, dict) else output
            mask_probs = torch.sigmoid(mask_logits)
            score = mask_probs[:, class_idx].sum() if class_idx is not None else mask_probs[:, 0].sum()
            self.model.zero_grad()
            score.backward(retain_graph=True)

        if self.gradients is None or self.activations is None:
            raise RuntimeError("GradCAM hooks did not capture activations/gradients. Check --target-layer.")

        weights = self.gradients.mean(dim=(2,3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.nn.functional.relu(cam)
        cam = torch.nn.functional.interpolate(cam, size=input_tensor.shape[2:], mode='bilinear', align_corners=False)

        cam_np = cam.squeeze().cpu().numpy()
        cam_np = (cam_np - cam_np.min()) / (cam_np.max() - cam_np.min() + 1e-8)
        return cam_np

# =============================================================================
# Modified eval with Grad-CAM
# =============================================================================
def eval_mask_slice2(valloader, model, opt, args, gradcam: GradCAM):
    model.eval()
    os.makedirs(args.save_dir, exist_ok=True)

    for batch_idx, datapack in enumerate(valloader):
        imgs = Variable(datapack['image'].to(dtype=torch.float32, device=opt.device))
        pt = get_click_prompt(datapack, opt)
        image_paths = datapack['img_path']

        with torch.no_grad():
            pred = model(imgs, pt)
        predict = torch.sigmoid(pred['masks']).detach().cpu().numpy()
        seg = predict[:, 0] > 0.5

        for j in range(seg.shape[0]):
            if opt.visual:
                visual_segmentation_sets_with_pt(seg[j:j+1], image_paths[j], opt, pt[0][j:j+1])

            img_input = imgs[j:j+1]
            img_input.requires_grad = True
            prompt_input = (pt[0][j:j+1], pt[1][j:j+1]) if isinstance(pt, tuple) else pt
            cam = gradcam(img_input, prompt_input)

            heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
            input_img = img_input[0].cpu().detach().numpy().squeeze()
            input_img = (input_img - input_img.min()) / (input_img.max() - input_img.min() + 1e-8)
            input_img = np.uint8(255 * input_img)
            input_img = cv2.cvtColor(input_img, cv2.COLOR_GRAY2BGR)

            overlay = cv2.addWeighted(input_img, 0.5, heatmap, 0.5, 0)
            base = os.path.basename(image_paths[j]).replace('.png','')
            save_path = os.path.join(args.save_dir, f"{base}.jpg")
 
            cv2.imwrite(save_path, overlay)
            # print(f"Saved Grad-CAM to {save_path}")


# =============================================================================
# Main script
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description='Visualize Grad-CAM for SAM segmentation')
    parser.add_argument('--modelname', default='SAMUS', type=str, help='type of model, e.g., SAM, SAMFull, SAMHead, MSA, SAMed, SAMUS...')
    parser.add_argument('-encoder_input_size', type=int, default=256, help='the image size of the encoder input, 1024 in SAM and MSA, 512 in SAMed, 256 in SAMUS') 
    parser.add_argument('-low_image_size', type=int, default=128, help='the image embedding size, 256 in SAM and MSA, 128 in SAMed and SAMUS') 
    parser.add_argument('--task', default='BUSI', help='task or dataset name')
    parser.add_argument('--vit_name', type=str, default='vit_b', help='select the vit model for the image encoder of sam')
    parser.add_argument('--sam_ckpt', type=str, default='checkpoints/sam_vit_b_01ec64.pth', help='Pretrained checkpoint of SAM')
    parser.add_argument('--batch_size', type=int, default=1, help='batch_size per gpu') # 8 # SAMed is 12 bs with 2n_gpu and lr is 0.005
    parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
    parser.add_argument('--base_lr', type=float, default=0.0001, help='segmentation network learning rate, 0.005 for SAMed, 0.0001 for MSA') #0.0006
    parser.add_argument('--warmup', type=bool, default=False, help='If activated, warp up the learning from a lower lr to the base_lr') # True
    parser.add_argument('--warmup_period', type=int, default=250, help='Warp up iterations, only valid whrn warmup is activated')
    parser.add_argument('-keep_log', type=bool, default=False, help='keep the loss&lr&dice during training or not')

    parser.add_argument("--test_dir", default="/mnt/HDD2/tuong/class-aware/segmentation/prompt_full_step/checkpoints_final/cp_BUSBRA2BUSI/results/results_11.0k")
    parser.add_argument('--source_test_dir', default='../../../dataset/BUSBRA/valid.txt')
    parser.add_argument('--save_dir', default="gradcam/UIStyler_BUSBRA2BUSI")
    parser.add_argument('--target-layer', type=str, default="mask_decoder.output_upscaling.4")
    
    args = parser.parse_args()
    

    opt = get_config(args.task)
    opt.mode = "train"
    opt.visual = True
    opt.batch_size = args.batch_size
    opt.data_path = args.test_dir
    device = torch.device(opt.device)

    print("task", args.task, "checkpoints:", opt.load_path)
    set_seed()

    tf_val = JointTransform2D(
        img_size=args.encoder_input_size,
        low_img_size=args.low_image_size,
        ori_size=opt.img_size,
        crop=opt.crop,
        p_flip=0,
        color_jitter_params=None,
        long_mask=True
    )
    val_dataset = ImageToImage2D_Test(
        args.test_dir,
        args.source_test_dir,
        tf_val,
        img_size=args.encoder_input_size,
        class_id=1
    )
    valloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = get_model(args.modelname, args=args, opt=opt).to(device)
    checkpoint = torch.load(opt.load_path)
    state_dict = {k[7:]:v for k,v in checkpoint.items()} if any(k.startswith('module.') for k in checkpoint) else checkpoint
    model.load_state_dict(state_dict)

    criterion = get_criterion(modelname=args.modelname, opt=opt)

    target_layer = get_module_by_name(model, args.target_layer)
    gradcam = GradCAM(model, target_layer)
    print(f"Using target layer '{args.target_layer}' for Grad-CAM.")

    eval_mask_slice2(valloader, model, opt, args, gradcam)
    gradcam.remove_hooks()

    print("✅ Grad-CAM visualization complete. Check the `grad_cam/` folder.")

if __name__ == '__main__':
    main()
