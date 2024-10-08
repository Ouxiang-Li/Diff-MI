"""
Step-2: Iterative Image Reconstruction
"""

import warnings
warnings.filterwarnings("ignore")

import os, sys, pdb
import math
import datetime
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia.augmentation as K
from torch.utils.data import DataLoader
from robustness import model_utils
from tqdm import tqdm

from utils.unet import UNetModel
from utils.diffusion import GaussianDiffusion
from utils.utils import *

sys.path.append(os.getcwd())
from models.classifier import VGG16, IR152, FaceNet64, FaceNet


parser = argparse.ArgumentParser()
# Sampling Configuration
parser.add_argument('--dataset', type=str, default='celeba', help='celeba | ffhq | facescrub')
parser.add_argument('--target', type=str, default='VGG16', help='FaceNet64 | IR152 | VGG16')
parser.add_argument('--path_D', type=str, default=None, required=True)
parser.add_argument('--steps', type=int, default=30)
parser.add_argument('--w', type=float, default=3.0)
parser.add_argument('--ddim_step', type=int, default=100)
parser.add_argument('--repeat_times', type=int, default=1)
parser.add_argument('--label_num', type=int, default=300)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--aug_times', type=int, default=4)
parser.add_argument('--k', type=int, default=20)
parser.add_argument('--alpha', type=float, default=1.0)
parser.add_argument('--seed', type=int, default=42)
# Evaluation Configuration
parser.add_argument('--cal_fid', default=True, action='store_true')
parser.add_argument('--cal_knn', default=True, action='store_true')
args = parser.parse_args()


def main():

    set_random_seed(args.seed)
    device = torch.device('cuda')

    now = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
    save_name = f"{args.target}_{now}_{args.steps}_{args.w:.1f}_{args.ddim_step}_" + \
                f"k{args.k}_A{args.aug_times}_{args.repeat_times}x{args.label_num}_bs{args.batch_size}"
    save_path = os.path.join('3_attack/logs', args.dataset, save_name)
    os.makedirs(save_path, exist_ok=True)
    print('Saving to', save_path)

    log_file = f"evaluation_{now}.txt"
    Tee(os.path.join(save_path, log_file), 'w')
    print(args)

    # Load models
    diff_net, T, E, PGD_model = load_models(args.target, args.path_D, device=device)

    recon_list, success_img_list, success_label_list = [], [], []
    labels = torch.cat([torch.randperm(args.label_num) for _ in range(args.repeat_times)]).to(device)
    label_dataset = DataLoader(labels, batch_size=args.batch_size, shuffle=False)
    batch_num = math.ceil(len(labels) / args.batch_size) - 1

    args.aug = K.container.ImageSequential(
        K.RandomHorizontalFlip(),
        K.ColorJitter(brightness=0.2, p=0.5),
        K.RandomGaussianBlur((7, 7), (3, 3), p=0.5),
    )

    for i, classes in enumerate(label_dataset):

        recon_imgs = Iterative_Image_Reconstruction(args, diff_net=diff_net,  classifier=T, classes=classes, 
                                                    iter=i, batch_num=batch_num, device=device,).clamp(0,1)
        PGD_kwargs = {
            'constraint':'2', 'eps': 0.5, 'step_size': 0.1, 'iterations': 10, 
            'random_start': True, 'targeted': True, 'use_best': True, 'with_image': True, 'make_adv': True,
        }
        img_translated = PGD_model(recon_imgs, target=classes, **PGD_kwargs)[-1].clamp(0,1)
        _, _, idx = calc_acc(E, K.Resize((112, 112))(img_translated), classes, with_success=True)

        recon_list.append(img_translated)
        success_img_list.append(img_translated[idx])
        success_label_list.append(classes[idx])

    recon_list = torch.cat(recon_list)
    success_img_list = torch.cat(success_img_list)
    success_label_list = torch.cat(success_label_list)

    acc1, acc5, var1, var5 = calc_acc_std(recon_list, labels, E, args.label_num)
    print(f"Final Top1: {acc1:.2%} ± {var1:.2%}, Top5: {acc5:.2%} ± {var5:.2%}")

    # Save Reconstructed Images
    save_tensor_to_image(recon_list, labels, f'./{save_path}/Diff-MI/all_imgs')
    save_tensor_to_image(success_img_list, success_label_list, f'./{save_path}/Diff-MI/success_imgs')
    print(f'Saved {args.repeat_times}x{args.label_num} generated images.')
        
    if args.cal_fid:
        print("===> Start Calculating FID <===")
        calc_pytorch_fid(f'./{save_path}/Diff-MI/all_imgs')

    if args.cal_knn:
        print("===> Start Calculating KNN Dist <===")
        calc_knn(success_img_list, success_label_list, E=E, device=device)


def Iterative_Image_Reconstruction(
    args,
    diff_net, 
    classifier, 
    classes, 
    iter=None, 
    batch_num=None, 
    device='cuda'
):
    
    p_reg = torch.load(f"data/p_reg/{args.dataset}_{args.target}_p_reg.pt")

    diffusion = GaussianDiffusion(T=1000, schedule='linear')
    model = InferenceModel(batch_size=classes.shape[0]).to(device)
    model.train()

    # Inference procedure steps
    steps = args.steps
    opt = torch.optim.Adamax(model.parameters(), lr=1.0)
    scheduler = torch.optim.lr_scheduler.LinearLR(opt, start_factor=1.0, end_factor=1.0, total_iters=steps)

    norm_track = 0

    print(f"--------------------- Epoch {iter}/{batch_num} ---------------------")
    bar = tqdm(range(steps))
    for i, _ in enumerate(bar): 

        bar.set_description(f'Epoch {iter}/{batch_num}')

        # Select t
        t = ((steps-i)/1.5 + (steps-i)/3*math.cos(3/(10*math.pi)*i))/steps*800 + 200 # Linearly decreasing + cosine
        t = np.array([t + np.random.randint(-50, 51) for _ in range(1)]).astype(int) # Add noise to t
        t = np.clip(t, 1, diffusion.T)

        # Denoise
        sample_img = model.encode()
        xt, epsilon = diffusion.sample(sample_img, t)
        t = torch.from_numpy(t).float().view(1)
        eps = diff_net(xt.float(), t.to(device), classes)
        nonEps = diff_net(xt.float(), t.to(device), torch.ones_like(classes) * (diff_net.num_classes - 1))
        epsilon_pred = args.w * eps - (args.w - 1) * nonEps

        # Compute diffusion loss: ||epsilon - epsilon_theta||^2
        loss = 1 * F.mse_loss(epsilon_pred, epsilon)

        opt.zero_grad()
        loss.backward()

        with torch.no_grad():
            grad_norm = torch.linalg.norm(model.img.grad)
            if i > 0:
                alpha = 0.5
                norm_track = alpha*norm_track + (1-alpha)*grad_norm
            else:
                norm_track = grad_norm
        opt.step()

        attr_input_batch = []
        for _ in range(args.aug_times):
            attr_input = model.encode()
            attr_input = args.aug(attr_input).clamp(-1,1)
            attr_input_batch.append(attr_input)

        attr_input_batch = torch.cat(attr_input_batch, dim=0)
        feats, logits = classifier.forward((attr_input_batch+1)/2)

        # topk loss + p_reg loss
        loss = topk_loss(logits, classes.repeat(args.aug_times), k=args.k) + \
               args.alpha * p_reg_loss(feats, classes.repeat(args.aug_times), p_reg)

        opt.zero_grad()
        loss.backward()

        # Clip attribute loss gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.25 * norm_track)
        opt.step()
        scheduler.step()

    with torch.no_grad():
        if args.ddim_step == 1000:
            t = np.array([args.ddim_step]).astype(int)
            xt = model.encode()
        else:
            t = np.array([args.ddim_step]).astype(int)
            xt, _ = diffusion.sample(model.encode(), t)
        fine_tuned = diffusion.inverse_ddim(diff_net, x=xt, start_t=t[0], w=args.w, y=classes, device=device)

    return (fine_tuned + 1) / 2


def load_models(target, ckpt_D, device='cuda'):

    D = UNetModel(
            image_size=64,
            in_channels=3,
            model_channels=128,
            out_channels=3,
            num_res_blocks=3,
            attention_resolutions=(4,8),
            dropout=0.0,
            channel_mult=(1, 2, 3, 4),
            num_classes=1001,
            use_checkpoint=False,
            use_fp16=False,
            num_heads=4,
            num_head_channels=-1,
            num_heads_upsample=-1,
            use_scale_shift_norm=True,
            resblock_updown=False,
            use_new_attention_order=False,
        ).to(device)

    D.load_state_dict(torch.load(ckpt_D))
    D.eval()
    print(f'1. Loaded Diffusion Model from {ckpt_D}')

    if target.startswith("VGG16"):
        T = VGG16(1000)
        path_T = 'assets/checkpoints/target_model/VGG16_88.26.tar'
    elif target.startswith('IR152'):
        T = IR152(1000)
        path_T = 'assets/checkpoints/target_model/IR152_91.16.tar'
    elif target == "FaceNet64":
        T = FaceNet64(1000)
        path_T = 'assets/checkpoints/target_model/FaceNet64_88.50.tar'
    T = torch.nn.DataParallel(T).to(device)
    ckp_T = torch.load(path_T)
    T.load_state_dict(ckp_T['state_dict'])
    T.eval()
    print(f'2. Loaded Target Classifier from {path_T}')

    E = FaceNet(1000)
    E = torch.nn.DataParallel(E).to(device)
    path_E = 'assets/checkpoints/evaluate_model/FaceNet_95.88.tar'
    ckp_E = torch.load(path_E)
    E.load_state_dict(ckp_E['state_dict'])
    E.eval()
    print(f'3. Loaded eval classifier from {path_E}')

    class mean_and_std():
        def __init__(self):
            self.mean = torch.tensor([0.0, 0.0, 0.0])
            self.std = torch.tensor([1.0, 1.0, 1.0])
    PGD_model, _ = model_utils.make_and_restore_model(arch=T, dataset=mean_and_std())
    PGD_model.eval()
    print('4. Loaded PGD Model')

    return D, T, E, PGD_model


def topk_loss(out, iden, k):
    iden = iden.unsqueeze(1)
    real = out.gather(1, iden).squeeze(1)
    if k == 0: return -1 * real.mean()
    tmp_out = torch.scatter(out, dim=1, index=iden, src=-torch.ones_like(iden)*1000.0)
    margin = torch.topk(tmp_out, k=k)[0]
    return -1 * real.mean() + margin.mean()


def p_reg_loss(featureT, classes, p_reg):
    fea_reg = p_reg[classes]
    assert featureT.shape == fea_reg.shape
    return F.mse_loss(featureT, fea_reg)


class InferenceModel(nn.Module):
    def __init__(self, x=None, batch_size=16):
        super(InferenceModel, self).__init__()
        if x is None:
            self.img = nn.Parameter(torch.randn(batch_size, 3, 64, 64))
        else:
            self.img = nn.Parameter(x)
        self.img.requires_grad = True
    def encode(self):
        return self.img


if __name__ == '__main__':
    main()