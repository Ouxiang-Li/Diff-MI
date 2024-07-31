import warnings
warnings.filterwarnings("ignore")

import os, sys, pdb
import argparse
import torch
import torchvision
from PIL import Image

from utils.utils import *
sys.path.append(os.getcwd())
from models.classifier import FaceNet

'''
CUDA_VISIBLE_DEVICES=0 python 3_attack/evaluate.py \
    --eval_path {path of recontructed images} \
    --cal_acc --cal_fid --cal_knn \
    --cal_piq --cal_lpips \
    --cal_PRCD
'''

# ====================== Configuration =======================

parser = argparse.ArgumentParser()
parser.add_argument('--eval_path', type=str)
parser.add_argument('--cal_acc', default=False, action='store_true')
parser.add_argument('--cal_fid', default=False, action='store_true')
parser.add_argument('--cal_kid', default=False, action='store_true')
parser.add_argument('--cal_knn', default=False, action='store_true')
parser.add_argument('--cal_PRCD', default=False, action='store_true')
parser.add_argument('--cal_piq', default=False, action='store_true')
parser.add_argument('--cal_lpips', default=False, action='store_true')
parser.add_argument('--all2success', default=True, action='store_true')

# ============================================================

def main():

    args = parser.parse_args()
    device = torch.device('cuda')

    E = FaceNet(1000)
    E = torch.nn.DataParallel(E).to(device)
    path_E = 'assets/checkpoints/evaluate_model/FaceNet_95.88.tar'
    ckp_E = torch.load(path_E)
    E.load_state_dict(ckp_E['state_dict'])
    E.eval()
    print(f'Loaded eval classifier from {path_E}')

    print(f"===> Evaluation on {args.eval_path} <===")
    eval_path = os.path.join(args.eval_path, "all_imgs")
    list_of_idx = sorted([int(idx) for idx in os.listdir(eval_path)])
    label_num = len(list_of_idx)
    images_list, targets_list = [], []
    for idx in list_of_idx:
        for filename in os.listdir(os.path.join(eval_path, str(idx))):
            image = Image.open(os.path.join(eval_path, str(idx), filename))
            image = torchvision.transforms.functional.to_tensor(image)
            images_list.append(image)
            targets_list.append(idx)
    mapping = []
    repeat_times = int(len(targets_list)/label_num)
    for i in range(repeat_times):
        mapping.extend(list(range(i, len(targets_list), repeat_times)))
    fake_images = torch.stack(images_list, dim=0).to(device)[mapping]
    fake_targets = torch.LongTensor(targets_list).to(device)[mapping]

    if args.all2success:
        all2success(fake_images, fake_targets, E, label_num, os.path.join(args.eval_path, "success_imgs"))

    if args.cal_acc:
        acc1, acc5, var1, var5 = calc_acc_std(fake_images, fake_targets, E, label_num)
        print(f"Final Top1: {acc1:.2%} ± {var1:.2%}, Top5: {acc5:.2%} ± {var5:.2%}")

    if args.cal_fid:
        calc_pytorch_fid(eval_path)
        torch.cuda.empty_cache()

    if args.cal_kid:
        calc_clean_kid(fdir1=eval_path, label_num=label_num, device=device)
        
    if args.cal_knn: 
        eval_path_success = os.path.join(args.eval_path, "success_imgs")
        list_of_idx = os.listdir(eval_path_success)
        images_list, targets_list = [], []
        for idx in list_of_idx:
            for filename in os.listdir(os.path.join(eval_path_success, idx)):
                image = Image.open(os.path.join(eval_path_success, idx, filename))
                image = torchvision.transforms.functional.to_tensor(image)
                images_list.append(image)
                targets_list.append(int(idx))
        fake_images_success = torch.stack(images_list, dim=0).to(device)
        fake_targets_success = torch.LongTensor(targets_list).to(device)
        calc_knn(fake_images_success, fake_targets_success, E=E, device=device)

    if args.cal_PRCD:
        calc_PRCD(eval_path)
        torch.cuda.empty_cache()

    if args.cal_piq:
        for metric in ["psnr", "ssim", "fsim", "vsi", "haarpsi", "srsim", "dss", "mdsi"]:
            calc_piq(fake_images, fake_targets, metric)

    if args.cal_lpips: 
        calc_lpips(fake_images, fake_targets)


if __name__ == '__main__':
    main()