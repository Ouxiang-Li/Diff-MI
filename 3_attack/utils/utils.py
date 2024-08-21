import os, sys, pdb
import math, statistics

import random
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from torchvision import datasets, transforms
from torch.nn import functional as F
from torch.utils.data import TensorDataset, Subset
from kornia import augmentation

import matplotlib.pyplot as plt
from tqdm import tqdm

from pytorch_fid.inception import InceptionV3
from pytorch_fid.fid_score import calculate_fid_given_paths


class Tee(object):
    def __init__(self, name, mode):
        self.file = open(name, mode)
        self.stdout = sys.stdout
        sys.stdout = self
    def __del__(self):
        sys.stdout = self.stdout
        self.file.close()
    def write(self, data):
        if '...' not in data:
            self.file.write(data)
        self.stdout.write(data)
        self.flush()
    def flush(self):
        self.file.flush()


def save_tensor_to_image(imgs, lables, save_path):
    fake_dataset = TensorDataset(imgs.cpu(), lables.cpu())
    for i, (x,y) in enumerate(torch.utils.data.DataLoader(fake_dataset, batch_size=1, shuffle=False)):
        label_path = os.path.join(save_path, str(y.item()))
        if not os.path.exists(label_path): os.makedirs(label_path)
        torchvision.utils.save_image(x.detach()[0,:,:,:], os.path.join(label_path, f"{i}_attack.png"), padding=0)


def all2success(imgs, labels, cls, label_num, outdir):

    if os.path.exists(outdir):
        return None
    else:
        os.mkdir(outdir)

    success_img, success_label = [], []
    os.makedirs(outdir, exist_ok=True)
    assert imgs.shape[0] % label_num == 0
    for i in range(int(imgs.shape[0]/label_num)):
        imgs_ = imgs[i * label_num: (i+1) * label_num]
        labels_ = labels[i * label_num: (i+1) * label_num]
        assert torch.max(labels_) - torch.min(labels_) == label_num - 1
        _, _, success_idx = calc_acc(cls, augmentation.Resize((112, 112))(imgs_), 
                                     labels_, with_success=True, enable_print=False)
        success_img.append(imgs_[success_idx])
        success_label.append(labels_[success_idx])
    success_img, success_label = torch.cat(success_img), torch.cat(success_label)
    save_tensor_to_image(success_img, success_label, outdir)


def set_random_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# region [evaluation]
    
PRIVATE_PATH = "data/celeba_private/celeba_private_300"

def calc_acc(classifier, imgs, labels, bs=64, anno='', with_success=False, enable_print=True):

    output, img_dataset = [], TensorDataset(imgs)
    for x in torch.utils.data.DataLoader(img_dataset, batch_size=bs, shuffle=False):
        output.append(classifier(x[0])[-1])
    output = torch.cat(output)
    top1_count = torch.eq(torch.topk(output, k=1)[1], labels.view(-1,1)).float()
    top5_count = torch.eq(torch.topk(output, k=5)[1], labels.view(-1,1)).float()
    if enable_print:
        print(f'===> top1_acc: {top1_count.mean().item():.2%}, top5_acc: {5*top5_count.mean().item():.2%} {anno}')
    if with_success:
        success_idx = torch.nonzero((output.max(1)[1] == labels).int()).squeeze(1)
        return top1_count.sum().item(), top5_count.sum().item(), success_idx
    else:
        return top1_count.sum().item(), top5_count.sum().item()


def calc_acc_std(imgs, labels, cls, label_num):
    top1_list, top5_list = [], []
    assert imgs.shape[0] % label_num == 0
    for i in range(int(imgs.shape[0]/label_num)):
        imgs_ = imgs[i * label_num: (i+1) * label_num]
        labels_ = labels[i * label_num: (i+1) * label_num]
        assert torch.max(labels_) - torch.min(labels_) == label_num - 1
        top1_count, top5_count = calc_acc(cls, augmentation.Resize((112, 112))(imgs_), 
                                          labels_, enable_print=False)
        top1_list.append(top1_count/label_num)
        top5_list.append(top5_count/label_num)
    try:
        acc1 = statistics.mean(top1_list)
        acc5 = statistics.mean(top5_list)
    except Exception:
        acc1, acc5 = top1_list[0], top5_list[0]

    try:
        var1 = statistics.stdev(top1_list)
        var5 = statistics.stdev(top5_list)
    except Exception:
        var1, var5 = 0.0, 0.0
        
    return acc1, acc5, var1, var5


def calc_pytorch_fid(file_path, file_path2=PRIVATE_PATH, anno=""):
    fid_value = calculate_fid_given_paths(paths=[file_path, file_path2], batch_size=64, device='cuda', dims=2048, num_workers=8,)
    print(f"[pytorch_fid] FID score computed on file {file_path.split('Inversion/')[-1]} is {fid_value:.2f} {anno}")


def calc_clean_fid(file_path):
    from cleanfid import fid
    fid_value = fid.compute_fid(file_path, PRIVATE_PATH, batch_size=64, num_workers=8)
    print(f"[clean_fid] FID score computed on file {file_path.split('Inversion/')[-1]} is {fid_value:.2f}")


def calc_clean_kid(fdir1, label_num, fdir2=PRIVATE_PATH, anno="", device="cuda"):

    import random
    from glob import glob
    from cleanfid.features import build_feature_extractor
    from cleanfid.utils import ResizeDataset

    def get_files_features(l_files, model=None, num_workers=12,
                        batch_size=128, device=torch.device("cuda"),
                        mode="clean", custom_fn_resize=None,
                        description="", fdir=None, verbose=True,
                        custom_image_tranform=None):
        # wrap the images in a dataloader for parallelizing the resize operation
        dataset = ResizeDataset(l_files, fdir=fdir, mode=mode)
        if custom_image_tranform is not None:
            dataset.custom_image_tranform=custom_image_tranform
        if custom_fn_resize is not None:
            dataset.fn_resize = custom_fn_resize

        dataloader = torch.utils.data.DataLoader(dataset,
                        batch_size=batch_size, shuffle=False,
                        drop_last=False, num_workers=num_workers)

        # collect all inception features
        l_feats = []
        if verbose:
            pbar = tqdm(dataloader, desc=description)
        else:
            pbar = dataloader
        
        for batch in pbar:
            with torch.no_grad():
                feat = model(batch.to(device))
            l_feats.append(feat.detach())
        ts_feats = torch.cat(l_feats)
        return ts_feats

    def kernel_distance(feats1, feats2, num_subsets=100, max_subset_size=1000):
        n = feats1.shape[1]
        m = min(min(feats1.shape[0], feats2.shape[0]), max_subset_size)
        t = 0

        for _subset_idx in range(num_subsets):
            indices_x = torch.randperm(feats2.shape[0])[:m]
            indices_y = torch.randperm(feats1.shape[0])[:m]

            x = feats2[indices_x]
            y = feats1[indices_y]

            a = (torch.mm(x, x.t()) / n + 1) ** 3 + (torch.mm(y, y.t()) / n + 1) ** 3
            b = (torch.mm(x, y.t()) / n + 1) ** 3

            t += (a.sum() - torch.diag(a).sum()) / (m - 1) - b.sum() * 2 / m

        kid = t / num_subsets / m
        return float(kid)

    np_feats, feat_model = [], build_feature_extractor(mode="clean", device=device, use_dataparallel=True)
    for fdir in [fdir1, fdir2]:
        files, list_of_idx = [], sorted([int(idx) for idx in os.listdir(fdir)])
        for idx in list_of_idx:
            files.extend(
                os.path.join(fdir, str(idx), filename)
                for filename in os.listdir(os.path.join(fdir, str(idx)))
            )
        with torch.no_grad():
            np_feats.append(get_files_features(files, feat_model, device=device, verbose=False))

    KID_list, repeat_times = [], int(len(np_feats[0])/label_num)
    for i in tqdm(range(label_num), desc='Calculating KID'):
        score = kernel_distance(
                    np_feats[0][i*repeat_times: (i+1)*repeat_times],
                    np_feats[1][i*27: (i+1)*27])  # 27 represents the number of images for every target label
        KID_list.append(score)
    KID = 1e3 * np.mean(np.array(KID_list))

    print(f"KID score is {KID:.2f} {anno}")


def calc_knn(fake_imgs, fake_targets, E, anno='', path="assets/celeba_private_feats", device='cuda'):

    # get features of reconstructed images
    infered_feats = None
    for i, images in enumerate(torch.utils.data.DataLoader(fake_imgs, batch_size=64)):
        images = augmentation.Resize((112, 112))(images).to(device)
        feats = E(images)[0]
        if i == 0:
            infered_feats = feats.detach().cpu()
        else:
            infered_feats = torch.cat([infered_feats, feats.detach().cpu()], dim=0)

    # get features of target images
    idens = fake_targets.to(device).long()
    feats = infered_feats.to(device)
    true_feats = torch.from_numpy(np.load(os.path.join(path, "private_feats.npy"))).float().to(device)
    info = torch.from_numpy(np.load(os.path.join(path, "private_targets.npy"))).view(-1).long().to(device)
    bs = feats.size(0)
    knn_dist = 0

    def row_mse(a, b):
        c = a - b
        d = torch.pow(c, 2)
        e = torch.sum(d, dim=1)
        return e

    # calculate knn dist
    for i in tqdm(range(bs), desc='Calculating KNN Dist'):
        knn = 1e8
        idx = torch.nonzero(info == idens[i]).squeeze(1)
        fake_feat = feats[i].repeat(idx.shape[0], 1)
        true_feat = true_feats[idx]
        knn = row_mse(fake_feat, true_feat)
        knn_dist += torch.min(knn)
    knn = (knn_dist / bs).item()
    print(f"KNN Dist computed on {fake_imgs.shape[0]} attack samples: {knn:.2f} {anno}")


def calc_feat_dis(fake_imgs, fake_targets, E, anno='', feats_mean="assets/celeba_private_feats/private_feats_mean.npy", device='cuda'):

    # get features of reconstructed images
    infered_feats = None
    for i, images in enumerate(torch.utils.data.DataLoader(fake_imgs, batch_size=64)):
        images = augmentation.Resize((112, 112))(images).to(device)
        feats = E(images)[0]
        if i == 0:
            infered_feats = feats.detach().cpu()
        else:
            infered_feats = torch.cat([infered_feats, feats.detach().cpu()], dim=0)

    # get features of target images
    idens = fake_targets.to(device).long()
    feats = infered_feats.to(device)
    true_feats = torch.from_numpy(np.load(feats_mean)).float().to(device)
    feats_correspond = torch.index_select(true_feats, dim=0, index=idens)

    feat_dis = torch.mean(torch.norm((feats-feats_correspond), dim=1)**2)

    print(f"Feat Dist computed on {fake_imgs.shape[0]} attack samples: {feat_dis:.2f} {anno}")


class PRCD:

    def __init__(self, dataset_real, dataset_fake, num_classes,
                 dims=2048, batch_size=32, num_workers=8, device="cuda"):
        
        self.dataset_real = dataset_real
        self.dataset_fake = dataset_fake
        self.dims = dims
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_classes = num_classes
        self.device = device
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[self.dims]
        inception_model = InceptionV3([block_idx])
        self.inception_model = inception_model.to(self.device)
        self.up = nn.Upsample(size=(299, 299), mode='bilinear', align_corners=True).to(self.device)

    def compute_metric(self, k=3):

        precision_list, recall_list, density_list, coverage_list = [], [], [], []
        for idx, cls in enumerate(tqdm(range(self.num_classes))):

            with torch.no_grad():
                embedding_fake = self.compute_embedding(self.dataset_fake, cls)
                embedding_real = self.compute_embedding(self.dataset_real, cls)
                if embedding_fake.shape[0] > embedding_real.shape[0]:
                    embedding_fake = embedding_fake.repeat(embedding_fake.shape[0] // embedding_real.shape[0], 1)[:embedding_real.shape[0]]
                else:
                    embedding_real = embedding_real.repeat(embedding_real.shape[0] // embedding_fake.shape[0], 1)[:embedding_fake.shape[0]]
                
                pair_dist_real = torch.cdist(embedding_real, embedding_real, p=2)
                pair_dist_real = torch.sort(pair_dist_real, dim=1, descending=False)[0]
                pair_dist_fake = torch.cdist(embedding_fake, embedding_fake, p=2)
                pair_dist_fake = torch.sort(pair_dist_fake, dim=1, descending=False)[0]
                radius_real = pair_dist_real[:, k]
                radius_fake = pair_dist_fake[:, k]

                # Compute precision
                distances_fake_to_real = torch.cdist(embedding_fake, embedding_real, p=2)
                min_dist_fake_to_real, nn_real = distances_fake_to_real.min(dim=1)
                precision = (min_dist_fake_to_real <= radius_real[nn_real]).float().mean()
                precision_list.append(precision.cpu().item())

                # Compute recall
                distances_real_to_fake = torch.cdist(embedding_real, embedding_fake, p=2)
                min_dist_real_to_fake, nn_fake = distances_real_to_fake.min(dim=1)
                recall = (min_dist_real_to_fake <= radius_fake[nn_fake]).float().mean()
                recall_list.append(recall.cpu().item())

                # Compute coverage
                num_samples = distances_fake_to_real.shape[0]
                num_neighbors = (distances_fake_to_real <= radius_real.repeat(num_samples, 1)).float().sum(dim=0)
                coverage = (num_neighbors > 0).float().mean()
                coverage_list.append(coverage.cpu().item())

                # Compute density
                sphere_counter = (distances_fake_to_real <= radius_real.repeat(num_samples, 1)).float().sum(dim=0).mean()
                density = sphere_counter / k
                density_list.append(density.cpu().item())

        # Compute mean over targets
        precision = np.mean(precision_list)
        recall = np.mean(recall_list)
        coverage = np.mean(coverage_list)
        density = np.mean(density_list)
        return precision, recall, coverage, density

    def compute_embedding(self, dataset, cls=None):
        self.inception_model.eval()
        if cls is not None:
            dataset = self.SingleClassSubset(dataset, cls)
        else:
            raise NotImplementedError
        dataloader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=self.batch_size,
                                                 shuffle=False,
                                                 drop_last=False,
                                                 pin_memory=True,
                                                 num_workers=self.num_workers)
        pred_arr = np.empty((len(dataset), self.dims))
        start_idx = 0
        max_iter = int(len(dataset) / self.batch_size)
        for step, (x, y) in enumerate(dataloader):
            x = x.to(self.device)
            with torch.no_grad():
                x = self.up(x)
                pred = self.inception_model(x)[0]
            pred = pred.squeeze(3).squeeze(2).cpu().numpy()
            pred_arr[start_idx:start_idx + pred.shape[0]] = pred
            start_idx = start_idx + pred.shape[0]

        return torch.from_numpy(pred_arr)
    
    def SingleClassSubset(self, dataset, cls):
        indices = np.where(np.array(dataset.targets) == cls)[0]
        return Subset(dataset, indices)  

def calc_PRCD(fake_path, real_path=PRIVATE_PATH):

    num_classes = len(os.listdir(fake_path))
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
    fake_dataset = datasets.ImageFolder(fake_path, transform=transform)
    real_dataset = datasets.ImageFolder(real_path, transform=transform)

    prcd = PRCD(real_dataset, fake_dataset, num_classes)
    Precision, Recall, Coverage, Density = prcd.compute_metric()
    print(f'Precision: {Precision:.4f} | Recall: {Recall:.4f} | Coverage: {Coverage:.4f} | Density: {Density:.4f}')


def calc_piq(fake_images, fake_targets, metric, anno='', device='cuda'):
    
    from piq import psnr, ssim, gmsd, fsim, vsi, mdsi, haarpsi, srsim, dss
    metric_functions = {
        "psnr": psnr, "ssim": ssim, "gmsd": gmsd,
        "fsim": fsim, "vsi": vsi, "mdsi": mdsi, 
        "haarpsi": haarpsi, "srsim": srsim, "dss": dss
    }
    function = metric_functions[metric]

    # get target images and labels
    inferred_image_path = PRIVATE_PATH
    list_of_idx = os.listdir(inferred_image_path)
    images_list, targets_list = [], []
    # load reconstructed images
    for idx in list_of_idx:
        for filename in os.listdir(os.path.join(inferred_image_path, idx)):
            image = Image.open(os.path.join(inferred_image_path, idx, filename))
            image = T.functional.to_tensor(image)
            images_list.append(image)
            targets_list.append(int(idx))
    real_images = torch.stack(images_list, dim=0).to(device)
    real_targets = torch.LongTensor(targets_list).to(device)

    # get fake images and labels
    fake_images = fake_images.to(device)
    fake_targets = fake_targets.to(device)
    bs = fake_targets.size(0)
    value = 0

    # calculate metric
    for i in range(bs):
        single_value = -1
        idx = torch.nonzero(real_targets == fake_targets[i]).squeeze(1)
        for j in idx:
            temp_value = function(fake_images[i].unsqueeze(0), real_images[j].unsqueeze(0))
            if temp_value > single_value: single_value = temp_value
        value += single_value
    value = (value / bs).item()
    print(f"{metric} : {value:.4f} {anno}")


def calc_lpips(fake_images, fake_targets, anno='', device='cuda'):
    
    import lpips
    loss_fn_alex = lpips.LPIPS(net='alex').to(device) # best forward scores
    loss_fn_vgg = lpips.LPIPS(net='vgg').to(device) # closer to "traditional" perceptual loss, when used for optimization

    # get target images and labels
    inferred_image_path = PRIVATE_PATH
    list_of_idx = os.listdir(inferred_image_path)
    images_list, targets_list = [], []
    # load reconstructed images
    for idx in list_of_idx:
        for filename in os.listdir(os.path.join(inferred_image_path, idx)):
            image = Image.open(os.path.join(inferred_image_path, idx, filename))
            image = T.functional.to_tensor(image)
            images_list.append(image)
            targets_list.append(int(idx))
    real_images = torch.stack(images_list, dim=0).to(device)
    real_targets = torch.LongTensor(targets_list).to(device)

    # get fake images and labels
    fake_images = fake_images.to(device)
    fake_targets = fake_targets.to(device)
    bs = fake_targets.size(0)
    value_a, value_v = 0, 0

    # calculate metric
    for i in range(bs):
        single_value_a, single_value_v = -1, -1
        idx = torch.nonzero(real_targets == fake_targets[i]).squeeze(1)
        for j in idx:
            temp_value_a = loss_fn_alex(fake_images[i].unsqueeze(0), real_images[j].unsqueeze(0))
            temp_value_v = loss_fn_vgg(fake_images[i].unsqueeze(0), real_images[j].unsqueeze(0))
            if temp_value_a > single_value_a: single_value_a = temp_value_a
            if temp_value_v > single_value_v: single_value_v = temp_value_v
        value_a += single_value_a
        value_v += single_value_v
    value_a = (value_a / bs).item()
    value_v = (value_v / bs).item()
    print(f"LPIPS : Alex {value_a:.4f} | VGG {value_v:.4f} {anno}")

# endregion