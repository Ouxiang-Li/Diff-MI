import os, pdb

import math, kornia
import numpy as np
import blobfile as bf
import torch as th
import torchvision
import torch.distributed as dist
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW

from . import dist_util, logger
from .fp16_util import MixedPrecisionTrainer
from .resample import LossAwareSampler, UniformSampler

from tqdm import tqdm
from dpm_solver_pytorch import NoiseScheduleVP, model_wrapper, DPM_Solver


class TrainLoop:

    def __init__(
        self,
        *,
        args,
        target,
        model,
        diffusion,
        batch_size,
        learning_rate,
        resume_checkpoint=None,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        schedule_sampler=None,
        weight_decay=0.0,
    ):
        self.threshold = args.threshold
        self.epochs = args.epochs
        self.target = target
        self.model = model
        self.diffusion = diffusion
        self.batch_size = batch_size
        self.lr = learning_rate
        self.resume_checkpoint = resume_checkpoint

        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay

        self.global_batch = self.batch_size * dist.get_world_size()
        self.sync_cuda = th.cuda.is_available()
        self._load_and_sync_parameters()

        self.mp_trainer_cls = MixedPrecisionTrainer(
            model=self.model,
            layers=[['label_emb.weight', 
                     'middle_block.0.in_layers.2', 
                     'middle_block.0.out_layers.3', 
                     'middle_block.2.in_layers.2', 
                     'middle_block.2.out_layers.3', 
                     'middle_block.1.proj'], []],  # 5
            # layers=[['label_emb.weight', 'middle_block'], ['in_layers.0', 'out_layers.0', 'norm']],  # 8
            # layers=[['label_emb.weight', 'middle_block'], []],  # middle_block
            # layers=[['label_emb.weight', 'input_blocks'], ['input_blocks.0.0']],  # input_blocks
            # layers=[['label_emb.weight', 'output_blocks'], []],  # output_blocks
            use_fp16=self.use_fp16,
            fp16_scale_growth=fp16_scale_growth,
        )
        self.opt_cls = AdamW(
            self.mp_trainer_cls.master_params, lr=self.lr, weight_decay=self.weight_decay
        )

        if th.cuda.is_available():
            self.use_ddp = True
            self.ddp_model = DDP(
                self.model,
                device_ids=[dist_util.dev()],
                output_device=dist_util.dev(),
                broadcast_buffers=False,
                bucket_cap_mb=128,
                find_unused_parameters=False,
            )
        else:
            if dist.get_world_size() > 1:
                logger.warn(
                    "Distributed training requires CUDA. "
                    "Gradients will not be synchronized properly!"
                )
            self.use_ddp = False
            self.ddp_model = self.model

        # Configuration for dpm_solver
        b0, bT, timesteps = 1e-4, 2e-2, 1000
        betas = th.tensor(np.linspace(b0, bT, timesteps), dtype=float)
        self.noise_schedule = NoiseScheduleVP(schedule='discrete', betas=betas)

        # Load for p_reg loss
        self.p_reg = th.load(os.path.join("data/p_reg", f"{args.dataset}_{args.target}_p_reg.pt"))

        self.aug = kornia.augmentation.container.ImageSequential(
            kornia.augmentation.RandomResizedCrop((64, 64), scale=(0.8, 1.0), ratio=(1.0, 1.0)),
            kornia.augmentation.ColorJitter(brightness=0.2, contrast=0.2, p=0.5),
            kornia.augmentation.RandomHorizontalFlip(),
            kornia.augmentation.RandomRotation(5),
        )
        
    def _load_and_sync_parameters(self):
        if resume_checkpoint := find_resume_checkpoint() or self.resume_checkpoint:
            if dist.get_rank() == 0:
                logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
                self.model.load_state_dict(
                    dist_util.load_state_dict(
                        resume_checkpoint, map_location=dist_util.dev()
                    )
                )
        dist_util.sync_params(self.model.parameters())

    def run_cls_only(self, guidance_scale=3.0, aug_times=2):

        acc, acc_mean, iter = [], 0.0, 0
        labels = th.tensor(np.arange(0, 300)).to(dist_util.dev())
        label_dataset = TensorDataset(labels)

        while (acc_mean < self.threshold) and (iter < self.epochs):

            bar = tqdm(DataLoader(dataset=label_dataset, batch_size=self.batch_size, shuffle=True))
            for classes in bar:
                bs = classes[0].shape[0]
                bar.set_description(f'Epoch {iter}')
                self.mp_trainer_cls.zero_grad()
                model_fn = model_wrapper(
                    self.model,
                    self.noise_schedule,
                    model_type="noise",
                    guidance_type="classifier-free",
                    condition=classes[0],
                    unconditional_condition=th.ones_like(classes[0])*1000,
                    guidance_scale=guidance_scale)
                dpm_solver = DPM_Solver(model_fn, self.noise_schedule, algorithm_type="dpmsolver++")

                x_T = th.randn((bs, 3, 64, 64)).to(dist_util.dev())
                x0_pred_list = dpm_solver.sample(
                    x_T,
                    steps=10,
                    order=2,
                    method="multistep",
                    skip_type='time_uniform',
                    return_pred_x0=True)
                samples = th.cat(x0_pred_list)

                img_input_batch = []
                for _ in range(aug_times):
                    img_input = samples
                    img_input = self.aug(img_input).clamp(-1,1)
                    img_input_batch.append(img_input)
                img_input_batch = th.cat(img_input_batch)

                feats, logits = self.target((img_input_batch + 1) / 2)
                loss1 = self.topk_loss(logits, classes[0].repeat(len(x0_pred_list)*aug_times), k=20)
                loss2 = 1.0 * self.p_reg_loss(feats, classes[0].repeat(len(x0_pred_list)*aug_times))
                loss = loss1 + loss2

                self.mp_trainer_cls.backward(loss)
                self.mp_trainer_cls.optimize(self.opt_cls)

                acc.append(th.eq(th.topk(logits[-bs * aug_times:], k=1)[1], classes[0].repeat(aug_times).view(-1,1)).float().mean().item())
                bar.set_postfix({'Loss1': loss1.item(), 'Loss2': loss2.item(), 'Loss': loss.item()})

            # Save the fine-tuned model
            with th.no_grad():
                acc_mean = np.mean(acc)
                logger.log(f"The mean acc in iteration {iter} is {acc_mean:.2%}")
                if acc_mean >= (self.threshold - 0.05) or iter == (self.epochs - 1):
                    state_dict = self.mp_trainer_cls.master_params_to_state_dict(self.mp_trainer_cls.model_params)
                    filename = f"{self.resume_checkpoint.split('/')[-1].split('.pt')[0]}_{iter}_{acc_mean:.2%}.pt"
                    with bf.BlobFile(bf.join(get_blob_logdir(), filename), "wb") as f:
                        th.save(state_dict, f)
                acc, iter = [], (iter + 1)

    def topk_loss(self, out, iden, k):
        assert out.shape[0] == iden.shape[0]
        iden = iden.unsqueeze(1)
        real = out.gather(1, iden).squeeze(1)
        if k == 0: return -1 * real.mean()
        tmp_out = th.scatter(out, dim=1, index=iden, src=-th.ones_like(iden) * 1000.0)
        margin = th.topk(tmp_out, k=k)[0]
        return -1 * real.mean() + margin.mean()
    
    def p_reg_loss(self, featureT, classes):
        fea_reg = self.p_reg[classes]
        return F.mse_loss(featureT, fea_reg)
    

def get_blob_logdir():
    # You can change this to be a separate path to save checkpoints to
    # a blobstore or some external drive.
    return logger.get_dir()


def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    return None