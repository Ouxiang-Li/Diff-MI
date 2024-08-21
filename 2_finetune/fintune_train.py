"""
Step-1: Fine-tune the Target-Specific CDM.
"""

import pdb
import argparse
import sys, os, random
import torch
import numpy as np 

from guided_diffusion import dist_util, logger
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from guided_diffusion.train_util import TrainLoop


def set_random_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_random_seed(42)


def main():

    args = create_argparser().parse_args()
    dist_util.setup_dist()
    logger.configure(dir=args.log_dir)

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.to(dist_util.dev())
    target = load_target(args.target).to(dist_util.dev())  
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("training...")
    TrainLoop(
        args=args,
        target=target,
        model=model,
        diffusion=diffusion,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
    ).run_cls_only()


def load_target(target_name):
    sys.path.append(os.getcwd())
    from models.classifier import VGG16, IR152, FaceNet64
    if target_name == "VGG16":
        T = VGG16(1000)
        path_T = 'assets/checkpoints/target_model/VGG16_88.26.tar'
    elif target_name == 'IR152':
        T = IR152(1000)
        path_T = 'assets/checkpoints/target_model/IR152_91.16.tar'
    elif target_name == "FaceNet64":
        T = FaceNet64(1000)
        path_T = 'assets/checkpoints/target_model/FaceNet64_88.50.tar'
    else:
        raise ValueError("Unsupported target model: {}".format(target_name))
    logger.log(f"loading target from checkpoint: {path_T}")
    T = torch.nn.DataParallel(T)
    ckp_T = torch.load(path_T)
    T.load_state_dict(ckp_T['state_dict'])
    T.eval()
    return T


def create_argparser():
    defaults = dict(
        log_dir="2_finetune/logger",
        schedule_sampler="uniform",
        lr=2e-4,
        weight_decay=0.0,
        batch_size=4,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        resume_checkpoint=None,
        # Setting for Target-specific Fine-tuning
        dataset=None,
        target=None,
        threshold=0.99,
        epochs=100,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
