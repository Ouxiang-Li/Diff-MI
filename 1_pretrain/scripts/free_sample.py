"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import pdb
import argparse
import os, sys

import numpy as np
import torch as th
import torch.distributed as dist

from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)


'''
MODEL_FLAGS="--image_size 64 --num_channels 128 --num_res_blocks 3 --class_cond True"
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule cosine"

python free_sample.py $MODEL_FLAGS $DIFFUSION_FLAGS --cuda 4 --class_cond True
'''

def main():
    args = create_argparser().parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"]= str(args.cuda)
    dist_util.setup_dist()
    logger.configure(args.logger_path)
    logger.log("creating model and diffusion...")

    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys()),
        w=args.w,
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    logger.log("sampling...")
    all_images = []
    all_labels = []
    while len(all_images) * args.batch_size < args.num_samples:
        model_kwargs = {}
        if args.class_cond:
            classes = th.randint(
                low=args.fix_id, high=args.fix_id+1, size=(args.batch_size,), device=dist_util.dev()
            )
            print('classes:', classes)
            model_kwargs["y"] = classes
        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        sample = sample_fn(
            model,
            (args.batch_size, 3, args.image_size, args.image_size),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
        )

        visualize(sample, dir=logger.get_dir(), label='free_samples')
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()

        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
        if args.class_cond:
            gathered_labels = [
                th.zeros_like(classes) for _ in range(dist.get_world_size())
            ]
            dist.all_gather(gathered_labels, classes)
            all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
        logger.log(f"created {len(all_images) * args.batch_size} samples")

    arr = np.concatenate(all_images, axis=0)
    arr = arr[: args.num_samples]
    if args.class_cond:
        label_arr = np.concatenate(all_labels, axis=0)
        label_arr = label_arr[: args.num_samples]
    if dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in arr.shape])
        out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}.npz")
        logger.log(f"saving to {out_path}")
        if args.class_cond:
            np.savez(out_path, arr, label_arr)
        else:
            np.savez(out_path, arr)

    dist.barrier()
    logger.log("sampling complete")


def visualize(img, dir='/data/liox', label=''):
    import math
    import matplotlib.pyplot as plt

    sample = ((img + 1) * 127.5).clamp(0, 255).to(th.uint8)
    sample = sample.permute(0, 2, 3, 1)
    sample = sample.contiguous()
    arr = np.array([i.cpu().numpy() for i in sample])

    N = min(int(math.sqrt(len(arr))), 4)
    for i in range(N*N):
        plt.subplot(N, N, i+1)
        plt.imshow(arr[i])
        plt.xticks([])
        plt.yticks([])
    plt.tight_layout()
    plt.savefig(dir + f'/visual_{label}.png')
    plt.close()


def create_argparser():
    defaults = dict(
        logger_path="./logger",
        clip_denoised=True,
        num_samples=16,
        batch_size=16,
        use_ddim=False,
        model_path="",
        cuda=0,
        w=3.0,
        fix_id=23,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
