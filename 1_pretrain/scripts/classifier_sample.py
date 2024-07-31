"""
Like image_sample.py, but use a noisy image classifier to guide the sampling
process towards more realistic images.
"""

import pdb
import argparse
import os
import sys

import numpy as np
import torch as th
import torch.distributed as dist
import torch.nn.functional as F
import torchvision.utils as vutils

from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    classifier_defaults,
    create_model_and_diffusion,
    create_classifier,
    add_dict_to_argparser,
    args_to_dict,
)


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist(args.cuda)
    logger.configure(args.logger_dir)

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    logger.log("loading classifier...")

    # ===========================================
    # from utils import Classifier
    # classifier = Classifier(num_classes=1000, architecture='resnet34')
    # classifier.load_state_dict(th.load('/data/liox/Plug-and-Play-Attacks/results/resnet34_64/Classifier.pth')["model_state_dict"])
    # classifier.eval()
    # ===========================================

    classifier = create_classifier(**args_to_dict(args, classifier_defaults().keys()))
    classifier.load_state_dict(
        dist_util.load_state_dict(args.classifier_path, map_location="cpu")
    )
    classifier.to(dist_util.dev())
    if args.classifier_use_fp16:
        classifier.convert_to_fp16()
    classifier.eval()

    # from torchsummary import summary
    # summary(classifier, (3, 64, 64))
    # sys.exit()

    # ===========================================
    # from diffusers import DDIMPipeline
    # import torchvision.transforms.functional
    # del model

    # def cond_fn(x, t, y=None):
    #     assert y is not None
    #     with th.enable_grad():
    #         x_in = x.detach().requires_grad_(True)
    #         x_in_cls = torchvision.transforms.functional.resize(x_in, 64)
    #         logits = classifier(x_in_cls, t.to(dist_util.dev()))
    #         log_probs = F.log_softmax(logits, dim=-1)
    #         selected = log_probs[range(len(logits)), y.view(-1)]
    #         return th.autograd.grad(selected.sum(), x_in)[0] * 5.0

    # model_id = "google/ddpm-celebahq-256"
    # fixed_id = 415
    # classes = th.randint(low=fixed_id, high=(fixed_id + 1), size=(args.batch_size,), device=dist_util.dev())
    # print('labels:', classes)

    # # load model and scheduler
    # ddim = DDIMPipeline.from_pretrained(model_id)  # you can replace DDPMPipeline with DDIMPipeline or PNDMPipeline for faster inference

    # # run pipeline in inference (sample random noise and denoise)
    # x = th.randn((args.batch_size, 3, 256, 256),).to(dist_util.dev())
    # image = ddim(x, num_inference_steps=100, cond_fn=cond_fn, y=classes)['sample']

    # # save images
    # visualize(image, 'guided_test')
    # test = torchvision.transforms.functional.resize(image, 64)
    # print('Prediction:', classifier(test, th.zeros(args.batch_size,).to(dist_util.dev())).max(1)[1])

    # sys.exit()
    # ============================================

    def cond_fn(x, t, y=None):
        assert y is not None
        with th.enable_grad():
            x_in = x.detach().requires_grad_(True)
            # logits = classifier(x_in)
            logits = classifier(x_in, t)
            log_probs = F.log_softmax(logits, dim=-1)
            selected = log_probs[range(len(logits)), y.view(-1)]
            return th.autograd.grad(selected.sum(), x_in)[0] * args.classifier_scale

    def model_fn(x, t, y=None):
        assert y is not None
        return model(x, t, y if args.class_cond else None)

    logger.log("sampling...")
    all_images = []
    all_labels = []
    while len(all_images) * args.batch_size < args.num_samples:
        model_kwargs = {}

        fixed_id = 415
        classes = th.randint(low=fixed_id, high=(fixed_id + 1), size=(args.batch_size,), device=dist_util.dev())
        print('labels:', classes)

        model_kwargs["y"] = classes
        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        sample = sample_fn(
            model_fn,
            (args.batch_size, 3, args.image_size, args.image_size),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            cond_fn=cond_fn,
            device=dist_util.dev(),
        )
        visualize(sample, dir=logger.get_dir(), label='guided_samples')
        # print('Prediction:', classifier(sample).max(1)[1])
        print('Prediction:', classifier(sample, th.zeros(args.batch_size,).to(dist_util.dev())).max(1)[1])

        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()

        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
        gathered_labels = [th.zeros_like(classes) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_labels, classes)
        all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
        logger.log(f"created {len(all_images) * args.batch_size} samples")

    arr = np.concatenate(all_images, axis=0)
    arr = arr[: args.num_samples]

    label_arr = np.concatenate(all_labels, axis=0)
    label_arr = label_arr[: args.num_samples]
    if dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in arr.shape])
        out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}.npz")
        logger.log(f"saving to {out_path}")
        np.savez(out_path, arr, label_arr)

    dist.barrier()
    logger.log("sampling complete")


def visualize(img, dir='/data/liox', label=''):
    import math
    import matplotlib.pyplot as plt

    sample = ((img + 1) * 127.5).clamp(0, 255).to(th.uint8)
    sample = sample.permute(0, 2, 3, 1)
    sample = sample.contiguous()
    arr = np.array([i.cpu().numpy() for i in sample])

    N = min(int(math.sqrt(len(arr))), 10)
    for i in range(N*N):
        plt.subplot(N, N, i+1)
        plt.imshow(arr[i])
        plt.xticks([])
        plt.yticks([])
    plt.savefig(dir + f'/visual_{label}.png')
    plt.close()


def create_argparser():
    defaults = dict(
        logger_dir="/data/liox/guided-diffusion/logger",
        clip_denoised=True,
        num_samples=4,
        batch_size=4,
        use_ddim=True,
        model_path="/data/liox/guided-diffusion/logger/openai-2022-10-20-15-45-27-733124/model150000.pt",
        classifier_path="/data/liox/guided-diffusion/logger/openai-2022-10-22-01-39-20-342874/model149999.pt",
        classifier_scale=7.0,
        cuda=0,
    )
    defaults.update(model_and_diffusion_defaults())
    defaults.update(classifier_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
