# Diff-MI

This is the official Pytorch implementation of our paper:

> [Model Inversion Attacks through Target-specific Conditional Diffusion Models](https://www.arxiv.org/abs/2407.11424)
>
>  Ouxiang Li, Yanbin Hao, Zhicai Wang, Shuo Wang, Zaixi Zhang, FuliFeng

![fig1](assets/fig1.jpg)

## Requirements

Install the environment as follows:

```python
# create conda environment
conda create -n Diff-MI python=3.9
conda activate Diff-MI
# install pytorch 
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
# install other dependencies
pip install -r requirements.txt
```

## Preparation

### Data Collection

- Datasets: We use [CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html), [FFHQ](https://drive.google.com/open?id=1tg-Ur7d4vk1T8Bn0pPpUSQPxlPGBlGfv) and [FaceScrub](http://vintage.winklerbros.net/facescrub.html) in our experiments (we use a cleaned version of FaceScrub from this [repo](https://github.com/AI-Machine-Vision-Lab/FPVT_BMVC22-Face-Pyramid-Vision-Transformer)). You can directly download the pre-processed datasets [`reclassified_public_data`](https://drive.google.com/drive/folders/1w5Uj-5nhRPaYImTpiTEZPNQ8wNY_P8Hz?usp=drive_link) after top-n selection.

- We follow [KED-MI](https://github.com/SCccc21/Knowledge-Enriched-DMI/) to divide the CelebA into private and public data and use the private data of CelebA (the first 300 classes) [`celeba_private_300` ](https://drive.google.com/file/d/1qNo2tHTc8ywjffToC3W7kQyCysaWZWq_/view?usp=drive_link) for evaluation.

- We pre-compute the regularization features [`p_reg`](https://drive.google.com/drive/folders/1r0-fX7R6REqtBUkC7bNmlzAnpFKCzW2B?usp=drive_link) for $\mathcal{L}_{\text{p-reg}}$.

- You should organize the above data as follows:

	```
	data
	├── celeba_private
	│   └── celeba_private_300
	├── p_reg
	│   └── celeba_VGG16_p_reg.pt
	│   └── ...
	├── reclassified_public_data
	│   └── celeba
	│   └── facescrub
	│   └── ffhq
	```

### Models

- You can train target models following [KED-MI](https://github.com/SCccc21/Knowledge-Enriched-DMI/) or direcly download the provided checkpoints at: https://drive.google.com/drive/folders/1qfoELNMY8jedL2dSDocxNCaIUkUlc8_8?usp=drive_link and put them in `./assets/checkpoints`.
- To calculate KNN Dist, we pre-compute the features of private data on the evaluation model. You can download at: https://drive.google.com/drive/folders/1X2nBz6ZNHo-6aLf-HeZ83I-XHLRvp_aL?usp=drive_link and put them in `./assets/celeba_private_feats`.

## Step-1: Training Target-specific CDM

Here we take **VGG16** as the target classifier and **CelebA** as the public dataset as an example to train the target-specific CDM from scratch.

### Pretrain CDM

```
CUDA_VISIBLE_DEVICES=0,1 mpiexec -n 2 python 1_pretrain/free_train.py \
    --class_cond True \
    --batch_size 150 \
    --dataset celeba \
    --data_dir data/reclassified_public_data/celeba/VGG16_top30
```

We pretrain the CDM with batch size $150$ for $50\text{K}$ iterations on two A40 GPUs. Our ablation indicates that extended training iterations (e.g., $100\text{K}$) would lead to better attack performance. The pre-trained checkpoints will be saved at `./1_pretrain/logger`.

### Fine-tune CDM

```
CUDA_VISIBLE_DEVICES=0 python 2_finetune/fintune_train.py \
    --batch_size 7 \
    --dataset celeba \
    --target VGG16 \
    --epochs 10
```

## Step-2: Iterative Image Reconstruction 

```
CUDA_VISIBLE_DEVICES=0 python 3_attack/attack.py \
    --dataset celeba \
    --target VGG16 \
    --label_num 300 \
    --repeat_times 5 \
    --path_D 1_pretrain/logger/celeba/top30_VGG16/ema_0.9999_050000.pt
```

# Examples of Reconstructed Images

![fig2](assets/fig2.jpg)

# Citing
