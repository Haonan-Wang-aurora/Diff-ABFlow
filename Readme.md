# [NeurIPS 2025 Spotlight] Injecting Frame-Event Complementary Fusion into Diffusion for Optical Flow in Challenging Scenes

[Haonan Wang](https://scholar.google.com.hk/citations?hl=zh-CN&view_op=list_works&gmla=AH8HC4wel7f5UzHZm3NN_RHl9by4ODKcg12HuynxhWBbyyFpY3GCQp_wRryBPNSci76ZfoOB8_IDasu-vEEyzy9skm3tDy0&user=LCNXgmAAAAAJ) $^{1}$, [Hanyu Zhou](https://hyzhouboy.github.io/) $^{2✉}$,  [Haoyue Liu](https://scholar.google.com.hk/citations?hl=zh-CN&user=DadbHdAAAAAJ) $^1$, [Luxin Yan](https://scholar.google.com.hk/citations?user=5CS6T8AAAAAJ&hl=zh-CN) $^{1}$

$^1$ Huazhong University of Science and Technology  $^2$ National University of Singapore

$^✉$ Corresponding Author.

## Overview

![fig2](./images/Figure_1.png)

![fig2](./images/Figure_2.png)

## News

2025.09.18: Our paper is accepted by NeurIPS 2025 as Spotlight paper. 

2026.01.04: We release the official implementation of our Diff-ABFlow.



## Installation

```shell
# Clone our repository
git clone https://github.com/Haonan-Wang-aurora/Diff-ABFlow.git
cd Diff-ABFlow

# Create environment
conda create -n diff-abflow python=3.8 -y
conda activate diff-abflow

# Install torch
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html

# Install dependency
pip install -r requirements.txt
```



## Datasets

The datasets used in our paper are released on [Hugging Face](https://huggingface.co/datasets/Aurora03/Diff-ABFlow-Datasets).

Please download the datasets and update the corresponding paths in `core/event_datasets.py`.



## Evaluation

We release the evaluation scripts along with our trained models, which are available on [Hugging Face](https://huggingface.co/Aurora03/Diff-ABFlow-models).

```shell
# Please modify the inference results saving path in 'evaluate_event.py' before running the following scripts.

# Evaluate ckpt on Event-KITTI dataset
sh eval_kitti.sh

# Evaluate ckpt on DSEC dataset
sh eval_dsec.sh

# Visualization optical flow
python visualization.py
```



## Training

We provide training code for the **Event-KITTI** and **DSEC** datasets. After downloading the processed datasets from [Hugging Face](https://huggingface.co/datasets/Aurora03/Diff-ABFlow-Datasets), you can train the model starting from the provided pretrained checkpoint `DiffABFlow-pretrained.pth`.

Please modify the checkpoint saving path in `train.py` to specify where your own checkpoints will be stored.

```shell
# Training on Event-KITTI dataset
sh train_kitti.sh

# Training on DSEC dataset
sh train_dsec.sh
```

