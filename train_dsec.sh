#!/bin/bash
mkdir -p DiffABFlow_dsec_night
CUDA_VISIBLE_DEVICES=5  python -u train.py --name DiffABFlow-dsec --stage dsec --validation dsec --restore_ckpt weights/FlowDiffuser-things.pth --gpus 0 --num_steps 50000 --batch_size 1 --lr 0.00009 --image_size 400 560 --wdecay 0.00001 --gamma=0.85