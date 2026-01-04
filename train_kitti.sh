#!/bin/bash
mkdir -p DiffABFlow_kitti_night
CUDA_VISIBLE_DEVICES=7 python -u train.py --name DiffABFlow-kitti --stage kitti --validation kitti --restore_ckpt weights/FlowDiffuser-things.pth --gpus 0 --num_steps 80000 --batch_size 1 --lr 0.00009 --image_size 288 720 --wdecay 0.00001 --gamma=0.85