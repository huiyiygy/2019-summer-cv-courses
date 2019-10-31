#!/usr/bin/env bash

# Multi classification
# experiment 0 xception pretrained
# CUDA_VISIBLE_DEVICES=0 python train.py --learn-rate 0.001 --weight-decay 0.0 --epochs 100 --batch-size 31 --base-size 300 --crop-size 299 --gpu-ids 0  --backbone xception --checkname xception --eval-interval 1 --dataset Multi

# experiment 1 xception pretrained
# CUDA_VISIBLE_DEVICES=0 python train.py --learn-rate 0.001 --weight-decay 0.00001 --epochs 200 --batch-size 31 --base-size 300 --crop-size 299 --gpu-ids 0  --backbone xception --checkname xception --eval-interval 1 --dataset Multi

# experiment 2 xception pretrained
CUDA_VISIBLE_DEVICES=0 python train.py --learn-rate 0.0001 --weight-decay 0.0 --epochs 200 --batch-size 31 --base-size 300 --crop-size 299 --gpu-ids 0  --backbone xception --checkname xception --eval-interval 1 --dataset Multi