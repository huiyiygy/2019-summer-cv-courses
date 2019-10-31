#!/usr/bin/env bash

# Species classification
# experiment 0 xception pretrained
# CUDA_VISIBLE_DEVICES=0 python train.py --learn-rate 0.001 --weight-decay 0.0 --epochs 50 --batch-size 31 --base-size 300 --crop-size 299 --gpu-ids 0  --backbone xception --checkname xception --eval-interval 1 --dataset Species

# experiment 1 inceptionv4 pretrained
CUDA_VISIBLE_DEVICES=0 python train.py --learn-rate 0.001 --weight-decay 0.0 --epochs 50 --batch-size 31 --base-size 300 --crop-size 299 --gpu-ids 0  --backbone inceptionv4 --checkname inceptionv4 --eval-interval 1 --dataset Species