#!/usr/bin/env bash



export EXPERIMENT="path to save fake images and parameters of NetG"
export DATASET=cifar10
export DATAROOT="path"/data
export NITER=20000
export IMAGESIZE=32
export NOISE=True #set False if you want to reproduce results without anomalies
export K=4
export SAVE_LOSS="path to save loss"


CUDA_VISIBLE_DEVICES=1 python3 "path"/MoM_Wgan.py \
  --experiment=$EXPERIMENT \
  --dataset=$DATASET \
  --dataroot=$DATAROOT \
  --niter=$NITER \
  --imageSize=$IMAGESIZE \
  --noise=$NOISE \
  --K=$K \
  --save_loss=$SAVE_LOSS 
