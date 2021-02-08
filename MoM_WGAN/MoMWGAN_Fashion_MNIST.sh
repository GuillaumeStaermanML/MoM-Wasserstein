#!/usr/bin/env bash



export EXPERIMENT="path to save fake images and parameters of NetG"
export DATASET=Fashion-MNIST_anom
export DATAROOT="path"/data
export NITER=20000
export IMAGESIZE=32
export PROBA=0.5
export NC=1
export K=4
export SAVE_LOSS="path to save loss"


CUDA_VISIBLE_DEVICES=1 python3 "path"/MoM_Wgan.py \
  --experiment=$EXPERIMENT \
  --dataset=$DATASET \
  --dataroot=$DATAROOT \
  --niter=$NITER \
  --imageSize=$IMAGESIZE \
  --proba=$PROBA \
  --K=$K \
  --nc=$NC \
  --save_loss=$SAVE_LOSS
