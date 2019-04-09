#!/usr/bin/env bash

# ./scripts/eval_nightdrive/eval_trainingprogress.sh

dataroot='/home/SharedFolder/CurrentDatasets/bdd100k_sorted/'
jsonfile='/home/SharedFolder/CurrentDatasets/bdd100k_sorted/bdd100k_sorted_cgan'
dataset_mode='nightdrive'
#
crop_size=512
lambda_identity=0.1
niter=5
init_gain=0.01
#
name='cgan_aws_v024'

# start visdom server & execute one model
python3 -m visdom.server & python3 nightdrive_train.py \
    --model nightdrivecyclegan \
    --phase train \
    --no_dropout \
    --preprocess scale_width_and_crop \
    --norm instance \
    --load_size 1280 \
    --crop_size ${crop_size} \
    --gpu_ids 0 \
    --out_style basic \
    --dataset_mode ${dataset_mode} \
    --dataroot ${dataroot} \
    --jsonfile ${jsonfile} \
    --name ${name} \
    --save_epoch_freq 1 \
    --save_by_iter \
    --update_html_freq 8000 \
    --display_freq 8000 \
    --display_id -1 \
    --lambda_identity ${lambda_identity} \
    --niter ${niter} \
    --init_gain ${init_gain}