#!/usr/bin/env bash
set -e

# Run a trained model on new data
#
# Example call:
#   ./scripts/eval_nightdrive/run_test.sh

# Settings
dataroot='/home/till/data/driving/BerkeleyDeepDrive/bdd100k_sorted/valid/'
jsonfile='/home/till/data/driving/BerkeleyDeepDrive/bdd100k_sorted/valid/bdd100k_sorted_valid'
dataset_mode='deepdrive'
out_style='basic'
num_test=10  # number of images to be processed (-1 for all)
name='cgan_aws_v023'  # name of model run / experiment
results_dir='./results/scratch1/'  # dir for saving model output
epoch='latest'

# Run model
python3 nightdrive_test.py \
    --model nightdrivecyclegan \
    --phase test \
    --no_dropout \
    --preprocess none \
    --load_size 1280 \
    --gpu_ids -1 \
    --out_style basic \
    --dataset_mode ${dataset_mode} \
    --dataroot ${dataroot} \
    --jsonfile ${jsonfile} \
    --results_dir ${results_dir} \
    --name ${name} \
    --num_test ${num_test} \
    --epoch ${epoch} \
    --eval



