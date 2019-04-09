#!/usr/bin/env bash
set -e

# Run a trained model on new data
#
# Example call:
#   ./scripts/eval_nightdrive/run_test_single.sh

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
# --model nightdrivecyclegan \
# --out_style basic_single \
python3 nightdrive_test.py \
    --model test \
    --direction AtoB \
    --phase test \
    --no_dropout \
    --preprocess none \
    --load_size 1280 \
    --gpu_ids -1 \
    --dataset_mode single \
    --dataroot ${dataroot} \
    --results_dir ${results_dir} \
    --name ${name} \
    --num_test ${num_test} \
    --model_suffix "_A" \
    --epoch ${epoch}




