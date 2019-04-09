#!/usr/bin/env bash
set -e

# Run domain-conversion of the complete Berkley Deep Drive data set
#
# Example call:
#   ./scripts/eval_nightdrive/run_conversion.sh

# ======================================================================
# Settings
# Specify model
name='cgan_aws_v032'  # name of model run / experiment
epoch=14

# Algorithmic parameters
dataset_mode='deepdrive'
out_style='conversion'
num_test=-1  # number of images to be processed (-1 for all)

# Paths and files
dataroot='/home/SharedFolder/CurrentDatasets'
case $USER in
    "till") dataroot='/home/till/SharedFolder/CurrentDatasets' ;;
    "night-drive") dataroot='/home/SharedFolder/CurrentDatasets' ;;
esac


# ======================================================================
# Run model for valid set
imagedir="${dataroot}/bdd100k/images/100k/val"
jsonfile="${dataroot}/bdd100k/labels/bdd100k_labels_images_val"
results_dir="${imagedir}_converted"  # dir for saving model output
python3 nightdrive_test.py \
    --model nightdrivecyclegan \
    --phase test \
    --no_dropout \
    --preprocess none \
    --load_size 1280 \
    --gpu_ids 0 \
    --out_style conversion \
    --dataset_mode ${dataset_mode} \
    --dataroot ${imagedir} \
    --jsonfile ${jsonfile} \
    --results_dir ${results_dir} \
    --name ${name} \
    --num_test ${num_test} \
    --epoch ${epoch}


# ======================================================================
# Run model for train set
imagedir="${dataroot}/bdd100k/images/100k/train"
jsonfile="${dataroot}/bdd100k/labels/bdd100k_labels_images_train"
results_dir="${imagedir}_converted"  # dir for saving model output
python3 nightdrive_test.py \
    --model nightdrivecyclegan \
    --phase test \
    --no_dropout \
    --preprocess none \
    --load_size 1280 \
    --gpu_ids 0 \
    --out_style conversion \
    --dataset_mode ${dataset_mode} \
    --dataroot ${imagedir} \
    --jsonfile ${jsonfile} \
    --results_dir ${results_dir} \
    --name ${name} \
    --num_test ${num_test} \
    --epoch ${epoch}