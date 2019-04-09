#!/usr/bin/env python
# Run domain-conversion of the complete Berkley Deep Drive data set
#
# Example call:
#   ./scripts/eval_nightdrive/run_conversion.py

# Settings
# Specify model
name='cgan_aws_v023'  # name of model run / experiment
epoch='latest'
# Paths and files
dataroot='/home/till/data/driving/BerkeleyDeepDrive/bdd100k/images/100k/val'
jsonfile="/home/till/data/driving/BerkeleyDeepDrive/bdd100k/labels/bdd100k_labels_images_val"
results_dir="${dataroot}"+"_converted"  # dir for saving model output

# Algorithmic parameters
dataset_mode='deepdrive'
out_style='conversion'
num_test=10 # -1  # number of images to be processed (-1 for all)

# Run model
import subprocess
cmd = "python3 /home/till/projects/git-forks/pytorch-CycleGAN-and-pix2pix/nightdrive_test.py \
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
    --epoch ${epoch}"

p = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
out, err = p.communicate()
result = out.split('\n')

