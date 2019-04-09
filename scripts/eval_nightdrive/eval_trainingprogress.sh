#!/usr/bin/env bash
# set -e

# ./scripts/eval_nightdrive/eval_trainingprogress.sh

which_host="Docker"
if [${which_host}=="Docker"]; then
    dataroot='/home/SharedFolder/CurrentDatasets/bdd100k_sorted/valid'
    jsonfile='/home/SharedFolder/CurrentDatasets/bdd100k_sorted/annotations/bdd100k_sorted_valid'
    gpu_ids=0
elif [${which_host}=="till"]; then
    dataroot='/home/till/SharedFolder/CurrentDatasets/bdd100k_sorted/valid/'
    jsonfile='/home/till/SharedFolder/CurrentDatasets/bdd100k_sorted/annotations/bdd100k_sorted_valid'
    gpu_ids=-1
else
    echo "This host is unknown."
fi

name='cgan_aws_v032_backupbeforerestart' #  name='cgan_aws_v032'  # name of model run / experiment
dataset_mode='deepdrive'
out_style='basic'
num_test=10
results_dir=$"./results/${name}_trainprogress"
eval_mode="by_iter"

# evaluate by epoch
if [${eval_mode}=="by_epoch"]; then

    for epoch in {100..100..1}; do

        # execute one model
        python3 nightdrive_test.py \
            --model nightdrivecyclegan \
            --phase test \
            --no_dropout \
            --preprocess none \
            --load_size 1280 \
            --gpu_ids ${gpu_ids} \
            --out_style ${out_style} \
            --dataset_mode ${dataset_mode} \
            --dataroot ${dataroot} \
            --jsonfile ${jsonfile} \
            --results_dir ${results_dir} \
            --name ${name} \
            --num_test ${num_test} \
            --epoch ${epoch} \
            --out_suffix "_epoch_${epoch}"

        echo "Successfully processed epoch ${epoch}." | lolcat

        # run night classifier on the data
    done
fi


# evaluate by iter
if [${eval_mode}=="by_iter"]; then

    for iter in {130000..150000..5000}; do

        # execute one model
        python3 nightdrive_test.py \
            --model nightdrivecyclegan \
            --phase test \
            --no_dropout \
            --preprocess none \
            --load_size 1280 \
            --gpu_ids ${gpu_ids} \
            --out_style ${out_style} \
            --dataset_mode ${dataset_mode} \
            --dataroot ${dataroot} \
            --jsonfile ${jsonfile} \
            --results_dir ${results_dir} \
            --name ${name} \
            --num_test ${num_test} \
            --load_iter ${iter} \
            --out_suffix "_iter_${iter}"

        echo "Successfully processed epoch ${epoch}." | lolcat

        # run night classifier on the data

    done
fi



