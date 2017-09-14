#!/bin/bash

# celebA, lsun, imagenet, cifar10..
dataset="cifar10"

lr_d=$1
lr_g=$2

counter_start=0
load_checkpoint=false
if ! $load_checkpoint; then
  dwt=`date "+%m%d_%H%M%S"`
  run_id=${dwt}_${lr_d}_${lr_g}
else
  run_id="MMdd_hhmmss_lrd_lrg"
fi

incept_path="inception-2015-12-05/classify_image_graph_def.pb"

case $dataset in
  celebA)
    data_path="data/celebA_cropped"
    stats_path="stats/fid_stats_celeba.npz"
    input_height=64
    output_height=64
    input_fname_pattern="*.jpg"
    epochs=81
    ;;
  lsun)
    data_path="data/lsun_cropped"
    stats_path="stats/fid_stats_lsun.npz"
    input_height=64
    output_height=64
    input_fname_pattern="*.jpg"
    epochs=9
    ;;
  imagenet)
    data_path="data/imagenet"
    stats_path="stats/fid_stats_imagenet.npz"
    input_height=64
    output_height=64
    input_fname_pattern="*.jpg"
    epochs=5
    ;;
  cifar10)
    data_path="data/cifar10_train"
    stats_path="stats/fid_stats_cifar10.npz"
    input_fname_pattern="*.png"
    input_height=32
    output_height=32
    epochs=500
    ;;
esac

python3 main.py \
--dataset=$dataset \
--input_height=$input_height \
--output_height=$output_height \
--input_fname_pattern=$input_fname_pattern \
--is_crop False \
--is_train=True \
--batch_size=64 \
--checkpoint_dir="logs/${run_id}/checkpoints" \
--log_dir="logs/${run_id}/logs" \
--sample_dir="logs/${run_id}/samples" \
--fid_n_samples 50000 \
--fid_sample_batchsize 1000 \
--fid_batch_size 100 \
--fid_eval_steps 5000 \
--learning_rate_d $lr_d \
--learning_rate_g $lr_g \
--beta1 0.5 \
--epoch $epochs \
--load_checkpoint $load_checkpoint \
--counter_start $counter_start \
--incept_path $incept_path \
--data_path $data_path \
--stats_path $stats_path \
