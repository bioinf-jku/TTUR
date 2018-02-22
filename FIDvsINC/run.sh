#!/bin/bash
python3 main.py \
--path_IncNet "path/to/inception/net.pb" \
--dataset "CelebA" # Kind of dataset (one of the following: "CelebA", "Cifar10" or "Other")
--path_data "path/to/data" \
--path_out "./out" \ # Path to output directory
--path_stats "./stats/<name_of_stats.npz>" \ # path to image statistics
--img_file_ext  "*" \ # file extension in te form e.g. "*.jpg", "*.png" or only "*" \
--noise_type "sp:rect:swirl:blur:gn" \
--n_imgs 50000 \
--gpu "0" \ # set CUDA_VISIBLE_DEVICES
--verbose "Y" \
--sub_paths "Y" \
--img_dims 64 64 3 # need only to be specified for dataset "Other"
