#!/usr/bin/env python3
from __future__ import absolute_import, division, print_function

import os
import glob
#os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import numpy as np
import fid
from scipy.misc import imread
import tensorflow as tf

########
# PATHS
########
data_path = '/local00/bioinf/celebA_cropped/' # set path to training set images
output_path = '/local00/bioinf/tom/fid_stats_celeba.npz' # path for where to store the statistics
# if you have downloaded and extracted
#   http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz
# set this path to the directory where the extracted files are, otherwise
# just set it to None and the script will later download the files for you
inception_path = None
inception_path = fid.check_or_download_inception(inception_path) # download inception if necessary


# loads all images into memory (this might require a lot of RAM!)
image_list = glob.glob(os.path.join(data_path, '*.jpg'))
images = np.array([imread(str(fn)).astype(np.float32) for fn in image_list])

fid.create_incpetion_graph(inception_path)  # load the graph into the current TF graph
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    mu, sigma = fid.calculate_activation_statistics(images, sess, batch_size=100)
    np.savez_compressed(output_path, mu=mu, sigma=sigma)
