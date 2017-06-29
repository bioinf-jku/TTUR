#!/usr/bin/env python3
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1"

import tensorflow as tf
import numpy as np
import scipy.misc
import fid
import pathlib
import urllib


def check_or_download_inception(inception_path):
    INCEPTION_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
    if inception_path is None:
        inception_path = '/tmp'
    inception_path = pathlib.Path(inception_path)
    model_file = inception_path / 'classify_image_graph_def.pb'
    if not model_file.exists():
        print("Downloading Inception model")
        from urllib import request
        import tarfile
        fn, _ = request.urlretrieve(INCEPTION_URL)
        with tarfile.open(fn, mode='r') as f:
            f.extract('classify_image_graph_def.pb', str(model_file.parent))
    return model_file


def calculate_fid(path, stats_path, inception_path, use_unbatched):
    if not os.path.exists(stats_path):
        raise RuntimeError("Invalid inception-statistics file")
    inception_path = check_or_download_inception(inception_path)

    path = pathlib.Path(path)
    files = list(path.glob('*.jpg')) + list(path.glob('*.png'))
    x = np.array([scipy.misc.imread(str(fn)).astype(np.float32) for fn in files])

    fid.create_incpetion_graph(str(inception_path))
    sigma, mu = fid.load_stats(stats_path)
    jpeg_tuple = fid.get_jpeg_encoder_tuple()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        query_tensor = fid.get_Fid_query_tensor(sess)
        if use_unbatched:
            fid_value = fid.FID_unbatched(x, query_tensor, mu ,sigma, jpeg_tuple, sess)
        else:
            pred_array = fid.get_predictions(x, query_tensor, sess, batch_size=128)
            fid_value, _, _ = fid.FID( pred_array, mu, sigma, sess)
        print("FID: ", fid_value)


if __name__ == "__main__":
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("image_dir", type=str, help='Path to the generated images', default='./img')
    parser.add_argument("-s", "--stats", type=str, help='Inception statistics of real data', required=True)
    parser.add_argument("-i", "--inception", type=str, help='Path to Inception model (will be downloaded if not provided)', default=None)
    parser.add_argument("--unbatched", help="Use the unbatched version", action="store_true")
    args = parser.parse_args()
    calculate_fid(parser.image_dir, parser.stats, parser.inception, parser.unbatched)
