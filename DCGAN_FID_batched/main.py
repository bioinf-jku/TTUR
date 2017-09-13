import os
#os.environ['CUDA_VISIBLE_DEVICES'] = "1"
import numpy as np

from model import DCGAN
from utils import pp, visualize, to_json

import tensorflow as tf
import fid

flags = tf.app.flags
flags.DEFINE_integer("epoch", 25, "Epoch to train [25]")

flags.DEFINE_float("learning_rate_d", 0.0002, "Discriminator learning rate of for adam [0.002]")
flags.DEFINE_float("learning_rate_g", 0.0002, "Generator learning rate of for adam [0.0002]")
flags.DEFINE_float("lr_decay_rate_d", 1.0, "Discriminator learning rate decay [1.0]")
flags.DEFINE_float("lr_decay_rate_g", 1.0, "Generator learning rate decay [1.0]")

flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_integer("train_size", np.inf, "The size of train images [np.inf]")
flags.DEFINE_integer("batch_size", 64, "The size of batch images [64]")
flags.DEFINE_integer("input_height", 108, "The size of image to use (will be center cropped). [108]")
flags.DEFINE_integer("input_width", None, "The size of image to use (will be center cropped). If None, same value as input_height [None]")
flags.DEFINE_integer("output_height", 64, "The size of the output images to produce [64]")
flags.DEFINE_integer("output_width", None, "The size of the output images to produce. If None, same value as output_height [None]")
flags.DEFINE_integer("c_dim", 3, "Dimension of image color. [3]")
flags.DEFINE_string("dataset", "celebA", "The name of dataset [celebA, mnist, lsun]")
flags.DEFINE_string("input_fname_pattern", "*.jpg", "Glob pattern of filename of input images [*]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("checkpoint_name", None, "Directory name to load a checkpoint from [None]")
flags.DEFINE_boolean("load_checkpoint", False, "Load checkpoint [False]")
flags.DEFINE_integer("counter_start", 0, "counter to start with [0]")
flags.DEFINE_string("sample_dir", "samples", "Directory name to save the image samples [samples]")
flags.DEFINE_string("log_dir", "logs", "Directory name for summary logs [logs]")
flags.DEFINE_boolean("is_train", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("is_crop", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("visualize", False, "True for visualizing, False for nothing [False]")

# added parameters for batched fid
flags.DEFINE_string("stats_path", None, "Path to pretrained statistics")
flags.DEFINE_string("data_path", None, "Path to input data")
flags.DEFINE_string("incept_path", None, "Path to inception net.")
flags.DEFINE_integer("fid_n_samples", 10000, "Total number of samples generated to calculate the FID statistics. Will be adjusted if not a multiple of fid_sample_batchsize [10000]")
flags.DEFINE_integer("fid_sample_batchsize", 5000, "Batchsize of batches that constitute all generated samples to calculate the FID statistics [5000]")
flags.DEFINE_integer("fid_batch_size", 100, "Batchsize used for FID calculation [500]")
flags.DEFINE_boolean("fid_verbose", True, "Report current state of FID calculation [True]")
flags.DEFINE_integer("fid_eval_steps", 1000, "Evaluate FID after this number of minibatches")


FLAGS = flags.FLAGS

def main(_):

  pp.pprint(flags.FLAGS.__flags)

  # Create directories if necessary
  if not os.path.exists(FLAGS.log_dir):
    print("*** create log dir %s" % FLAGS.log_dir)
    os.makedirs(FLAGS.log_dir)
  if not os.path.exists(FLAGS.sample_dir):
    print("*** create sample dir %s" % FLAGS.sample_dir)
    os.makedirs(FLAGS.sample_dir)
  if not os.path.exists(FLAGS.checkpoint_dir):
    print("*** create checkpoint dir %s" % FLAGS.checkpoint_dir)
    os.makedirs(FLAGS.checkpoint_dir)

  # Write flags to log dir
  flags_file = open("%s/flags.txt" % FLAGS.log_dir, "w")
  for k, v in flags.FLAGS.__flags.items():
        line = '{}, {}'.format(k, v)
        print(line, file=flags_file)
  flags_file.close()

  if FLAGS.input_width is None:
    FLAGS.input_width = FLAGS.input_height
  if FLAGS.output_width is None:
    FLAGS.output_width = FLAGS.output_height

  if not os.path.exists(FLAGS.checkpoint_dir):
    os.makedirs(FLAGS.checkpoint_dir)
  if not os.path.exists(FLAGS.sample_dir):
    os.makedirs(FLAGS.sample_dir)

  run_config = tf.ConfigProto()
  run_config.gpu_options.allow_growth=True

  # load model
  fid.create_inception_graph(FLAGS.incept_path)

  with tf.Session(config=run_config) as sess:
    # get querry tensor
    if FLAGS.dataset == 'mnist':
      dcgan = DCGAN(
          sess,
          input_width=FLAGS.input_width,
          input_height=FLAGS.input_height,
          output_width=FLAGS.output_width,
          output_height=FLAGS.output_height,
          batch_size=FLAGS.batch_size,
          batch_size_m=FLAGS.batch_size_m,
          y_dim=10,
          c_dim=1,
          dataset_name=FLAGS.dataset,
          input_fname_pattern=FLAGS.input_fname_pattern,
          is_crop=FLAGS.is_crop,
          checkpoint_dir=FLAGS.checkpoint_dir,
          sample_dir=FLAGS.sample_dir,
          log_dir=FLAGS.log_dir,
          stats_path=FLAGS.stats_path,
          data_path=FLAGS.data_path,
          fid_n_samples=FLAGS.fid_n_samples,
          fid_sample_batchsize=FLAGS.fid_sample_batchsize,
          fid_batch_size=FLAGS.fid_batch_size,
          fid_verbose=FLAGS.fid_verbose,
          beta1=FLAGS.beta1)
    else:
      dcgan = DCGAN(
          sess,
          input_width=FLAGS.input_width,
          input_height=FLAGS.input_height,
          output_width=FLAGS.output_width,
          output_height=FLAGS.output_height,
          batch_size=FLAGS.batch_size,
          c_dim=FLAGS.c_dim,
          dataset_name=FLAGS.dataset,
          input_fname_pattern=FLAGS.input_fname_pattern,
          is_crop=FLAGS.is_crop,
          load_checkpoint=FLAGS.load_checkpoint,
          counter_start=FLAGS.counter_start,
          checkpoint_dir=FLAGS.checkpoint_dir,
          sample_dir=FLAGS.sample_dir,
          log_dir=FLAGS.log_dir,
          stats_path=FLAGS.stats_path,
          data_path=FLAGS.data_path,
          fid_n_samples=FLAGS.fid_n_samples,
          fid_sample_batchsize=FLAGS.fid_sample_batchsize,
          fid_batch_size=FLAGS.fid_batch_size,
          fid_verbose=FLAGS.fid_verbose,
          beta1=FLAGS.beta1)

    if FLAGS.is_train:
      dcgan.train(FLAGS)
    else:
      if not dcgan.load(FLAGS.checkpoint_dir):
        raise Exception("[!] Train a model first, then run test mode")


    # to_json("./web/js/layers.js", [dcgan.h0_w, dcgan.h0_b, dcgan.g_bn0],
    #                 [dcgan.h1_w, dcgan.h1_b, dcgan.g_bn1],
    #                 [dcgan.h2_w, dcgan.h2_b, dcgan.g_bn2],
    #                 [dcgan.h3_w, dcgan.h3_b, dcgan.g_bn3],
    #                 [dcgan.h4_w, dcgan.h4_b, None])

    # Below is codes for visualization
    #OPTION = 4
    #visualize(sess, dcgan, FLAGS, OPTION)

if __name__ == '__main__':
  tf.app.run()
