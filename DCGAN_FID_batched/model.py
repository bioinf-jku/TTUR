import os
import time
import math
from glob import glob
import tensorflow as tf
import numpy as np
from random import sample

from ops import *
from utils import *

# import fid
import fid

def conv_out_size_same(size, stride):
  return int(math.ceil(float(size) / float(stride)))

class DCGAN(object):
  def __init__(self, sess, input_height=108, input_width=108, is_crop=True,
         batch_size=64, sample_num = 64,
         output_height=64, output_width=64,
         y_dim=None, z_dim=100, gf_dim=64, df_dim=64,
         gfc_dim=1024, dfc_dim=1024, c_dim=3,
         dataset_name='default',
         input_fname_pattern='*.jpg',
         load_checkpoint=False, counter_start=0,
         checkpoint_dir=None,
         sample_dir=None,
         log_dir=None,
         stats_path=None,
         data_path=None,
         fid_n_samples=10000,
         fid_sample_batchsize=5000,
         fid_batch_size=500,
         fid_verbose=False,
         beta1=0.5):
    """

    Args:
      sess: TensorFlow session
      batch_size: The size of batch. Should be specified before training.
      y_dim: (optional) Dimension of dim for y. [None]
      z_dim: (optional) Dimension of dim for Z. [100]
      gf_dim: (optional) Dimension of gen filters in first conv layer. [64]
      df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
      gfc_dim: (optional) Dimension of gen units for for fully connected layer. [1024]
      dfc_dim: (optional) Dimension of discrim units for fully connected layer. [1024]
      c_dim: (optional) Dimension of image color. For grayscale input, set to 1. [3]
    """

    self.sess = sess
    self.is_crop = is_crop
    self.is_grayscale = (c_dim == 1)

    self.batch_size = batch_size
    self.sample_num = sample_num

    self.input_height = input_height
    self.input_width = input_width
    self.output_height = output_height
    self.output_width = output_width

    self.y_dim = y_dim
    self.z_dim = z_dim

    self.gf_dim = gf_dim
    self.df_dim = df_dim

    self.gfc_dim = gfc_dim
    self.dfc_dim = dfc_dim

    self.c_dim = c_dim

    # Batch normalization : deals with poor initialization helps gradient flow
    self.d_bn1 = batch_norm(name='d_bn1')
    self.d_bn2 = batch_norm(name='d_bn2')
    self.d_bn3 = batch_norm(name='d_bn3')

    self.g_bn0 = batch_norm(name='g_bn0')
    self.g_bn1 = batch_norm(name='g_bn1')
    self.g_bn2 = batch_norm(name='g_bn2')
    self.g_bn3 = batch_norm(name='g_bn3')

    self.dataset_name = dataset_name
    self.input_fname_pattern = input_fname_pattern
    self.load_checkpoint = load_checkpoint
    self.checkpoint_dir = checkpoint_dir
    self.counter_start = counter_start
    self.log_dir = log_dir
    self.stats_path = stats_path
    self.data_path = data_path
    self.fid_n_samples=fid_n_samples
    self.fid_sample_batchsize=fid_sample_batchsize
    self.fid_batch_size = fid_batch_size
    self.fid_verbose = fid_verbose

    self.beta1 = beta1

    print("build model.. ", end="", flush=True)
    self.build_model()
    print("ok")

  # Model
  def build_model(self):

    # Learning rate
    self.learning_rate_d = tf.Variable(0.0, trainable=False)
    self.learning_rate_g = tf.Variable(0.0, trainable=False)

    # Placeholders

    if self.is_crop:
      image_dims = [self.output_height, self.output_width, self.c_dim]
    else:
      image_dims = [self.input_height, self.input_width, self.c_dim]

    self.inputs = tf.placeholder(
      tf.float32, [self.batch_size] + image_dims, name='real_images')
    self.sample_inputs = tf.placeholder(
      tf.float32, [self.sample_num] + image_dims, name='sample_inputs')

    self.z = tf.placeholder(
      tf.float32, [None, self.z_dim], name='z')
    self.z_sum = tf.summary.histogram("z", self.z)

    self.z_fid = tf.placeholder(
      tf.float32, [None, self.z_dim], name='z_fid')

    self.fid = tf.Variable(0.0, trainable=False)

    # Inputs
    inputs = self.inputs
    sample_inputs = self.sample_inputs

    # Discriminator and generator
    if self.y_dim:
      print()
      print("Conditional GAN for MNIST not supported.")
      raise SystemExit()

    else:
      self.G = self.generator(self.z, batch_size=self.batch_size)
      self.D_real, self.D_logits_real = self.discriminator(inputs)

      self.sampler_fid = self.sampler_func(self.z_fid, self.fid_sample_batchsize)
      self.sampler = self.sampler_func(self.z, self.batch_size)
      self.D_fake, self.D_logits_fake = self.discriminator(self.G, reuse=True)

    # Summaries
    self.d_real_sum = tf.summary.histogram("d_real", self.D_real)
    self.d_fake_sum = tf.summary.histogram("d_fake", self.D_fake)
    self.G_sum = tf.summary.image("G", self.G)

    def sigmoid_cross_entropy_with_logits(x, y):
      try:
        return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=y)
      except:
        return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, targets=y)

    # Discriminator Loss Real
    self.d_loss_real = tf.reduce_mean(
      sigmoid_cross_entropy_with_logits(self.D_logits_real, tf.ones_like(self.D_real)))
    # Discriminator Loss Fake
    self.d_loss_fake = tf.reduce_mean(
      sigmoid_cross_entropy_with_logits(self.D_logits_fake, tf.zeros_like(self.D_fake)))
    # Generator Loss
    self.g_loss = tf.reduce_mean(
      sigmoid_cross_entropy_with_logits(self.D_logits_fake, tf.ones_like(self.D_fake)))

    self.d_loss_real_sum = tf.summary.scalar("d_loss_real", self.d_loss_real)
    self.d_loss_fake_sum = tf.summary.scalar("d_loss_fake", self.d_loss_fake)

    # Discriminator Loss Combined
    self.d_loss = self.d_loss_real + self.d_loss_fake

    self.g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)
    self.d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)

    self.lrate_sum_d = tf.summary.scalar('learning rate d', self.learning_rate_d)
    self.lrate_sum_g = tf.summary.scalar('learning rate g', self.learning_rate_g)

    self.fid_sum = tf.summary.scalar("FID", self.fid)

    # Variables
    t_vars = tf.trainable_variables()

    self.d_vars = [var for var in t_vars if 'd_' in var.name]
    self.g_vars = [var for var in t_vars if 'g_' in var.name]

    # Train optimizers
    opt_d = tf.train.AdamOptimizer(self.learning_rate_d, beta1=self.beta1)
    opt_g = tf.train.AdamOptimizer(self.learning_rate_g, beta1=self.beta1)

    # Discriminator
    grads_and_vars = opt_d.compute_gradients(self.d_loss, var_list=self.d_vars)
    grads = []
    self.d_optim = opt_d.apply_gradients(grads_and_vars)

    # Gradient summaries discriminator
    sum_grad_d = []
    for i, (grad, vars_) in enumerate(grads_and_vars):
      grad_l2 = tf.sqrt(tf.reduce_sum(tf.square(grad)))
      sum_grad_d.append(tf.summary.scalar("grad_l2_d_%d_%s" % (i, vars_.name), grad_l2))

    # Generator
    grads_and_vars = opt_g.compute_gradients(self.g_loss, var_list=self.g_vars)
    self.g_optim = opt_g.apply_gradients(grads_and_vars)

    # Gradient summaries generator
    sum_grad_g = []
    for i, (grad, vars_) in enumerate(grads_and_vars):
      grad_l2 = tf.sqrt(tf.reduce_sum(tf.square(grad)))
      sum_grad_g.append(tf.summary.scalar("grad_l2_g_%d_%s" % (i, vars_.name), grad_l2))

    # Init:
    tf.global_variables_initializer().run()

    # Summaries
    self.g_sum = tf.summary.merge([self.z_sum, self.d_fake_sum,
      self.G_sum, self.d_loss_fake_sum, self.g_loss_sum, self.lrate_sum_g] + sum_grad_g)
    self.d_sum = tf.summary.merge(
        [self.z_sum, self.d_real_sum, self.d_loss_real_sum, self.d_loss_sum, self.lrate_sum_d] + sum_grad_d)
    self.writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)


    # Checkpoint saver
    self.saver = tf.train.Saver()

    # check if fid_sample_batchsize is a multiple of fid_n_samples
    if not (self.fid_n_samples % self.fid_sample_batchsize == 0):
      new_bs = self.fid_n_samples // self.fid_sample_batchsize
      n_old =  self.fid_n_samples
      self.fid_n_samples = new_bs * self.fid_sample_batchsize
      print("""!WARNING: fid_sample_batchsize is not a multiple of fid_n_samples.
      Number of generated sample will be adjusted form %d to %d """ % (n_old, self.fid_n_samples))

  # Train model
  def train(self, config):
    """Train DCGAN"""

    print("load train stats.. ", end="", flush=True)
    # load precalculated training set statistics
    f = np.load(self.stats_path)
    mu_real, sigma_real = f['mu'][:], f['sigma'][:]
    f.close()
    print("ok")

    if config.dataset == 'mnist':
      print("scan files", end=" ", flush=True)
      data_X, data_y = self.load_mnist()
    else:
      if (config.dataset == "celebA") or (config.dataset == "cifar10"):
        print("scan files", end=" ", flush=True)
        data = glob(os.path.join(self.data_path, self.input_fname_pattern))
      else:
        if config.dataset == "lsun":
          print("scan files")
          data = []
          for i in range(304):
            print("\r%d" % i, end="", flush=True)
            data += glob(os.path.join(self.data_path, str(i), self.input_fname_pattern))
        else:
          print("Please specify dataset in run.sh [mnist, celebA, lsun, cifar10]")
          raise SystemExit()

    print()
    print("%d images found" % len(data))

    # Z sample
    #sample_z = np.random.normal(0, 1.0, size=(self.sample_num , self.z_dim))
    sample_z = np.random.uniform(-1.0, 1.0, size=(self.sample_num , self.z_dim))

    # Input samples
    sample_files = data[0:self.sample_num]
    sample = [
          get_image(sample_file,
                    input_height=self.input_height,
                    input_width=self.input_width,
                    resize_height=self.output_height,
                    resize_width=self.output_width,
                    is_crop=self.is_crop,
                    is_grayscale=self.is_grayscale) for sample_file in sample_files]
    if (self.is_grayscale):
      sample_inputs = np.array(sample).astype(np.float32)[:, :, :, None]
    else:
      sample_inputs = np.array(sample).astype(np.float32)

    if self.load_checkpoint:
      if self.load(self.checkpoint_dir):
        print(" [*] Load SUCCESS")
      else:
        print(" [!] Load failed...")

    # Batch preparing
    batch_nums = min(len(data), config.train_size) // config.batch_size
    data_idx = list(range(len(data)))

    counter = self.counter_start

    start_time = time.time()

    # Loop over epochs
    for epoch in range(config.epoch):

      # Assign learning rates for d and g
      lrate =  config.learning_rate_d # * (config.lr_decay_rate_d ** epoch)
      self.sess.run(tf.assign(self.learning_rate_d, lrate))
      lrate =  config.learning_rate_g # * (config.lr_decay_rate_g ** epoch)
      self.sess.run(tf.assign(self.learning_rate_g, lrate))

      # Shuffle the data indices
      np.random.shuffle(data_idx)

      # Loop over batches
      for batch_idx in range(batch_nums):

        # Prepare batch
        idx = data_idx[batch_idx * config.batch_size:(batch_idx + 1) * config.batch_size]
        batch = [
              get_image(data[i],
                        input_height=self.input_height,
                        input_width=self.input_width,
                        resize_height=self.output_height,
                        resize_width=self.output_width,
                        is_crop=self.is_crop,
                        is_grayscale=self.is_grayscale) for i in idx]
        if (self.is_grayscale):
          batch_images = np.array(batch).astype(np.float32)[:, :, :, None]
        else:
          batch_images = np.array(batch).astype(np.float32)

        #batch_z = np.random.normal(0, 1.0, size=(config.batch_size , self.z_dim)).astype(np.float32)
        batch_z = np.random.uniform(-1.0, 1.0, size=(config.batch_size , self.z_dim)).astype(np.float32)

        # Update D network
        _, summary_str = self.sess.run([self.d_optim, self.d_sum],
                                       feed_dict={self.inputs: batch_images,
                                                  self.z: batch_z})
        if np.mod(counter, 20) == 0:
          self.writer.add_summary(summary_str, counter)

        # Update G network
        _, summary_str = self.sess.run([self.g_optim, self.g_sum],
                                       feed_dict={self.z: batch_z})
        if np.mod(counter, 20) == 0:
          self.writer.add_summary(summary_str, counter)

        errD_fake = self.d_loss_fake.eval({ self.z: batch_z })
        errD_real = self.d_loss_real.eval({ self.inputs: batch_images })
        errG = self.g_loss.eval({self.z: batch_z})

        # Print
        if np.mod(counter, 100) == 0:
          print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
            % (epoch, batch_idx, batch_nums, time.time() - start_time, errD_fake+errD_real, errG))

        # Save generated samples and FID
        if np.mod(counter, config.fid_eval_steps) == 0:

          # Save
          try:
            samples, d_loss, g_loss = self.sess.run(
                [self.sampler, self.d_loss, self.g_loss],
                feed_dict={self.z: sample_z,
                           self.inputs: sample_inputs})
            save_images(samples, [8, 8], '{}/train_{:02d}_{:04d}.png'.format(config.sample_dir, epoch, batch_idx))
            print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss))
          except Exception as e:
            print(e)
            print("sample image error!")

          # FID
          print("samples for incept", end="", flush=True)

          samples = np.zeros((self.fid_n_samples, self.output_height, self.output_width, 3))
          n_batches = self.fid_n_samples // self.fid_sample_batchsize
          lo = 0
          for btch in range(n_batches):
            print("\rsamples for incept %d/%d" % (btch + 1, n_batches), end=" ", flush=True)
            #sample_z_fid = np.random.normal(0, 1.0, size=(self.fid_sample_batchsize, self.z_dim))
            sample_z_fid = np.random.uniform(-1.0, 1.0, size=(self.fid_sample_batchsize, self.z_dim))
            samples[lo:(lo+self.fid_sample_batchsize)] = self.sess.run( self.sampler_fid,
                                     feed_dict={self.z_fid: sample_z_fid})
            lo += self.fid_sample_batchsize

          samples = (samples + 1.) * 127.5
          print("ok")

          mu_gen, sigma_gen = fid.calculate_activation_statistics( samples,
                                                           self.sess,
                                                           batch_size=self.fid_batch_size,
                                                           verbose=self.fid_verbose)

          print("calculate FID:", end=" ", flush=True)
          try:
              FID = fid.calculate_frechet_distance(mu_gen, sigma_gen, mu_real, sigma_real)
          except Exception as e:
              print(e)
              FID=500

          print(FID)

          # Update event log with FID
          self.sess.run(tf.assign(self.fid, FID))
          summary_str = self.sess.run(self.fid_sum)
          self.writer.add_summary(summary_str, counter)

        # Save checkpoint
        if (counter != 0) and (np.mod(counter, 2000) == 0):
          self.save(config.checkpoint_dir, counter)

        counter += 1

  # Discriminator
  def discriminator(self, image, y=None, reuse=False):
    with tf.variable_scope("discriminator") as scope:
      if reuse:
        scope.reuse_variables()
      h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))
      h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim*2, name='d_h1_conv')))
      h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim*4, name='d_h2_conv')))
      h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim*8, name='d_h3_conv')))
      h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_h3_lin')
      return tf.nn.sigmoid(h4), h4


  # Generator
  def generator(self, z, y=None, batch_size=None):
    with tf.variable_scope("generator") as scope:

      if not self.y_dim:
        s_h, s_w = self.output_height, self.output_width
        s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
        s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
        s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
        s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)

        # Project `z` and reshape
        self.z_, self.h0_w, self.h0_b = linear(
            z, self.gf_dim*8*s_h16*s_w16, 'g_h0_lin', with_w=True)
        self.h0 = tf.reshape(
            self.z_, [-1, s_h16, s_w16, self.gf_dim * 8])
        h0 = tf.nn.relu(self.g_bn0(self.h0))

        # Deconv
        self.h1, self.h1_w, self.h1_b = deconv2d(
            h0, [batch_size, s_h8, s_w8, self.gf_dim*4], name='g_h1', with_w=True)
        h1 = tf.nn.relu(self.g_bn1(self.h1))
        h2, self.h2_w, self.h2_b = deconv2d(
            h1, [batch_size, s_h4, s_w4, self.gf_dim*2], name='g_h2', with_w=True)
        h2 = tf.nn.relu(self.g_bn2(h2))
        h3, self.h3_w, self.h3_b = deconv2d(
            h2, [batch_size, s_h2, s_w2, self.gf_dim*1], name='g_h3', with_w=True)
        h3 = tf.nn.relu(self.g_bn3(h3))
        h4, self.h4_w, self.h4_b = deconv2d(
            h3, [batch_size, s_h, s_w, self.c_dim], name='g_h4', with_w=True)

        return tf.nn.tanh(h4)


  # Sampler
  def sampler_func(self, z, batch_size, y=None):
    with tf.variable_scope("generator") as scope:
      scope.reuse_variables()

      if not self.y_dim:
        s_h, s_w = self.output_height, self.output_width
        s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
        s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
        s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
        s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)

        # Project `z` and reshape
        h0 = tf.reshape(
            linear(z, self.gf_dim*8*s_h16*s_w16, 'g_h0_lin'),
            [-1, s_h16, s_w16, self.gf_dim * 8])
        h0 = tf.nn.relu(self.g_bn0(h0, train=False))

        # Deconv
        h1 = deconv2d(h0, [batch_size, s_h8, s_w8, self.gf_dim*4], name='g_h1')
        h1 = tf.nn.relu(self.g_bn1(h1, train=False))
        h2 = deconv2d(h1, [batch_size, s_h4, s_w4, self.gf_dim*2], name='g_h2')
        h2 = tf.nn.relu(self.g_bn2(h2, train=False))
        h3 = deconv2d(h2, [batch_size, s_h2, s_w2, self.gf_dim*1], name='g_h3')
        h3 = tf.nn.relu(self.g_bn3(h3, train=False))
        h4 = deconv2d(h3, [batch_size, s_h, s_w, self.c_dim], name='g_h4')

        return tf.nn.tanh(h4)


  @property
  def model_dir(self):
    return "{}_{}_{}_{}".format(
        self.dataset_name, self.batch_size,
        self.output_height, self.output_width)

  # Save checkpoint
  def save(self, checkpoint_dir, step):
    model_name = "DCGAN.model"
    checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

    if not os.path.exists(checkpoint_dir):
      os.makedirs(checkpoint_dir)

    self.saver.save(self.sess,
            os.path.join(checkpoint_dir, model_name),
            global_step=step)

  # Load checkpoint
  def load(self, checkpoint_dir):
    print(" [*] Reading checkpoints...")
    checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
      self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
      print(" [*] Success to read {}".format(ckpt_name))
      return True
    else:
      print(" [*] Failed to find a checkpoint")
      return False

