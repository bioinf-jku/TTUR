import os, sys
sys.path.append(os.getcwd())

import time
import functools

import numpy as np
import tensorflow as tf

import tflib as lib
import tflib.ops.linear
import tflib.ops.conv2d
import tflib.ops.batchnorm
import tflib.ops.deconv2d
import tflib.save_images
import tflib.data_loader
import tflib.ops.layernorm
import tflib.plot

import fid

DATA_DIR = 'data/lsun'
DATASET = "lsun" # celeba, cifar10, svhn, lsun
if len(DATA_DIR) == 0:
    raise Exception('Please specify path to data directory in gan_64x64.py!')

# Download the Inception model from here
# http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz
# And set the path to the extracted model here:
INCEPTION_DIR = "inception-2015-12-05"

# Path to the real world statistics file.
STAT_FILE = "stats/fid_stats_lsun.npz"

MODE = 'wgan-gp' # dcgan, wgan, wgan-gp, lsgan
DIM = 64 # Model dimensionality

# Settings for TTUR and orig
TTUR = True
if TTUR:
  CRITIC_ITERS = 1 # How many iterations to train the critic for
  D_LR = 0.0003
  G_LR = 0.0001
  BETA1_D = 0.0
  BETA1_G = 0.0
  FID_STEP = 1000 # FID evaluation every FID_STEP
  ITERS = 100000 # How many iterations to train for
else:
  CRITIC_ITERS = 5 # How many iterations to train the critic for
  D_LR = 0.0005
  G_LR = 0.0005
  BETA1_D = 0.0
  BETA1_G = 0.0
  FID_STEP = 333 # FID evaluation every FID_STEP
  ITERS = 25009 # How many iterations to train for

OUTPUT_STEP = 200 # Print output every OUTPUT_STEP
SAVE_SAMPLES_STEP = 200 # Generate and save samples every SAVE_SAMPLES_STEP

LOAD_CHECKPOINT = FALSE
DIR = "mmdd_hhmmss_lrd_lrg"
ITER_START = 0

# Switch on and off batchnormalizaton for the discriminator
# and the generator. Default is on for both.
BN_D=True
BN_G=True

# Log subdirectories are automatically created from
# the above settings and the current timestamp.
CHECKPOINT_STEP = FID_STEP
LOG_DIR = "logs" # Directory for Tensorboard events, checkpoints and samples
N_GPUS = 1 # Number of GPUs
BATCH_SIZE = 64 # Batch size. Must be a multiple of N_GPUS
LAMBDA = 10 # Gradient penalty lambda hyperparameter
OUTPUT_DIM = DIM * DIM * 3 # Number of pixels in each iamge

if not LOAD_CHECKPOINT:
    timestamp = time.strftime("%m%d_%H%M%S")
    DIR = "%s_%6f_%.6f" % (timestamp, D_LR, G_LR)

LOG_DIR = os.path.join(LOG_DIR, DIR)
SAMPLES_DIR = os.path.join(LOG_DIR, "samples")
CHECKPOINT_DIR = os.path.join(LOG_DIR, "checkpoints")
TBOARD_DIR = os.path.join(LOG_DIR, "logs")

# Create directories if necessary
if not os.path.exists(SAMPLES_DIR):
  print("*** create sample dir %s" % SAMPLES_DIR)
  os.makedirs(SAMPLES_DIR)
if not os.path.exists(CHECKPOINT_DIR):
  print("*** create checkpoint dir %s" % CHECKPOINT_DIR)
  os.makedirs(CHECKPOINT_DIR)
if not os.path.exists(TBOARD_DIR):
  print("*** create tboard dir %s" % TBOARD_DIR)
  os.makedirs(TBOARD_DIR)

# FID evaluation.
FID_EVAL_SIZE = 50000 # Number of samples for evaluation
FID_SAMPLE_BATCH_SIZE = 1000  # Batch size of generating samples, lower to save GPU memory
FID_BATCH_SIZE = 200 # Batch size for final FID calculation i.e. inception propagation etc.

# Load checkpoint
# from https://github.com/carpedm20/DCGAN-tensorflow/blob/master/model.py
def load_checkpoint(session, saver, checkpoint_dir):
  print(" [*] Reading checkpoints...")
  ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
  if ckpt and ckpt.model_checkpoint_path:
    ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
    saver.restore(session, os.path.join(checkpoint_dir, ckpt_name))
    print(" [*] Success to read {}".format(ckpt_name))
    return True
  else:
    print(" [*] Failed to find a checkpoint")
    return False

lib.print_model_settings(locals().copy(), LOG_DIR)

def GeneratorAndDiscriminator():
    """
    Choose which generator and discriminator architecture to use by
    uncommenting one of these lines.
    """

    # For actually generating decent samples, use this one
    return GoodGenerator, GoodDiscriminator

    # Baseline (G: DCGAN, D: DCGAN)
    #return DCGANGenerator, DCGANDiscriminator

    # No BN and constant number of filts in G
    # return WGANPaper_CrippledDCGANGenerator, DCGANDiscriminator

    # 512-dim 4-layer ReLU MLP G
    # return FCGenerator, DCGANDiscriminator

    # No normalization anywhere
    # return functools.partial(DCGANGenerator, bn=False), functools.partial(DCGANDiscriminator, bn=False)

    # Gated multiplicative nonlinearities everywhere
    # return MultiplicativeDCGANGenerator, MultiplicativeDCGANDiscriminator

    # tanh nonlinearities everywhere
    # return functools.partial(DCGANGenerator, bn=True, nonlinearity=tf.tanh), \
    #        functools.partial(DCGANDiscriminator, bn=True, nonlinearity=tf.tanh)

    # 101-layer ResNet G and D
    #return ResnetGenerator, ResnetDiscriminator

    raise Exception('You must choose an architecture!')

DEVICES = ['/gpu:{}'.format(i) for i in range(N_GPUS)]

def LeakyReLU(x, alpha=0.2):
    return tf.maximum(alpha*x, x)

def ReLULayer(name, n_in, n_out, inputs):
    output = lib.ops.linear.Linear(name+'.Linear', n_in, n_out, inputs, initialization='he')
    return tf.nn.relu(output)

def LeakyReLULayer(name, n_in, n_out, inputs):
    output = lib.ops.linear.Linear(name+'.Linear', n_in, n_out, inputs, initialization='he')
    return LeakyReLU(output)

def Normalize(name, axes, inputs):
    if ('Discriminator' in name) and (MODE == 'wgan-gp'):
        if axes != [0,2,3]:
            raise Exception('Layernorm over non-standard axes is unsupported')
        return lib.ops.layernorm.Layernorm(name,[1,2,3],inputs)
    else:
        #return lib.ops.layernorm.Layernorm(name,[1,2,3],inputs)
        return lib.ops.batchnorm.Batchnorm(name,axes,inputs,fused=True)

def pixcnn_gated_nonlinearity(a, b):
    return tf.sigmoid(a) * tf.tanh(b)

def SubpixelConv2D(*args, **kwargs):
    kwargs['output_dim'] = 4*kwargs['output_dim']
    output = lib.ops.conv2d.Conv2D(*args, **kwargs)
    output = tf.transpose(output, [0,2,3,1])
    output = tf.depth_to_space(output, 2)
    output = tf.transpose(output, [0,3,1,2])
    return output

def ConvMeanPool(name, input_dim, output_dim, filter_size, inputs, he_init=True, biases=True):
    output = lib.ops.conv2d.Conv2D(name, input_dim, output_dim, filter_size, inputs, he_init=he_init, biases=biases)
    output = tf.add_n([output[:,:,::2,::2], output[:,:,1::2,::2], output[:,:,::2,1::2], output[:,:,1::2,1::2]]) / 4.
    return output

def MeanPoolConv(name, input_dim, output_dim, filter_size, inputs, he_init=True, biases=True):
    output = inputs
    output = tf.add_n([output[:,:,::2,::2], output[:,:,1::2,::2], output[:,:,::2,1::2], output[:,:,1::2,1::2]]) / 4.
    output = lib.ops.conv2d.Conv2D(name, input_dim, output_dim, filter_size, output, he_init=he_init, biases=biases)
    return output

def UpsampleConv(name, input_dim, output_dim, filter_size, inputs, he_init=True, biases=True):
    output = inputs
    output = tf.concat([output, output, output, output], axis=1)
    output = tf.transpose(output, [0,2,3,1])
    output = tf.depth_to_space(output, 2)
    output = tf.transpose(output, [0,3,1,2])
    output = lib.ops.conv2d.Conv2D(name, input_dim, output_dim, filter_size, output, he_init=he_init, biases=biases)
    return output

def BottleneckResidualBlock(name, input_dim, output_dim, filter_size, inputs, resample=None, he_init=True):
    """
    resample: None, 'down', or 'up'
    """
    if resample=='down':
        conv_shortcut = functools.partial(lib.ops.conv2d.Conv2D, stride=2)
        conv_1        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=input_dim, output_dim=input_dim//2)
        conv_1b       = functools.partial(lib.ops.conv2d.Conv2D, input_dim=input_dim//2, output_dim=output_dim//2, stride=2)
        conv_2        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=output_dim//2, output_dim=output_dim)
    elif resample=='up':
        conv_shortcut = SubpixelConv2D
        conv_1        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=input_dim, output_dim=input_dim//2)
        conv_1b       = functools.partial(lib.ops.deconv2d.Deconv2D, input_dim=input_dim//2, output_dim=output_dim//2)
        conv_2        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=output_dim//2, output_dim=output_dim)
    elif resample==None:
        conv_shortcut = lib.ops.conv2d.Conv2D
        conv_1        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=input_dim,  output_dim=input_dim//2)
        conv_1b       = functools.partial(lib.ops.conv2d.Conv2D, input_dim=input_dim//2,  output_dim=output_dim//2)
        conv_2        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=input_dim//2, output_dim=output_dim)

    else:
        raise Exception('invalid resample value')

    if output_dim==input_dim and resample==None:
        shortcut = inputs # Identity skip-connection
    else:
        shortcut = conv_shortcut(name+'.Shortcut', input_dim=input_dim, output_dim=output_dim, filter_size=1,
                                 he_init=False, biases=True, inputs=inputs)

    output = inputs
    output = tf.nn.relu(output)
    output = conv_1(name+'.Conv1', filter_size=1, inputs=output, he_init=he_init)
    output = tf.nn.relu(output)
    output = conv_1b(name+'.Conv1B', filter_size=filter_size, inputs=output, he_init=he_init)
    output = tf.nn.relu(output)
    output = conv_2(name+'.Conv2', filter_size=1, inputs=output, he_init=he_init, biases=False)
    output = Normalize(name+'.BN', [0,2,3], output)

    return shortcut + (0.3*output)

def ResidualBlock(name, input_dim, output_dim, filter_size, inputs, resample=None, he_init=True, bn=False):
    """
    resample: None, 'down', or 'up'
    """
    if resample=='down':
        conv_shortcut = MeanPoolConv
        conv_1        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=input_dim, output_dim=input_dim)
        conv_2        = functools.partial(ConvMeanPool, input_dim=input_dim, output_dim=output_dim)
    elif resample=='up':
        conv_shortcut = UpsampleConv
        conv_1        = functools.partial(UpsampleConv, input_dim=input_dim, output_dim=output_dim)
        conv_2        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=output_dim, output_dim=output_dim)
    elif resample==None:
        conv_shortcut = lib.ops.conv2d.Conv2D
        conv_1        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=input_dim,  output_dim=input_dim)
        conv_2        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=input_dim, output_dim=output_dim)
    else:
        raise Exception('invalid resample value')

    if output_dim==input_dim and resample==None:
        shortcut = inputs # Identity skip-connection
    else:
        shortcut = conv_shortcut(name+'.Shortcut', input_dim=input_dim, output_dim=output_dim, filter_size=1,
                                 he_init=False, biases=True, inputs=inputs)

    output = inputs
    if bn:
      output = Normalize(name+'.BN1', [0,2,3], output)
    output = tf.nn.relu(output)
    output = conv_1(name+'.Conv1', filter_size=filter_size, inputs=output, he_init=he_init, biases=False)
    if bn:
      output = Normalize(name+'.BN2', [0,2,3], output)
    output = tf.nn.relu(output)
    output = conv_2(name+'.Conv2', filter_size=filter_size, inputs=output, he_init=he_init)

    return shortcut + output


# ! Generators

def GoodGenerator(n_samples, noise=None, dim=DIM, nonlinearity=tf.nn.relu, bn=BN_G):
    if noise is None:
        noise = tf.random_normal([n_samples, 128])

    ## supports 32x32 images
    fact = DIM // 16

    output = lib.ops.linear.Linear('Generator.Input', 128, fact*fact*8*dim, noise)
    output = tf.reshape(output, [-1, 8*dim, fact, fact])
    output = ResidualBlock('Generator.Res1', 8*dim, 8*dim, 3, output, resample='up', bn=bn)
    output = ResidualBlock('Generator.Res2', 8*dim, 4*dim, 3, output, resample='up', bn=bn)
    output = ResidualBlock('Generator.Res3', 4*dim, 2*dim, 3, output, resample='up', bn=bn)
    output = ResidualBlock('Generator.Res4', 2*dim, 1*dim, 3, output, resample='up', bn=bn)
    if bn:
      output = Normalize('Generator.OutputN', [0,2,3], output)
    output = tf.nn.relu(output)
    output = lib.ops.conv2d.Conv2D('Generator.Output', 1*dim, 3, 3, output)
    output = tf.tanh(output)

    return tf.reshape(output, [-1, OUTPUT_DIM])

def FCGenerator(n_samples, noise=None, FC_DIM=512):
    if noise is None:
        noise = tf.random_normal([n_samples, 128])

    output = ReLULayer('Generator.1', 128, FC_DIM, noise)
    output = ReLULayer('Generator.2', FC_DIM, FC_DIM, output)
    output = ReLULayer('Generator.3', FC_DIM, FC_DIM, output)
    output = ReLULayer('Generator.4', FC_DIM, FC_DIM, output)
    output = lib.ops.linear.Linear('Generator.Out', FC_DIM, OUTPUT_DIM, output)

    output = tf.tanh(output)

    return output

def DCGANGenerator(n_samples, noise=None, dim=DIM, bn=True, nonlinearity=tf.nn.relu):
    lib.ops.conv2d.set_weights_stdev(0.02)
    lib.ops.deconv2d.set_weights_stdev(0.02)
    lib.ops.linear.set_weights_stdev(0.02)

    if noise is None:
        noise = tf.random_normal([n_samples, 128])

    output = lib.ops.linear.Linear('Generator.Input', 128, 4*4*8*dim, noise)
    output = tf.reshape(output, [-1, 8*dim, 4, 4])
    if bn:
        output = Normalize('Generator.BN1', [0,2,3], output)
    output = nonlinearity(output)

    output = lib.ops.deconv2d.Deconv2D('Generator.2', 8*dim, 4*dim, 5, output)
    if bn:
        output = Normalize('Generator.BN2', [0,2,3], output)
    output = nonlinearity(output)

    output = lib.ops.deconv2d.Deconv2D('Generator.3', 4*dim, 2*dim, 5, output)
    if bn:
        output = Normalize('Generator.BN3', [0,2,3], output)
    output = nonlinearity(output)

    output = lib.ops.deconv2d.Deconv2D('Generator.4', 2*dim, dim, 5, output)
    if bn:
        output = Normalize('Generator.BN4', [0,2,3], output)
    output = nonlinearity(output)

    output = lib.ops.deconv2d.Deconv2D('Generator.5', dim, 3, 5, output)
    output = tf.tanh(output)

    lib.ops.conv2d.unset_weights_stdev()
    lib.ops.deconv2d.unset_weights_stdev()
    lib.ops.linear.unset_weights_stdev()

    return tf.reshape(output, [-1, OUTPUT_DIM])

def WGANPaper_CrippledDCGANGenerator(n_samples, noise=None, dim=DIM):
    if noise is None:
        noise = tf.random_normal([n_samples, 128])

    output = lib.ops.linear.Linear('Generator.Input', 128, 4*4*dim, noise)
    output = tf.nn.relu(output)
    output = tf.reshape(output, [-1, dim, 4, 4])

    output = lib.ops.deconv2d.Deconv2D('Generator.2', dim, dim, 5, output)
    output = tf.nn.relu(output)

    output = lib.ops.deconv2d.Deconv2D('Generator.3', dim, dim, 5, output)
    output = tf.nn.relu(output)

    output = lib.ops.deconv2d.Deconv2D('Generator.4', dim, dim, 5, output)
    output = tf.nn.relu(output)

    output = lib.ops.deconv2d.Deconv2D('Generator.5', dim, 3, 5, output)
    output = tf.tanh(output)

    return tf.reshape(output, [-1, OUTPUT_DIM])

def ResnetGenerator(n_samples, noise=None, dim=DIM):
    if noise is None:
        noise = tf.random_normal([n_samples, 128])

    output = lib.ops.linear.Linear('Generator.Input', 128, 4*4*8*dim, noise)
    output = tf.reshape(output, [-1, 8*dim, 4, 4])

    for i in range(6):
        output = BottleneckResidualBlock('Generator.4x4_{}'.format(i), 8*dim, 8*dim, 3, output, resample=None)
    output = BottleneckResidualBlock('Generator.Up1', 8*dim, 4*dim, 3, output, resample='up')
    for i in range(6):
        output = BottleneckResidualBlock('Generator.8x8_{}'.format(i), 4*dim, 4*dim, 3, output, resample=None)
    output = BottleneckResidualBlock('Generator.Up2', 4*dim, 2*dim, 3, output, resample='up')
    for i in range(6):
        output = BottleneckResidualBlock('Generator.16x16_{}'.format(i), 2*dim, 2*dim, 3, output, resample=None)
    output = BottleneckResidualBlock('Generator.Up3', 2*dim, 1*dim, 3, output, resample='up')
    for i in range(6):
        output = BottleneckResidualBlock('Generator.32x32_{}'.format(i), 1*dim, 1*dim, 3, output, resample=None)
    output = BottleneckResidualBlock('Generator.Up4', 1*dim, dim//2, 3, output, resample='up')
    for i in range(5):
        output = BottleneckResidualBlock('Generator.64x64_{}'.format(i), dim//2, dim//2, 3, output, resample=None)

    output = lib.ops.conv2d.Conv2D('Generator.Out', dim//2, 3, 1, output, he_init=False)
    output = tf.tanh(output / 5.)

    return tf.reshape(output, [-1, OUTPUT_DIM])


def MultiplicativeDCGANGenerator(n_samples, noise=None, dim=DIM, bn=True):
    if noise is None:
        noise = tf.random_normal([n_samples, 128])

    output = lib.ops.linear.Linear('Generator.Input', 128, 4*4*8*dim*2, noise)
    output = tf.reshape(output, [-1, 8*dim*2, 4, 4])
    if bn:
        output = Normalize('Generator.BN1', [0,2,3], output)
    output = pixcnn_gated_nonlinearity(output[:,::2], output[:,1::2])

    output = lib.ops.deconv2d.Deconv2D('Generator.2', 8*dim, 4*dim*2, 5, output)
    if bn:
        output = Normalize('Generator.BN2', [0,2,3], output)
    output = pixcnn_gated_nonlinearity(output[:,::2], output[:,1::2])

    output = lib.ops.deconv2d.Deconv2D('Generator.3', 4*dim, 2*dim*2, 5, output)
    if bn:
        output = Normalize('Generator.BN3', [0,2,3], output)
    output = pixcnn_gated_nonlinearity(output[:,::2], output[:,1::2])

    output = lib.ops.deconv2d.Deconv2D('Generator.4', 2*dim, dim*2, 5, output)
    if bn:
        output = Normalize('Generator.BN4', [0,2,3], output)
    output = pixcnn_gated_nonlinearity(output[:,::2], output[:,1::2])

    output = lib.ops.deconv2d.Deconv2D('Generator.5', dim, 3, 5, output)
    output = tf.tanh(output)

    return tf.reshape(output, [-1, OUTPUT_DIM])

# ! Discriminators

def GoodDiscriminator(inputs, dim=DIM, bn=BN_D):
    output = tf.reshape(inputs, [-1, 3, DIM, DIM])
    output = lib.ops.conv2d.Conv2D('Discriminator.Input', 3, dim, 3, output, he_init=False)

    output = ResidualBlock('Discriminator.Res1', dim, 2*dim, 3, output, resample='down', bn=bn)
    output = ResidualBlock('Discriminator.Res2', 2*dim, 4*dim, 3, output, resample='down', bn=bn)
    output = ResidualBlock('Discriminator.Res3', 4*dim, 8*dim, 3, output, resample='down', bn=bn)
    output = ResidualBlock('Discriminator.Res4', 8*dim, 8*dim, 3, output, resample='down', bn=bn)

    output = tf.reshape(output, [-1, 4*4*8*dim])
    output = lib.ops.linear.Linear('Discriminator.Output', 4*4*8*dim, 1, output)

    return tf.reshape(output, [-1])

def MultiplicativeDCGANDiscriminator(inputs, dim=DIM, bn=True):
    output = tf.reshape(inputs, [-1, 3, DIM, DIM])

    output = lib.ops.conv2d.Conv2D('Discriminator.1', 3, dim*2, 5, output, stride=2)
    output = pixcnn_gated_nonlinearity(output[:,::2], output[:,1::2])

    output = lib.ops.conv2d.Conv2D('Discriminator.2', dim, 2*dim*2, 5, output, stride=2)
    if bn:
        output = Normalize('Discriminator.BN2', [0,2,3], output)
    output = pixcnn_gated_nonlinearity(output[:,::2], output[:,1::2])

    output = lib.ops.conv2d.Conv2D('Discriminator.3', 2*dim, 4*dim*2, 5, output, stride=2)
    if bn:
        output = Normalize('Discriminator.BN3', [0,2,3], output)
    output = pixcnn_gated_nonlinearity(output[:,::2], output[:,1::2])

    output = lib.ops.conv2d.Conv2D('Discriminator.4', 4*dim, 8*dim*2, 5, output, stride=2)
    if bn:
        output = Normalize('Discriminator.BN4', [0,2,3], output)
    output = pixcnn_gated_nonlinearity(output[:,::2], output[:,1::2])

    output = tf.reshape(output, [-1, 4*4*8*dim])
    output = lib.ops.linear.Linear('Discriminator.Output', 4*4*8*dim, 1, output)

    return tf.reshape(output, [-1])


def ResnetDiscriminator(inputs, dim=DIM):
    output = tf.reshape(inputs, [-1, 3, DIM, DIM])
    output = lib.ops.conv2d.Conv2D('Discriminator.In', 3, dim//2, 1, output, he_init=False)

    for i in range(5):
        output = BottleneckResidualBlock('Discriminator.64x64_{}'.format(i), dim//2, dim//2, 3, output, resample=None)
    output = BottleneckResidualBlock('Discriminator.Down1', dim//2, dim*1, 3, output, resample='down')
    for i in range(6):
        output = BottleneckResidualBlock('Discriminator.32x32_{}'.format(i), dim*1, dim*1, 3, output, resample=None)
    output = BottleneckResidualBlock('Discriminator.Down2', dim*1, dim*2, 3, output, resample='down')
    for i in range(6):
        output = BottleneckResidualBlock('Discriminator.16x16_{}'.format(i), dim*2, dim*2, 3, output, resample=None)
    output = BottleneckResidualBlock('Discriminator.Down3', dim*2, dim*4, 3, output, resample='down')
    for i in range(6):
        output = BottleneckResidualBlock('Discriminator.8x8_{}'.format(i), dim*4, dim*4, 3, output, resample=None)
    output = BottleneckResidualBlock('Discriminator.Down4', dim*4, dim*8, 3, output, resample='down')
    for i in range(6):
        output = BottleneckResidualBlock('Discriminator.4x4_{}'.format(i), dim*8, dim*8, 3, output, resample=None)

    output = tf.reshape(output, [-1, 4*4*8*dim])
    output = lib.ops.linear.Linear('Discriminator.Output', 4*4*8*dim, 1, output)

    return tf.reshape(output / 5., [-1])


def FCDiscriminator(inputs, FC_DIM=512, n_layers=3):
    output = LeakyReLULayer('Discriminator.Input', OUTPUT_DIM, FC_DIM, inputs)
    for i in range(n_layers):
        output = LeakyReLULayer('Discriminator.{}'.format(i), FC_DIM, FC_DIM, output)
    output = lib.ops.linear.Linear('Discriminator.Out', FC_DIM, 1, output)

    return tf.reshape(output, [-1])

def DCGANDiscriminator(inputs, dim=DIM, bn=True, nonlinearity=LeakyReLU):
    output = tf.reshape(inputs, [-1, 3, DIM, DIM])

    lib.ops.conv2d.set_weights_stdev(0.02)
    lib.ops.deconv2d.set_weights_stdev(0.02)
    lib.ops.linear.set_weights_stdev(0.02)

    output = lib.ops.conv2d.Conv2D('Discriminator.1', 3, dim, 5, output, stride=2)
    output = nonlinearity(output)

    output = lib.ops.conv2d.Conv2D('Discriminator.2', dim, 2*dim, 5, output, stride=2)
    if bn:
        output = Normalize('Discriminator.BN2', [0,2,3], output)
    output = nonlinearity(output)

    output = lib.ops.conv2d.Conv2D('Discriminator.3', 2*dim, 4*dim, 5, output, stride=2)
    if bn:
        output = Normalize('Discriminator.BN3', [0,2,3], output)
    output = nonlinearity(output)

    output = lib.ops.conv2d.Conv2D('Discriminator.4', 4*dim, 8*dim, 5, output, stride=2)
    if bn:
        output = Normalize('Discriminator.BN4', [0,2,3], output)
    output = nonlinearity(output)

    output = tf.reshape(output, [-1, 4*4*8*dim])
    output = lib.ops.linear.Linear('Discriminator.Output', 4*4*8*dim, 1, output)

    lib.ops.conv2d.unset_weights_stdev()
    lib.ops.deconv2d.unset_weights_stdev()
    lib.ops.linear.unset_weights_stdev()

    return tf.reshape(output, [-1])

Generator, Discriminator = GeneratorAndDiscriminator()

with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as session:

    all_real_data_conv = tf.placeholder(tf.int32, shape=[BATCH_SIZE, 3, DIM, DIM])
    if tf.__version__.startswith('1.'):
        split_real_data_conv = tf.split(all_real_data_conv, len(DEVICES))
    else:
        split_real_data_conv = tf.split(0, len(DEVICES), all_real_data_conv)
    gen_costs, disc_costs = [],[]

    for device_index, (device, real_data_conv) in enumerate(zip(DEVICES, split_real_data_conv)):
        with tf.device(device):

            real_data = tf.reshape(2*((tf.cast(real_data_conv, tf.float32)/255.)-.5), [BATCH_SIZE//len(DEVICES), OUTPUT_DIM])
            fake_data = Generator(BATCH_SIZE//len(DEVICES), bn=BN_G)

            disc_real = Discriminator(real_data)
            disc_fake = Discriminator(fake_data)

            if MODE == 'wgan':
                gen_cost = -tf.reduce_mean(disc_fake)
                disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)

            elif MODE == 'wgan-gp':
                gen_cost = -tf.reduce_mean(disc_fake)
                disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)

                alpha = tf.random_uniform(
                    shape=[BATCH_SIZE//len(DEVICES),1],
                    minval=0.,
                    maxval=1.
                )
                differences = fake_data - real_data
                interpolates = real_data + (alpha*differences)
                gradients = tf.gradients(Discriminator(interpolates, bn=BN_D), interpolates)[0]
                slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
                gradient_penalty = tf.reduce_mean((slopes-1.)**2)
                disc_cost += LAMBDA*gradient_penalty

            elif MODE == 'dcgan':
                try: # tf pre-1.0 (bottom) vs 1.0 (top)
                    gen_cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_fake,
                                                                                      labels=tf.ones_like(disc_fake)))
                    disc_cost =  tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_fake,
                                                                                        labels=tf.zeros_like(disc_fake)))
                    disc_cost += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_real,
                                                                                        labels=tf.ones_like(disc_real)))
                except Exception as e:
                    gen_cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(disc_fake, tf.ones_like(disc_fake)))
                    disc_cost =  tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(disc_fake, tf.zeros_like(disc_fake)))
                    disc_cost += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(disc_real, tf.ones_like(disc_real)))
                disc_cost /= 2.

            elif MODE == 'lsgan':
                gen_cost = tf.reduce_mean((disc_fake - 1)**2)
                disc_cost = (tf.reduce_mean((disc_real - 1)**2) + tf.reduce_mean((disc_fake - 0)**2))/2.

            else:
                raise Exception()

            gen_costs.append(gen_cost)
            disc_costs.append(disc_cost)

    gen_cost = tf.add_n(gen_costs) / len(DEVICES)
    disc_cost = tf.add_n(disc_costs) / len(DEVICES)

    if MODE == 'wgan':
        gen_train_op = tf.train.RMSPropOptimizer(learning_rate=G_LR).minimize(gen_cost,
                                             var_list=lib.params_with_name('Generator'), colocate_gradients_with_ops=True)
        disc_train_op = tf.train.RMSPropOptimizer(learning_rate=D_LR).minimize(disc_cost,
                                             var_list=lib.params_with_name('Discriminator.'), colocate_gradients_with_ops=True)

        clip_ops = []
        for var in lib.params_with_name('Discriminator'):
            clip_bounds = [-.01, .01]
            clip_ops.append(tf.assign(var, tf.clip_by_value(var, clip_bounds[0], clip_bounds[1])))
        clip_disc_weights = tf.group(*clip_ops)

    elif MODE == 'wgan-gp':
        gen_train_op = tf.train.AdamOptimizer(learning_rate=G_LR, beta1=BETA1_G, beta2=0.9).minimize(gen_cost,
                                          var_list=lib.params_with_name('Generator'), colocate_gradients_with_ops=True)
        disc_train_op = tf.train.AdamOptimizer(learning_rate=D_LR, beta1=BETA1_D, beta2=0.9).minimize(disc_cost,
                                           var_list=lib.params_with_name('Discriminator.'), colocate_gradients_with_ops=True)

    elif MODE == 'dcgan':
        gen_train_op = tf.train.AdamOptimizer(learning_rate=2e-4, beta1=0.5).minimize(gen_cost,
                                          var_list=lib.params_with_name('Generator'), colocate_gradients_with_ops=True)
        disc_train_op = tf.train.AdamOptimizer(learning_rate=2e-4, beta1=0.5).minimize(disc_cost,
                                           var_list=lib.params_with_name('Discriminator.'), colocate_gradients_with_ops=True)

    elif MODE == 'lsgan':
        gen_train_op = tf.train.RMSPropOptimizer(learning_rate=1e-4).minimize(gen_cost,
                                             var_list=lib.params_with_name('Generator'), colocate_gradients_with_ops=True)
        disc_train_op = tf.train.RMSPropOptimizer(learning_rate=1e-4).minimize(disc_cost,
                                              var_list=lib.params_with_name('Discriminator.'), colocate_gradients_with_ops=True)

    else:
        raise Exception()

    # For generating samples
    fixed_noise = tf.constant(np.random.normal(size=(BATCH_SIZE, 128)).astype('float32'))
    all_fixed_noise_samples = []
    for device_index, device in enumerate(DEVICES):
        n_samples = BATCH_SIZE // len(DEVICES)
        all_fixed_noise_samples.append(Generator(n_samples,
                                                 noise=fixed_noise[device_index*n_samples:(device_index+1)*n_samples]))
    if tf.__version__.startswith('1.'):
        all_fixed_noise_samples = tf.concat(all_fixed_noise_samples, axis=0)
    else:
        all_fixed_noise_samples = tf.concat(0, all_fixed_noise_samples)

    def generate_image(iteration):
        samples = session.run(all_fixed_noise_samples)
        samples = ((samples+1.)*(255.99//2)).astype('int32')
        lib.save_images.save_images(samples.reshape((BATCH_SIZE, 3, DIM, DIM)), '%s/samples_%d.png' % (SAMPLES_DIR, iteration))

    fid_tfvar = tf.Variable(0.0, trainable=False)
    fid_sum = tf.summary.scalar("FID", fid_tfvar)
    writer = tf.summary.FileWriter(TBOARD_DIR, session.graph)

    # Dataset iterator
    train_gen, dev_gen = lib.data_loader.load(BATCH_SIZE, DATA_DIR, DATASET)

    def inf_train_gen():
        while True:
            for (images,) in train_gen():
                yield images

    # Save a batch of ground-truth samples
    _x = inf_train_gen().__next__()
    _x_r = session.run(real_data, feed_dict={real_data_conv: _x[:BATCH_SIZE//N_GPUS]})
    _x_r = ((_x_r+1.)*(255.99//2)).astype('int32')
    lib.save_images.save_images(_x_r.reshape((BATCH_SIZE//N_GPUS, 3, DIM, DIM)), '%s/samples_groundtruth.png' % SAMPLES_DIR)

    session.run(tf.global_variables_initializer())

    # Checkpoint saver
    ckpt_saver = tf.train.Saver()

    if LOAD_CHECKPOINT:
      if load_checkpoint(session, ckpt_saver, CHECKPOINT_DIR):
        print(" [*] Load SUCCESS")
      else:
        print(" [!] Load failed...")
        
    gen = inf_train_gen()

    # load model
    print("load inception model..", end=" ", flush=True)
    fid.create_inception_graph(os.path.join(INCEPTION_DIR, "classify_image_graph_def.pb"))
    print("ok")
    
    print("load train stats.. ", end="", flush=True)
    # load precalculated training set statistics
    f = np.load(STAT_FILE)
    mu_real, sigma_real = f['mu'][:], f['sigma'][:]
    f.close()
    print("ok")

    # Train loop

    for it in range(ITERS):
        
        iteration = it + ITER_START

        start_time = time.time()

        # Train generator
        if iteration > 0:
            _ = session.run(gen_train_op)

        # Train critic
        if (MODE == 'dcgan') or (MODE == 'lsgan'):
            disc_iters = 1
        else:
            disc_iters = CRITIC_ITERS
        for i in range(disc_iters):
            _data = gen.__next__()
            _disc_cost, _ = session.run([disc_cost, disc_train_op], feed_dict={all_real_data_conv: _data})
            if MODE == 'wgan':
                _ = session.run([clip_disc_weights])

        lib.plot.plot('train disc cost', _disc_cost)
        lib.plot.plot('time', time.time() - start_time)

        if iteration % SAVE_SAMPLES_STEP == 0:
        #    t = time.time()
        #    dev_disc_costs = []
        #    for (images,) in dev_gen():
        #        _dev_disc_cost = session.run(disc_cost, feed_dict={all_real_data_conv: images})
        #        dev_disc_costs.append(_dev_disc_cost)
        #    lib.plot.plot('dev disc cost', np.mean(dev_disc_costs))

            generate_image(iteration)

        if iteration % OUTPUT_STEP == 0:
            lib.plot.flush()

        if (iteration % FID_STEP == 0):

          # FID
          samples = np.zeros((FID_EVAL_SIZE, OUTPUT_DIM))

          n_fid_batches = FID_EVAL_SIZE // FID_SAMPLE_BATCH_SIZE

          for i in range(n_fid_batches):

            print("\rgenerate fid sample batch %d/%d " % (i + 1, n_fid_batches), end="", flush=True)

            frm = i * FID_SAMPLE_BATCH_SIZE
            to = frm + FID_SAMPLE_BATCH_SIZE

            samples[frm:to] = session.run(Generator(FID_SAMPLE_BATCH_SIZE))

          # Cast, reshape and transpose (BCHW -> BHWC)
          samples = ((samples + 1.0) * 127.5).astype('uint8')
          samples = samples.reshape(FID_EVAL_SIZE, 3, DIM, DIM)
          samples = samples.transpose(0,2,3,1)

          print("ok")

          mu_gen, sigma_gen = fid.calculate_activation_statistics(samples,
                                                                  session,
                                                                  batch_size=FID_BATCH_SIZE,
                                                                  verbose=True)

          print("calculate FID:", end=" ", flush=True)
          try:
              FID = fid.calculate_frechet_distance(mu_gen, sigma_gen, mu_real, sigma_real)
          except Exception as e:
              print(e)
              FID=500

          print(FID)

          session.run(tf.assign(fid_tfvar, FID))
          summary_str = session.run(fid_sum)
          writer.add_summary(summary_str, iteration)

        # Save checkpoint
        if (iteration != 0) and (iteration % CHECKPOINT_STEP == 0):
          ckpt_saver.save(session,
                          os.path.join(CHECKPOINT_DIR, "WGAN_GP.model"),
                          iteration)

        lib.plot.tick()
