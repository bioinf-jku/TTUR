# Two time-scale update rule for training GANs

This repository contains code accompanying the paper [GANs Trained by a Two Time-Scale Update Rule
Converge to a Local Nash Equilibrium](https://arxiv.org/abs/1706.08500).

## Fréchet Inception Distance (FID)
The FID is the performance measure used to evaluate the experiments in the paper. There, a detailed description can be found
in the experiment section as well as in the the appendix in section A1.

In short:
The Fréchet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1) and X_2 ~ N(mu_2, C_2) is

                       d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

The FID is calculated by assuming that X_1 and X_2 are the activations of the coding layer pool_3 of the inception model (see below) for generated samples and real world samples respectivly. mu_n is the mean and C_n the covariance of the activations of the coding layer over all real world or generated samples.

IMPORTANT: The number of samples to calculate the Gaussian statistics (mean and covariance) should be greater than the
dimension of the coding layer, here 2048 for the Inception pool 3 layer. Otherwise the covariance is not full rank resulting in complex numbers and nans by calculating the square root.

We recommend using a minimum sample size of 10,000 to calculate the FID otherwise the true FID of the generator is
underestimated.

### Compatibility notice
Previous versions of this repository contained two implementations to calculate the FID, a "unbatched" and a "batched" version.
The "unbatched" version should not be used anymore. If you've downloaded this code previously, please update it immediately to
the new version. The old version included a bug!

## A pytorch implementation of the FID
If you're looking for a pytorch implementation we recommend https://github.com/mseitzer/pytorch-fid

## Provided Code

Requirements: TF 1.1+, Python 3.x

#### fid.py
This file contains the implementation of all necessary functions to calculate the FID. It can be used either
as a python module imported into your own code, or as a standalone
script to calculate the FID between precalculated (training set) statistics and a directory full of images, or between
two directories of images.

To compare directories with pre-calculated statistics (e.g. the ones from http://bioinf.jku.at/research/ttur/), use:

    fid.py /path/to/images /path/to/precalculated_stats.npz

To compare two directories, use

    fid.py /path/to/images /path/to/other_images

See `fid.py --help` for more details.

#### fid_example.py
Example code to show the usage of `fid.py` in your own Python scripts.

#### precalc_stats_example.py
Example code to show how to calculate and save training set statistics.


#### WGAN_GP
Improved WGAN (WGAN-GP) implementation forked from https://github.com/igul222/improved_wgan_training
with added FID evaluation for the image model and switchable TTUR/orig settings. Lanuage model with
JSD Tensorboard logging and switchable TTUR/orig settings.

## Precalculated Statistics for FID calculation

Precalculated statistics for datasets
- [cropped CelebA](http://bioinf.jku.at/research/ttur/ttur_stats/fid_stats_celeba.npz) (64x64, calculated on all samples)
- [LSUN bedroom](http://bioinf.jku.at/research/ttur/ttur_stats/fid_stats_lsun_train.npz) (calculated on all training samples)
- [CIFAR 10](http://bioinf.jku.at/research/ttur/ttur_stats/fid_stats_cifar10_train.npz) (calculated on all training samples)
- [SVHN](http://bioinf.jku.at/research/ttur/ttur_stats/fid_stats_svhn_train.npz) (calculated on all training samples)
- [ImageNet Train](http://bioinf.jku.at/research/ttur/ttur_stats/fid_stats_imagenet_train.npz) (calculated on all training samples)
- [ImageNet Valid](http://bioinf.jku.at/research/ttur/ttur_stats/fid_stats_imagenet_valid.npz) (calculated on all validation samples)


are provided at: http://bioinf.jku.at/research/ttur/

## Additional Links

For FID evaluation download the Inception modelf from http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz

The cropped CelebA dataset can be downloaded here http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html

To download the LSUN bedroom dataset go to: http://www.yf.io/p/lsun

The 64x64 downsampled ImageNet training and validation datasets can be found here http://image-net.org/small/download.php
