# Two time-scale update rule for training GANs

This repository contains code accompanying the paper [GANs Trained by a Two Time-Scale Update Rule
Converge to a Nash Equilibrium](https://arxiv.org/abs/1706.08500).

## Frechet Inception Distance (FID)
The FID is the performance measure used to evaluate the experiments in the paper. There, a detailed description can be found
in the experiment section as well as in the the appendix in section A1.

In short:
The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1) and X_2 ~ N(mu_2, C_2) is

                       d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

The FID is calculated by assuming that X_1 and X_2 are the activations of the pool_3 layer of the inception model (see below)
for generated samples and real world samples respectivly.

### Compatibility notice
Previous versions of this repository contained two implementations to calculate the FID, a "unbatched" and a "batched" version.
The "unbatched" version should not be used anymore. If you've downloaded this code previously, update it immediately to the new
version. The old version included a bug that made the results incomparable!

## Provided Code

Requirements: TF 1.1, Python 3.x

#### fid.py
This file contains the implementation of all necessary functions to calculate the FID. It can be used either
as a python module imported into your own code, or as a standalone
script to calculate the FID between precalculated (training set) statistics and a directory full of images, or between
two directories of images.

To compare directories with pre-calculated statistics (e.g. the ones from http://bioinf.jku.at/research/ttur/ttur.html), use:

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

## Additional info
Precalculated unbatched statistics for datasets
- cropped CelebA (calculated on 100000 randomly choosen training samples)
- LSUN bedroom (calculated on 100000 randomly choosen training samples)
- Imagenet (calculated on the validation set)

are provided at: http://bioinf.jku.at/research/ttur/ttur.html

For FID evaluation download the Inception modelf from http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz

The cropped CelebA dataset can be downloaded here http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html

To download the LSUN bedroom dataset go to: http://www.yf.io/p/lsun

The 64x64 downsampled ImageNet training and validation datasets can be found here http://image-net.org/small/download.php
