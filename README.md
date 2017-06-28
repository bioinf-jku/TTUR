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

### Batched and unbatched implementation
In this repository we provide two implementations to calculate the FID, a "unbatched" and a "batched" version. Here "unbatched" 
and "batched" refer to the way the data is fed into the inception net. The used pretrained model (see below for the link) takes 
individual images in JPEG format as input. The "unbatched" version uses this original input layer whereas the "batched" version
skips this layer. This results in a different FID for the two versions, since the conversion into and from JPEG slightly
changes the RGB values. Note, that while the two versions behave consistently on theire own, they are not directly compareable.

The experiments in the paper are done with the "unbatched" version, except for the reported consistency tests. 
The downside of the "unbatched" version is, that it is very slow (but since we started with this version we had to stick 
with it).  Therefore, if a direct comparison with the results in the paper is not necessary, it might be better to use the
batched version.

## Provided Code

-- Example code for FID in preparation --

#### FID.py
This file contains the implementation of all necessary functions to calculate the FID. Code for precalculating statistics will
be added soon.

#### FID_example_batched.py
Example code to show the usage of the batched version of the FID implementation on the CelebA dataset. 

#### FID_example_unbatched.py
Example code to show the usage of the unbatched version of the FID implementaion on the CelebA dataset.

#### data_container.py
Containes a helper class for data handling.

## Additional info 
- precalculated statistics stat_trn.pkl.gz used in FID_Examples is to big to store on github. It will be made available differently 

- download the inception model from https://github.com/taey16/tf/blob/master/imagenet/classify_image_graph_def.pb and fill in the path to that file in line 67 of FID_Example.py

- download the cropped CelebA dataset from http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
