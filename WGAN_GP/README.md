This is a fork of the Improved Wasserstein Implementation

https://github.com/igul222/improved_wgan_training

We ported the implementation to Python 3.x and added a FID
evaluation to the image model (gan_64x64_FID.py) which is
logged and trackable with Tensorboard.

The language model is altered to also log Tensorboard events
i.e. the JSD.
