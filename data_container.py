import tensorflow as tf
import numpy as np
import random
import os
import scipy as sp


class GEN_NN_EXCEPTION(Exception):
    pass
#======================EOF CLASS GEN_NN_EXCEPTION===============================

#
# A simple data container
#
class DataContainer:
    def __init__(self,data, labels=None, epoch_shuffle=True):  # todo: labels
        self._data = data
        self._labels = labels
        self._d0 = 0
        self._d1 = 0
        self._cur_samp = 0
        self._mean = None
        self._std = None
        self._min = None
        self._max = None
        self.__init_and_check()
        self._epoch_shuffle = epoch_shuffle
        self._transf_data = None
        self._reshuffle_idx = None
    #---------------------------------------------------------------------------

    # TODO: check parameters
    def __init_and_check(self):
        [self._d0, self._d1] = self._data.shape
        self._cur_samp = 0
        if not self._labels is None:
            ls = self._labels.shape
            if ls[0] != self._d0:
                raise GEN_NN_EXCEPTION("Data and labels must have the same number of samples!")
    #---------------------------------------------------------------------------


    def studentize_data(self):
        self._mean = self._data.mean(0)
        self._std = self._data.std(0)
        self._std[self._std < 1e-12] = 1e-12
        self._data = (self._data - self._mean)/(self._std)
    #---------------------------------------------------------------------------

    def _batch_studentize(self,data):
        mn = data.mean(0)
        std = data.std(0)
        std[std < 1e-12] = 1e-12
        data = (data-mn)/std
        return data
    #---------------------------------------------------------------------------

    def minmax_scale_data(self, fac=None):
        self._min = self._data.min()
        self._max = self._data.max()
        sc = self._max - self._min
        if sc > 0:
            self._data = (self._data - self._min)/sc
            if not fac is None:
                self._data *= fac
    #---------------------------------------------------------------------------


    def scale(self,displace, scale):
        stmp = scale[abs(scale) < 1e-12]
        scale[abs(scale) < 1e-12] = stmp*1e-12
        self._data = (self._data - displace)/scale
    #---------------------------------------------------------------------------

    def get_next_batch(self,batch_size):
        ret_D = None
        ret_L = None
        tmp_smp = self._cur_samp + batch_size
        if tmp_smp <= self._d0:
            ret_D = self._data[self._cur_samp:tmp_smp,:]
            if not self._labels is None:
                ret_L = self._labels[self._cur_samp:tmp_smp,:]
            if tmp_smp < self._d0:
                self._cur_samp = tmp_smp
            else:
                self._cur_samp = 0
        else:
            if self._epoch_shuffle:
                self.reshuffle()
            self._cur_samp = batch_size
            ret_D = self._data[0:self._cur_samp,:]
            if not self._labels is None:
                ret_L = self._labels[0:self._cur_samp,:]
        return [ret_D, ret_L]
    #---------------------------------------------------------------------------

    def get_next_transformed_batch(self, batch_size):
        ret_D = None
        ret_L = None
        tmp_smp = self._cur_samp + batch_size
        if tmp_smp <= self._d0:
            ret_D = self._transf_data[self._cur_samp:tmp_smp, :]
            if not self._labels is None:
                ret_L = self._labels[self._cur_samp:tmp_smp, :]
            if tmp_smp < self._d0:
                self._cur_samp = tmp_smp
            else:
                self._cur_samp = 0
        else:
            if self._epoch_shuffle:
                self.reshuffle()
            self._cur_samp = batch_size
            ret_D = self._transf_data[0:self._cur_samp, :]
            if not self._labels is None:
                ret_L = self._labels[0:self._cur_samp, :]
        return [ret_D, ret_L]
    #---------------------------------------------------------------------------

    def reset_counter(self):
        self._cur_samp = 0;
    #---------------------------------------------------------------------------

    def get_mean(self):
        return self._mean
    #---------------------------------------------------------------------------

    def get_std(self):
        return self._std
    #---------------------------------------------------------------------------

    def get_data(self):
        return self._data
    #---------------------------------------------------------------------------

    def get_transformed_data(self):
        return  self._transf_data
    #---------------------------------------------------------------------------

    def get_labels(self):
        return self._labels
    #---------------------------------------------------------------------------

    def reshuffle(self):
        idx = np.array(range(self._d0))
        np.random.shuffle(idx)
        self._data = self._data[idx,:]
        if not self._labels is None:
            self._labels = self._labels[idx,:]
    #---------------------------------------------------------------------------

    def apply_mult_rect(self, n_rect, hi, wi, chan, share, val=0.0):
        self._transf_data = np.zeros_like(self._data)
        for i in range(self._d0):
            img = self._data[i,:].reshape(hi,wi,chan)
            self._transf_data[i,:] = drop_rect(img, hi, wi, chan, share=share, positioning="random", val=val).flatten()
            for j in range(1,n_rect):
                img = self._transf_data[i,:].reshape(hi,wi,chan)
                self._transf_data[i,:] = drop_rect(img, hi, wi, chan, share=share, positioning="random", val=val).flatten()
    #---------------------------------------------------------------------------

    def set_reshuffle_idx(self):
        self._reshuffle_idx = np.array(range(self._d0))
        np.random.shuffle(self._reshuffle_idx)
    #---------------------------------------------------------------------------


# helper function
def drop_rect(img_in, hi, wi, chan, share=0.5, positioning="random", val=0.0):
    img = img_in.copy()
    if positioning != "random":
        raise NotImplementedError("TODO!")
    rhi = np.int(hi*share)
    rwi = np.int(wi*share)
    xpos = random.randint(0, hi-rhi)
    ypos = random.randint(0, wi-rwi)
    xdim = xpos + rhi
    ydim = ypos + rwi
    if chan == 1:
        img = img.reshape(hi,wi)
        img[xpos:xdim,ypos:ydim] = np.ones((rhi, rwi))*val
    else:
        img = img.reshape(hi,wi, chan)
        img[xpos:xdim,ypos:ydim,:] = np.ones((rhi, rwi, chan))*val
    return img
#-------------------------------------------------------------------------------
