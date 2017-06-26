import tensorflow as tf
import numpy as np
import random
import os
import scipy as sp
import random


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


    def get_next_random_blurred_batch(self, batch_size, img_h, img_w, std_min=0.0, std_max=1.0):
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
            self.apply_gaussian_blur_rnd(img_h,img_w,std_min,std_max)
            self._cur_samp = batch_size
            ret_D = self._transf_data[0:self._cur_samp, :]
            if self._labels != None:
                ret_L = self._labels[0:self._cur_samp, :]
        return [ret_D, ret_L]
    #---------------------------------------------------------------------------

    def get_next_normalized_batch(self,batch_size):
        batch = self.get_next_batch(batch_size)
        batch[0] = self._batch_studentize(batch[0])
        return batch
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


    def apply_gaussian_blur_rgb2(self,sigma, m, n):
        self._transf_data = np.zeros_like(self._data)
        for i in range(self._d0):
            tmp = gaussian( self._data[i].reshape(m,n,3), sigma)
            self._transf_data[i,:] = tmp.reshape(n*m*3,)
    #---------------------------------------------------------------------------

    def apply_gaussian_blur_rgb(self, sigma, m, n):
        self._transf_data = np.zeros_like(self._data)
        l = m*n
        for i in range(self._d0):
            tmp0 = fi.gaussian_filter( self._data[i].reshape(m,n, 3)[:,:,0], sigma)
            tmp1 = fi.gaussian_filter( self._data[i].reshape(m,n, 3)[:,:,1], sigma)
            tmp2 = fi.gaussian_filter( self._data[i].reshape(m,n, 3)[:,:,2], sigma)
            self._transf_data[i,0:l] = tmp0.flatten()
            self._transf_data[i,l:2*l] = tmp1.flatten()
            self._transf_data[i,2*l:] = tmp2.flatten()
    #--------------------------------------------------------------------------

    def apply_gaussian_blur_rnd(self, m , n, chan=1, std_min=0.0, std_max=1.0):
        self._transf_data = np.zeros_like(self._data)
        d = std_max - std_min
        std = np.random.rand(self._d0,) * d + std_min
        if chan == 1:
            for i in range(self._d0): #!!TODO!!
                tmp = fi.gaussian_filter(self._data[i].reshape(m, n), std[i])
                #self._data[i, :] = tmp.reshape(n * m, )
                self._transf_data[i,:] = tmp.reshape(n*m, )
        elif chan == 3:
            for i in range(self._d0):
                img = self._data[i].reshape(m, n, 3)
                for j in range(3):
                    img[:,:,j] = fi.gaussian_filter(img[:,:,3], std[i])
                    #self._data[i, :] = tmp.reshape(n * m, )
                self._transf_data[i,:] = tmp.reshape(n*m, )
        else:
            raise Exception("Unknown number of chanels: " + str(chan))
    # ---------------------------------------------------------------------------

    def apply_gauss_noise(self, alpha, scale=1.0):
        rnd = np.random.randn(self._d0, self._d1)
        rnd = (rnd - rnd.min()) / (rnd.max() - rnd.min()) * scale
        if alpha > 1e-7:
            self._transf_data = (1-alpha)*self._data + alpha*rnd
        else:
            self._transf_data = self._data.copy()
    #----------------------------------------------------------------------------

    def apply_rect(self, hi, wi, chan, share, positioning="random", val=0.0):
        self._transf_data = np.zeros_like(self._data)
        for i in range(self._d0):
            img = self._data[i,:].reshape(hi,whi,chan)
            self._transf_data[i,:] = drop_rect(img, hi, wi, chan, share=share, positioning=positioning, val=val).flatten()
    #----------------------------------------------------------------------------

    def apply_mult_rect(self, n_rect, hi, wi, chan, share, val=0.0):
        self._transf_data = np.zeros_like(self._data)
        for i in range(self._d0):
            img = self._data[i,:].reshape(hi,wi,chan)
            self._transf_data[i,:] = drop_rect(img, hi, wi, chan, share=share, positioning="random", val=val).flatten()
            for j in range(1,n_rect):
                img = self._transf_data[i,:].reshape(hi,wi,chan)
                self._transf_data[i,:] = drop_rect(img, hi, wi, chan, share=share, positioning="random", val=val).flatten()
    #---------------------------------------------------------------------------


    #lokal_swirl(img_in, hi, wi, chan, n_swirls, radius, strength, positioning="random", directions="random"):
    def apply_local_swirl(self, hi, wi, chan, n_swirls, radius, strength, positioning="random", directions="random"):
        self._transf_data = np.zeros_like(self._data)
        for i in range(self._d0):
            img_in = self._data[i,:].reshape(hi,wi,chan)
            img = lokal_swirl(img_in, hi, wi, chan, n_swirls, radius, strength, positioning=positioning, directions=directions)
            self._transf_data[i,:] = img.flatten()
    #----------------------------------------------------------------------------

    # TODO:
    def salt_and_pepper_rgb_naive(self, h, w,  p=0.5, mi=0.0, ma=1.0):
        self._transf_data = self._data.copy()
        for i in range(self._d0):
            for j in range(h):
                for k in range(w):
                    if (np.random.rand() - p)  < 0:
                        if (np.random.rand() - 0.5)  < 0:
                            self._transf_data[i].reshape(h,w,3)[j,k,:] = mi
                        else:
                            self._transf_data[i].reshape(h,w,3)[j,k,:] = ma
    #---------------------------------------------------------------------------


    # TODO: implement without loops
    def get_mixed_batch(self, batchsize, D, p, eps=1e-7):
        batch = None
        if p > eps:
            d0, d1 = D.shape
            batch = np.zeros((batchsize, self._d1))
            for i in range(batchsize):
                rnd = np.sign(np.random.rand(1,)[0]-p)
                if rnd > 0:
                    idx = random.randint(0,self._d0-1)
                    batch[i] = self._data[idx]
                else:
                    idx = random.randint(0,d0-1)
                    batch[i] = D[idx]
        else:
            batch = self.get_next_batch(batchsize)[0]
        return batch
    #--------------------------------------------------------------------------


    def set_reshuffle_idx(self):
        self._reshuffle_idx = np.array(range(self._d0))
        np.random.shuffle(self._reshuffle_idx)
    #---------------------------------------------------------------------------

    def apply_merging(self, alpha1, alpha2):
        if self._reshuffle_idx is None:
            self.set_reshuffle_idx()
        self._transf_data = alpha1*self._data + alpha2*self._data[self._reshuffle_idx,:]
    #---------------------------------------------------------------------------

    def apply_merging_ext_img(self, img, alpha1, alpha2):
        self._transf_data = alpha1*self._data + alpha2*img
    #---------------------------------------------------------------------------


    def get_sampled_batch(self, batch_size, n_features, fun):
        return fun(batch_size, n_features)
    #---------------------------------------------------------------------------
