import os
import sys
import math
import random
import tensorflow as tf
import numpy as np
import scipy.stats as st
from scipy.misc import toimage
import scipy as sp
from skimage import data
from skimage.transform import swirl
from skimage.filters import gaussian
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.offsetbox import (TextArea, DrawingArea, OffsetImage,AnnotationBbox)
from matplotlib.cbook import get_sample_data
import matplotlib.gridspec as gridspec
import shutil


#
# derived from https://github.com/openai/improved-gan/blob/master/inception_score/model.py
#

# calculate inception score
def get_inception_score(images, softmax, sess, splits=10, verbose=False):
    inps = images
    bs = 50
    preds = []
    n_batches = int(math.ceil(float(inps.shape[0]) / float(bs)))
    for i in range(n_batches):
        #sys.stdout.write(".")
        #sys.stdout.flush()
        if verbose:
            print("\rPropagating batch %d/%d" % (i+1, n_batches), end="", flush=True)
        inp = inps[(i * bs):min((i + 1) * bs, inps.shape[0])]
        pred = sess.run(softmax, {'FID_Inception_Net/ExpandDims:0': inp})
        preds.append(pred)
    preds = np.concatenate(preds, 0)
    scores = []
    for i in range(splits):
        part = preds[(i * preds.shape[0] // splits):((i + 1) * preds.shape[0] // splits), :]
        kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
        kl = np.mean(np.sum(kl, 1))
        scores.append(np.exp(kl))
    if verbose:
        print(" done")
    return np.mean(scores), np.std(scores)
#-------------------------------------------------------------------------------

# get softmax output
def get_softmax(sess, pool3):
    w = sess.graph.get_operation_by_name("FID_Inception_Net/softmax/logits/MatMul").inputs[1]
    logits = tf.matmul(tf.squeeze(pool3), w)
    softmax = tf.nn.softmax(logits)
    return softmax
#===============================================================================


#
# simple data container with image transformations
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
                raise RuntimeError("Data and labels must have the same number of samples!")
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


    def reset_counter(self):
        self._cur_samp = 0;
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

    def apply_gaussian_blur(self,sigma, m, n):
        self._transf_data = np.zeros_like(self._data)
        for i in range(self._d0):
            tmp = gaussian( self._data[i].reshape(m,n,3), sigma)
            self._transf_data[i,:] = tmp.reshape(n*m*3,)
    #---------------------------------------------------------------------------

    def apply_gauss_noise(self, alpha, mi=-1, ma=1):
        rnd = np.random.randn(self._d0, self._d1)
        rnd = (rnd - rnd.min()) / (rnd.max() - rnd.min())
        rnd = rnd*(ma - mi) + mi
        if alpha > 1e-6:
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


    def apply_local_swirl(self, hi, wi, chan, n_swirls, radius, strength, positioning="random", directions="random"):
        self._transf_data = np.zeros_like(self._data)
        for i in range(self._d0):
            img_in = self._data[i,:].reshape(hi,wi,chan)
            img = lokal_swirl(img_in, hi, wi, chan, n_swirls, radius, strength, positioning=positioning, directions=directions)
            self._transf_data[i,:] = img.flatten()
    #----------------------------------------------------------------------------


    def salt_and_pepper(self, h=64, w=64, c=3, p=0.5, mi=-1, ma=1.0):
        self._transf_data = self._data.copy()
        ns, d0, d1, d2 = self._transf_data.reshape(-1,h,w,c).shape
        coords = np.random.rand(ns,d0,d1) < p
        n_co = coords.sum()
        if n_co > 0:
            vals = (np.random.rand(n_co) < 0.5).astype(np.float32)
            vals[vals < 0.5] = mi; vals[vals > 0.5] = ma
            for i in range(c):
                self._transf_data.reshape(-1,h,w,c)[coords,i] = vals
#------------------------------------------------------------------------------

# helper functions for data container
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

def lokal_swirl(img_in, hi, wi, chan, n_swirls, radius, strength, positioning="random", directions="random", corr_size=3):
    img = img_in.copy()
    if not positioning in  ["random", "center"]:
        raise NotImplementedError("TODO!")
    size = corr_size
    for i in range(n_swirls):
        sign = None
        if directions == "random":
            sign = np.sign(np.random.rand(1) - 0.5)[0]
        elif directions == "left":
            sign = -1
        else:
            sign = 1
        xpos, ypos = None, None
        if positioning == "random":
            xpos = random.randint(0, hi - radius)
            ypos = random.randint(0, wi - radius)
        elif positioning == "center":
            xpos = hi // 2
            ypos = wi // 2
        center = (xpos,ypos)
        img = swirl(img, rotation=0, strength=sign*strength, radius=radius, center=center)
        img[0:size] = img_in[0:size]
        img[-(size+1):] = img_in[-(size+1):]
        img[:,0:size] = img_in[:,0:size]
        img[:,-(size+1):] = img_in[:,-(size+1):]
    return img
#============================EOF DataContainer==================================


#
# simple class to track results
#
class ResultWriter:
    def __init__(self, pth, out_dir_name, out_name="res", zfill=0, out_imgs=True):
        self._pth = pth
        self._out_imgs = out_imgs
        self._out_dir_name = out_dir_name
        self._out_name = out_name
        self._zfill = zfill
        self._enumerator = 0
        self._check_input()
        self._current_path = None
        self._res_dat = {}
        self._res_writer_idx = 0
        self._res_writer_rep = 0
        self._fig = None
    #---------------------------------------------------------------------------

    def _check_input(self):
        self._check(self._pth)
    #---------------------------------------------------------------------------

    def _check(self, pth):
        if not os.path.exists(pth):
            raise RESULT_WRITER_EXCEPTION("No such path found: " + pth)
    #---------------------------------------------------------------------------

    def reset_pth(self, pth):
        self._pth = pth
        self._check_pth()
    #---------------------------------------------------------------------------


    def write_result(self, dir_name, res_dic, res_mat, force=False):
        newpth = os.path.join(self._pth, dir_name)
        if not force:
            if not os.path.exists(newpth):
                os.mkdir(newpth)
            else:
                raise RESULT_WRITER_EXCEPTION("Path already exists: " +  newpth)
        else:
            if os.path.exists(newpth):
                shutil.rmtree(newpth)
            os.mkdir(newpth)

        # write results
        np.save(newpth + self._out_name+'_data.npy', res_mat)
        np.save(newpth + self._out_name+'_descriptor.npy', res_dic)
    #---------------------------------------------------------------------------


    def write_result_enumerate(self, res_dic, res_mat, force=False):
        newpth = os.path.join(self._pth,self._out_dir_name + "_" + str(self._enumerator).zfill(self._zfill)+'/')
        self._current_path = newpth
        if not force:
            if not os.path.exists(newpth):
                os.mkdir(newpth)
            else:
                raise RESULT_WRITER_EXCEPTION("Path already exists: " +  newpth)
        else:
            if os.path.exists(newpth):
                shutil.rmtree(newpth)
            os.mkdir(newpth)

        # write results
        np.save(newpth + self._out_name+'_data.npy', res_mat)
        np.save(newpth + self._out_name+'_descriptor.npy', res_dic)
        self._enumerator += 1
    #---------------------------------------------------------------------------


    def read_result(seld, dir_name):
        newpth = os.path.join(self._pth,dir_name)
        self._check(newpth)
        ret = []
        ret.append(np.load(newpth + self._out_name+'_descriptor.npy'))
        ret.append(np.load(newpth + self._out_name+'_data.npy'))
        return ret
    #---------------------------------------------------------------------------

    def get_current_path(self):
        print(self._current_path)
        return self._current_path + self._out_name
    #--------------------------------------------------------------------------

    def add_iter_tracker(self, name, n_saves, n_repeats):
        self._res_dat[name] = np.zeros((n_repeats, n_saves))
    #--------------------------------------------------------------------------

    def get_iter_tracker_names(self):
        return self._res_dat.keys()
    #--------------------------------------------------------------------------

    def add_append_tracker(self, name):
        self._res_dat[name] = []
    #--------------------------------------------------------------------------

    def add_rep_append_tracker(self,name, n_repats):
        self._res_dat[name] = [[]]*n_repats
    #--------------------------------------------------------------------------

    def reset_saved_vars(self):
        self._res_dat = {}
    #--------------------------------------------------------------------------

    def inc_idx(self):
        self._res_writer_idx += 1
    #--------------------------------------------------------------------------

    def inc_rep(self):
        self._res_writer_rep += 1
    #--------------------------------------------------------------------------

    def reset_idx(self):
        self._res_writer_idx = 0
    #--------------------------------------------------------------------------

    def reset_rep(self):
        self._res_writer_rep = 0
    #-------------------------------------------------------------------------

    def save_to_iter_tracker(self, name, val, warn=False):
        dim = self._res_dat[name].shape
        if self._res_writer_rep < dim[0]:
            if self._res_writer_idx < dim[1]:
                self._res_dat[name][self._res_writer_rep, self._res_writer_idx] = val
        elif warn:
            print("# Warning! Number of repeats or number of saved iterations exceeds the initial set values.")
        else:
            RESULT_WRITER_EXCEPTION("Number of repeats or number of saved iterations exceeds the initial set values.")
    #--------------------------------------------------------------------------

    def save_to_rep_append(self, name, rep, val, warn=False):
        n_reps = len(self._res_dat[name])
        if n_reps > rep:
            self._res_dat[name][rep].append(val)
        elif warn:
            print("# Warning! Number of repeats exceeds the initial set values.")
        else:
            RESULT_WRITER_EXCEPTION("Number of repeats exceeds the initial set values.")
    #---------------------------------------------------------------------------


    def save_to_img_iter_tracker(self, name, val, warn=False):
        pass
    #--------------------------------------------------------------------------


    def save_to_append_tracker(self, name, val):
        self._res_dat[name].append(val)
    #---------------------------------------------------------------------------

    def reset(self):
        self.reset_idx()
        self.reset_rep()
        self._res_dat = {}
    #---------------------------------------------------------------------------

    def plot(self,samples, h,w, it, sqrt_n_imgs=3):
        if self._fig == None:
            self._fig = plt.figure(figsize=(sqrt_n_imgs, sqrt_n_imgs))
        gs = gridspec.GridSpec(sqrt_n_imgs, sqrt_n_imgs)
        gs.update(wspace=0.05, hspace=0.05)
        n_feat = h*w
        for i, sample in enumerate(samples):
            ax = plt.subplot(gs[i])
            plt.axis('off')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')
            plt.imshow(sample[0:n_feat].reshape(h,w), cmap='Greys_r')
        plt.savefig(self._pth + 'imgs/{}.png'.format(str(it).zfill(6)), bbox_inches='tight')
    #---------------------------------------------------------------------------


    def new_enumerated_path(self, force=False):
        self._enumerator += 1
        #newpth = self._pth + self._out_name + "_" + str(self._enumerator).zfill(self._zfill) + '/'
        newpth = os.path.join(self._pth,self._out_dir_name + "_" + str(self._enumerator).zfill(self._zfill)+'/')
        self._current_path = newpth
        if not force:
            if not os.path.exists(newpth):
                os.mkdir(newpth)
            else:
                raise RESULT_WRITER_EXCEPTION("Path already exists: " + newpth)
        else:
            if os.path.exists(newpth):
                shutil.rmtree(newpth)
            os.mkdir(newpth)
        #if self._out_name:
        os.mkdir(newpth + "imgs")
    #---------------------------------------------------------------------------

    def plot_enumerate(self,samples, h,w, it, sqrt_n_imgs=3):
        fig = plt.figure(figsize=(sqrt_n_imgs, sqrt_n_imgs))
        gs = gridspec.GridSpec(sqrt_n_imgs, sqrt_n_imgs)
        gs.update(wspace=0.05, hspace=0.05)
        n_feat = h * w
        for i, sample in enumerate(samples):
            ax = plt.subplot(gs[i])
            plt.axis('off')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')
            plt.imshow(sample[0:n_feat].reshape(h, w), cmap='Greys_r')
        plt.savefig(self._current_path + 'imgs/{}.png'.format(str(it).zfill(6)), bbox_inches='tight')
        # ----------------------------------------------------------------------


    def plot_enumerate_RGB(self, sample, h, w, it):
        img = sample.reshape(h,w,3)
        fig = plt.figure(figsize=(1, 1), frameon=False)
        gs = gridspec.GridSpec(1, 1)
        gs.update(wspace=0.05, hspace=0.05)
        ax = plt.subplot(gs[0])
        plt.axis('off')
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        fig.add_axes(ax)
        plt.imshow(toimage(sample.reshape(h, w, 3)))
        plt.savefig(self._current_path + 'imgs/{}.png'.format(str(it).zfill(6))) #, bbox_inches='tight')
    #---------------------------------------------------------------------------


    def plot_batch_enumerate_RGB(self, samples, h, w, it):
        img = sample.reshape(h,w,3)
        fig = plt.figure(figsize=(1, 1), frameon=False)
        gs = gridspec.GridSpec(1, 1)
        gs.update(wspace=0.05, hspace=0.05)
        ax = plt.subplot(gs[0])
        plt.axis('off')
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        fig.add_axes(ax)
        plt.imshow(toimage(sample.reshape(h, w, 3)))
        plt.savefig(self._current_path + 'imgs/{}.png'.format(str(it).zfill(6))) #, bbox_inches='tight')
    #---------------------------------------------------------------------------


    def write_result_enumerate_internal(self, res_dic):
        # write results
        np.save(self._current_path + self._out_name + '_data.npy', self._res_dat)
        np.save(self._current_path + self._out_name + '_descriptor.npy', res_dic)
        #self._enumerator += 1
    # ---------------------------------------------------------------------------

class RESULT_WRITER_EXCEPTION(Exception):
    pass
#===============EOF RESULT_WRITER====================================#

#
# visualizing experiments
#
class Vis():
    def __init__(self):
        self._curr_path = None
        self._curr_desc = None
        self._curr_data = None
        self._curr_folder = None
        self._val_keys  = None
        self._sum_data  = None
        self._sum_filtered = None
        self._plot = plt.plot()
        self._sum_grads = {}
    #--------------------------------------------------------------------------

    def set_path(self, pth):
        if pth[-1] != "/":
            pth += "/"
        if not os.path.exists(pth):
            raise VIS_EXCEPTION("path " + pth + " doesn't exist.")
        self._curr_path = pth
    #--------------------------------------------------------------------------

    def read_data(self):
        if self._curr_path == None:
            raise VIS_EXCEPTION("Path is not set. Use method set_path(path).")
        d = os.listdir(self._curr_path)
        for i in d:
            if "_data.npy" in i:
                self._curr_data = np.load(self._curr_path + i)
            elif "_descriptor.npy" in i:
                self._curr_desc = np.load(self._curr_path + i)
        self._curr_folder = self._curr_path.split('/')[-2]
        self._curr_data = self._curr_data.take(0)
        self._val_keys = self._curr_desc[0].keys()
    #---------------------------------------------------------------------------

    def get_data_keys(self):
        ret = None
        if self._curr_data != None:
            ret = self._curr_data.keys()
        return ret
    #---------------------------------------------------------------------------

    def get_descriptor(self):
        return self._curr_desc
    #---------------------------------------------------------------------------

    def get_description(self):
        return self._curr_desc[0]
    #---------------------------------------------------------------------------

    def get_data_description(self):
        return self._curr_data.keys()
    #---------------------------------------------------------------------------

    def get_data(self,key):
        return self._curr_data[key]
    #---------------------------------------------------------------------------


    def annotated_plot(self, key, zoom=0.2, rep=0, pad=0, xybox=(0,50),max_hight=None,
                        fd=None, figposx=None, figposy=None, xlim=None, ylim=None,ylabel=None, n_datapoints=None, add_points=False):
        pth = self._curr_path + "imgs/"
        if not os.path.exists(pth):
            raise VIS_EXCEPTION("No image directory found in path " + pth)
        imgs = os.listdir(pth)
        imgs.sort()
        data = self._curr_data[key][rep]
        l = len(data)
        if len(imgs) != l:
            raise VIS_EXCEPTION("Number of images must match the number of data points")
        x = None
        if n_datapoints is None:
            x = range(len(data))
        else:
            if n_datapoints < len(data):
                data = data[:n_datapoints]
                x = range(n_datapoints)
        if not fd is None:
            data = fd(data)
        #fig, ax = plt.subplots()
        _, ax = plt.subplots()
        ax.plot(x, data, linewidth=2.0)
        if add_points:
            ax.plot(x,data, "k*", mew=0.5, ms=10)
        if figposy is None:
            figposy = data
        if figposx is None:
            figposx = x
        i = 0
        for pos in zip(figposx, figposy):
            imgpth = pth + imgs[i];
            i += 1
            fn = get_sample_data(imgpth, asfileobj=False)
            arr_img = plt.imread(fn, format='png')
            im = OffsetImage(arr_img, zoom=zoom)
            im.image.axes = ax
            ab = AnnotationBbox(im, pos,
                                xybox=xybox,
                                xycoords='axes fraction',
                                boxcoords="offset points",
                                #pad=pad,
                                arrowprops=dict(arrowstyle="<-"))
            ax.add_artist(ab)

        mi = np.min(data)
        ma = np.max(data)
        r = ma-mi
        if not xlim is None:
                ax.set_xlim(xlim[0], xlim[1])
        if not ylim is None:
                ax.set_ylim(ylim[0], ylim[1])
        if ylabel is None:
            plt.ylabel(key, fontsize=15)
        else:
            plt.ylabel(ylabel, fontsize=15)
        labs = range(len(data))
        labs = list(map(lambda x: str(x),labs))
        ax.set_xticklabels(labs)
        plt.xlabel("disturbance level", fontsize=15)
        plt.xticks(x)
        plt.show()
    #---------------------------------------------------------------------------

class VIS_EXCEPTION(Exception):
    pass
#==============================EOF VIS==========================================


#
# class to read data written by result writer class
#
class Experiment_reader:
    def __init__(self):
        self._visualizer = []
        self._curr_path = None
    #---------------------------------------------------------------------------

    def set_path(self, pth):
        if pth[-1] != "/":
            pth += "/"
        if not os.path.exists(pth):
            raise VIS_EXCEPTION("path " + pth + " doesn't exist.")
        self._curr_path = pth
    #---------------------------------------------------------------------------

    def read_all_expriments(self, verbouse=False):
        if self._curr_path == None:
            raise VIS_EXCEPTION("No path is set. Set path with method set_path.")
        dir  = os.listdir(self._curr_path)
        self._visualizer = []
        for d in dir:
            p = self._curr_path + d
            if verbouse:
                print("# reading: " + p)
            v = Vis()
            v.set_path(p)
            v.read_data()
            self._visualizer.append(v)
    #---------------------------------------------------------------------------

    def print_param_description(self,idx=None):
        if self._visualizer == []:
            raise VIS_EXCEPTION("No data present. Use method read_all_experiments(path) to load experiments.")
        if idx == None:
            for v in self._visualizer:
                print(v.get_curr_folder_name() + ": " + str(v.get_description()))
        else:
            print(self._visualizer[idx].get_curr_folder_name() + ": " + str(self._visualizer[idx].get_description()))
    # --------------------------------------------------------------------------

    def get_data_description(self):
        keys = {}
        for v in self._visualizer:
            keys.append(v.get_data_description())
            keys[v.get_curr_folder_name()] = v.get_data_description()
        return keys
    #---------------------------------------------------------------------------

    def print_data_description(self, idx=None):
        if idx == None:
            for v in self._visualizer:
                print( v.get_curr_folder_name() + ": " + str(v.get_data_description()))
        else:
            print(self._visualizer[idx].get_curr_folder_name() + ": " + str(self._visualizer[idx].get_data_description()))
    #---------------------------------------------------------------------------


    def get_data(self, key, v_idx=None):
        ret = {}
        if v_idx == None:
            v_idx = range(len(self._visualizer))
        for idx in v_idx:
            ret[self._visualizer[idx].get_curr_folder_name()] = self._visualizer[idx].get_data(key)
        return ret
    #---------------------------------------------------------------------------


    def annotated_plot(self, vis_idx, key, zoom=0.2, rep=0, pad=0, xybox=(0.,50.),
                        max_hight=None, fd=None, figposx=None, figposy=None, xlim=None, ylim=None,
                        ylabel=None, n_datapoints=None, add_points=False):
        self._visualizer[vis_idx].annotated_plot(key,zoom,rep=rep,pad=pad,xybox=xybox,
                                                  max_hight=max_hight, fd=fd, figposx=figposx, figposy=figposy,
                                                  xlim=xlim, ylim=ylim, ylabel=ylabel, n_datapoints=n_datapoints, add_points=add_points)
    #---------------------------------------------------------------------------

#===============================EOF Experiment_reader===========================
