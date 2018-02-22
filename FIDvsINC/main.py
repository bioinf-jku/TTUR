import matplotlib
matplotlib.use('Agg')
import os
import tensorflow as tf
import fid
import numpy as np
import math
import utils
import fidutils
from glob import glob
import argparse

#
# parse params
#
parser = argparse.ArgumentParser()
parser.add_argument('--path_IncNet', type=str, help='Path to inception net.')
parser.add_argument('--dataset',     type=str, default='CelebA', help='Possible options: CelebA, Cifar10, Other. (default: CelebA)')
parser.add_argument('--path_data',   type=str, help='Path to images')
parser.add_argument('--path_out',    type=str, help="Path to output directory")
parser.add_argument('--path_stats',  type=str, help='Path to precalculated statistics')
hp_str = '''Possible nois types: sp (salt and pepper),
                     rect (black rectangles),
                     swirl,
                     blur,
                     gn (gaussian noise)
                     mixed (mixture with ImageNet images)
To make multiple experiments, pass noise types seperated by colons (e.g. sp:rect:swirl).
(default: sp)
'''
parser.add_argument('--noise_type',  type=str, default='sp',    help=hp_str)
parser.add_argument('--img_file_ext',type=str, default='*.png', help='Extension of image files. If no specific extenison i ')
parser.add_argument('--n_imgs',      type=int, default=50000,   help='Number of images used to calc the distances. (default: 50000)')
parser.add_argument('--gpu',         type=str, default='',      help='GPU to use (leave blank for CPU only)')
parser.add_argument('--verbose',     type=str, default='',      help='Report status of program in console. \"Y\" for yes. (default: status is not reported)')
parser.add_argument('--sub_paths',   type=str, default='',      help='Create sub directories per distortion type. \"Y\" for yes. (default: sub directories are not created)')
parser.add_argument('--img_dims',    type=int, default=None, nargs=3, metavar=('HIGHT', 'WIDTH', 'CHANNELS'),
                                     help='dimensions of images in the order "H W C" for hight, width and channels. Only needed for dataset "Other" (no default value)')
args = parser.parse_args()
#-------------------------------------------------------------------------------


#
# check parameters
#
PATH_INC = args.path_IncNet
if not PATH_INC.endswith("classify_image_graph_def.pb"):
    PATH_INC = os.join(PATH_INC,"classify_image_graph_def.pb")
if not os.path.exists(PATH_INC):
    raise RuntimeError("Invalid path: %s" % PATH_INC)

PATH_DATA = args.path_data
#print("# DEBUG:::PATH_DATA = " +  str(PATH_DATA))
if not os.path.exists(PATH_DATA):
    raise RuntimeError("Invalid path: %s" % PATH_DATA)
PATH_DATA = os.path.join(PATH_DATA,'*')
data = glob(PATH_DATA)
#data = glob(os.path.join(PATH_DATA, '*.jp')

PATH_OUT = args.path_out
if not os.path.exists(PATH_OUT):
    raise RuntimeError("Invalid path: %s" % PATH_OUT)

_H_, _W_, _C_ = None, None, None
PATH_STATS = args.path_stats
#print("# DEBUG:::args.dataset = " + str(args.dataset))
if args.dataset == "CelebA":
    _H_ = 64; _W_ = 64; _C_ = 3
    if not PATH_STATS.endswith("fid_stats_celeba.npz"):
        PATH_STATS = os.path.join(PATH_STATS,"fid_stats_celeba.npz")
elif args.dataset == "Cifar10":
    _H_ = 32; _W_ = 32; _C_ = 3
    if not PATH_STATS.endswith("fid_stats_cifar10_train.npz"):
        PATH_STATS = os.path.join(PATH_STATS,"fid_stats_cifar10_train.npz")
elif args.dataset == "Other":
    _H_ = args.img_dims[0]; _W_ = args.img_dims[1]; _C_ = args.img_dims[2]
    if not PATH_STATS.endswith(".npz"):
        raise RuntimeError("Invalid path: pleas state the full path, including the file name <file_name>.npz")
if not os.path.exists(PATH_STATS):
    raise RuntimeError("Invalid path: %s" % PATH_STATS)

args.noise_type =  args.noise_type.split(':')
for t in args.noise_type:
    if not t in ["sp", "rect", "swirl", "blur", "gn", "mixed"]:
        raise RuntimeError("Invalid noise type: %s" % args.nois_type)

verbose=False
if args.verbose == 'Y':
    verbose = True

if args.verbose and args.gpu != "":
    print("# Setting CUDA_VISIBLE_DEVICES to: " + str(args.gpu))
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
pth_out = args.path_out #pth_gan = "/publicwork/ramsauer/experiments2/celebA_sanity_collaps_FIDandINC"
n_repeats = 1
#-------------------------------------------------------------------------------


#
# read data
#
if verbose:
    print("# Reading %d images..." % args.n_imgs ,end="", flush=True)
# read stats
f = np.load(PATH_STATS)
mu_real, sigma_real = f['mu'][:], f['sigma'][:]
f.close()
# read imgs
N_FEATURES = _H_*_W_*_C_
N_LOAD_IMGS = args.n_imgs # 50000
X = fidutils.DataContainer(np.zeros((N_LOAD_IMGS, N_FEATURES)), epoch_shuffle=False)

for i in range(N_LOAD_IMGS):
    img = utils.get_image(data[i],
                input_height=_H_,
                input_width=_W_,
                resize_height=_H_,
                resize_width=_W_,
                is_crop=False,
                is_grayscale=False)
    X._data[i,:] = img.flatten()

assert X._data.max() <= 1.
assert X._data.min() >= -1.

if verbose:
    print("done")
    print("# image values in range [%.2f, %.2f]" % (X._data.min(), X._data.max()))
#-------------------------------------------------------------------------------


#
# load inference model
#
fid.create_inception_graph(PATH_INC)
batch_size = 100
softmax = None
#-------------------------------------------------------------------------------

#
# run
#
init = tf.global_variables_initializer()
sess = tf.Session()
with sess.as_default():
    sess.run(init)
    query_tensor = fid._get_inception_layer(sess)

    if softmax is None:
        softmax = fidutils.get_softmax(sess, query_tensor)

    for noise_type in args.noise_type:
        if args.verbose:
            print("# Noise type: " + noise_type)
        alphas = None
        if noise_type in ["gn", "rect", "mixed"] :
            alphas = [0.0, 0.25, 0.5, 0.75]
        elif noise_type in ["blur", "swirl"]:
            alphas = [0.0, 1.0, 2.0, 4.0]
        elif noise_type == "sp":
            alphas = [0.0, 0.1, 0.2, 0.3]

        # prepare result writer
        tmp_PATH_OUT = PATH_OUT
        if args.sub_paths:
            tmp_PATH_OUT = PATH_OUT + "/" + noise_type
            os.mkdir(tmp_PATH_OUT)
        res_writer = fidutils.ResultWriter(tmp_PATH_OUT, out_dir_name=noise_type, out_name=noise_type, zfill=3)
        res_writer.new_enumerated_path(force=True)
        n_repeats=1
        save_interval = len(alphas)
        res_writer.add_iter_tracker('Fid', save_interval, n_repeats)
        res_writer.add_iter_tracker('Inc', save_interval, n_repeats)
        res_desc = []
        n_rect = 5

        for i,a in enumerate(alphas):
            if args.verbose:
                print("#  Alpha = %s" % a)
            res_desc.append({'alpha':a})
            if noise_type == "gn":
                X.apply_gauss_noise(alpha=a, mi=-1, ma=1)
            elif noise_type == "rect":
                X.apply_mult_rect(n_rect, _H_, _W_, _C_, share=a, val=X._data.min())
            elif noise_type == "blur":
                X.apply_gaussian_blur(a, _H_, _W_)
            elif noise_type == "swirl":
                if args.dataset == "CelebA": # bigger radius to make the effect more visible
                    X.apply_local_swirl(_H_, _W_, _C_, n_swirls=1, radius=70, strength=a, positioning="center", directions="random")
                else:
                    X.apply_local_swirl(_H_, _W_, _C_, n_swirls=1, radius=25, strength=a, positioning="center", directions="random")
            elif noise_type == "sp":
                X.salt_and_pepper( _H_, _W_, _C_, p=a, mi=-1, ma=1)


            if args.verbose:
                print("#  -- Range of transformed images: [%.2f, %.2f]" % ( X._transf_data.min(), X._transf_data.max()) )
            X._transf_data = (X._transf_data + 1.) * 127.5
            if args.verbose:
                print("#  -- Range of upscaled images:    [ %.2f, %.2f]" % ( X._transf_data.min(), X._transf_data.max()) )
            res_writer.plot_enumerate_RGB(X._transf_data[0], _H_, _W_, i)

            # calc FID
            if args.verbose:
                print("#  -- Calculating frechet distance...", flush=True)
            mu_gen, sigma_gen = fid.calculate_activation_statistics( X._transf_data.reshape( -1, _H_, _W_, _C_),
                                                                     sess,
                                                                     batch_size=batch_size)
            act = fid.get_activations( X._transf_data.reshape( -1, _H_, _W_, _C_),
                                       sess,
                                       batch_size=batch_size,
                                       verbose=False)
            fid_value = fid.calculate_frechet_distance(mu_gen, sigma_gen, mu_real, sigma_real)
            res_writer.save_to_iter_tracker('Fid', fid_value)
            if args.verbose:
                print("#  -- FID = %.5f" % fid_value)

            # calc Inception score
            inc = None
            if args.verbose:
                print("#  -- Calculating inception score...", flush=True)
            inc,_ = fidutils.get_inception_score( X._transf_data.reshape( -1, _H_, _W_, _C_),
                                                  softmax,
                                                  sess,
                                                  splits=10,
                                                  verbose=False)
            if args.verbose:
                print("#  -- INC = %.5f" % inc)
            res_writer.save_to_iter_tracker('Inc', inc);
            res_writer.inc_idx()
        res_writer.write_result_enumerate_internal(res_desc)
