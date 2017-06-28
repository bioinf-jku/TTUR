#import os
#os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import tensorflow as tf
import numpy as np
import scipy.misc
import FID
import data_container as dc
from glob import glob
import os


#
# Functions taken from: https://github.com/carpedm20/DCGAN-tensorflow/blob/master/utils.py
#
def get_image(image_path, input_height, input_width,
              resize_height=64, resize_width=64,
              is_crop=True, is_grayscale=False):
  image = imread(image_path, is_grayscale)
  return transform(image, input_height, input_width,
                   resize_height, resize_width, is_crop)

def imread(path, is_grayscale = False):
  if (is_grayscale):
    return scipy.misc.imread(path, flatten = True).astype(np.float)
  else:
    return scipy.misc.imread(path).astype(np.float)

def transform(image, input_height, input_width,
              resize_height=64, resize_width=64, is_crop=True):
  if is_crop:
    cropped_image = center_crop(
      image, input_height, input_width,
      resize_height, resize_width)
  else:
    if (input_height != resize_height) or (input_width != resize_width):
      cropped_image = scipy.misc.imresize(image, [resize_height, resize_width])
    else:
      cropped_image = image
  return np.array(cropped_image)
#-------------------------------------------------------------------------------





# read N_IMGS data samples and store them in an data container
print("Reading data...", end="", flush=True)
celeb_path = "/publicdata/image/celebA_cropped/"# add path to celabA dataset
data = glob( os.path.join(celeb_path,"*"))
N_IMGS = 50; N_FEATURES = 64*64*3
X = dc.DataContainer(np.zeros((N_IMGS, N_FEATURES)), epoch_shuffle=True)
for i in range(N_IMGS):
    img = get_image( data[i],
                    input_height=64,
                    input_width=64,
                    resize_height=64,
                    resize_width=64,
                    is_crop=False,
                    is_grayscale=False)
    X._data[i,:] = img.flatten()
print("done")



# load inference model
# download model at: https://github.com/taey16/tf/blob/master/imagenet/classify_image_graph_def.pb
inc_pth = "/system/user/ramsauer/GANs/imgnet/tf/imagenet/classify_image_graph_def.pb"# add path to classify_image_graph_def.pb
FID.create_incpetion_graph(inc_pth)

# get tuple for jpeg encoding
jpeg_tuple = FID.get_jpeg_encoder_tuple()

# batch size for batched version
batch_size = 500
init = tf.global_variables_initializer()
sess = tf.Session()
with sess.as_default():
    sess.run(init)
    query_tensor = FID.get_Fid_query_tensor(sess)

    #
    # caĺculate statistics for batch version
    #
    sigma_b, mu_b = FID.precalc_stats_batched( X.get_data().reshape(-1,64,64,3),
                                               query_tensor,
                                               sess,
                                               batch_size=batch_size,
                                               verbous=True)
    # save statistics of batch version
    FID.save_stats(sigma_b, mu_b, "stats_b.pkl.gz")
    # load saved statistics
    (sigma_b_loaded, mu_b_loaded) = FID.load_stats("stats_b.pkl.gz")


    #
    # calculate statistic for unbatched version
    #
    sigma_u, mu_u = FID.precalc_stats_unbatched( X.get_data().reshape(-1,64,64,3),
                                                 query_tensor,
                                                 jpeg_tuple,
                                                 sess)
    # save statistics of unbatched version
    FID.save_stats(sigma_u, mu_u, "stats_u.pkl.gz")
    # load statistics of unbatched version
    (sigma_u_loaded, mu_u_loaded) = FID.load_stats("stats_u.pkl.gz")
