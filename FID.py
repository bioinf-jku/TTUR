# code partly derived from https://github.com/openai/improved-gan/blob/master/inception_score/model.py
import tensorflow as tf
import numpy as np
import scipy as sp
import os
import gzip, pickle


def create_incpetion_graph(path):
    """Creates a graph from saved GraphDef file."""
    # Creates graph from saved graph_def.pb.
    print( "load inception v3..", end=" ")
    with tf.gfile.FastGFile( path, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString( f.read())
        _ = tf.import_graph_def( graph_def, name='')
    print("ok")
#-------------------------------------------------------------------------------



#def load_stats(path, mu="mu_train.npy", sigma="sigma_train.npy"):
#    """Load precalculated statistics stored as  to use for FID calculation."""
#    mu = np.load(os.path.join(path,mu))
#    sigma = np.load(os.path.join(path,sigma))
#    return mu, sigma
#-------------------------------------------------------------------------------

def load_stats(pickle_file):
    fgz = gzip.open(pickle_file, "rb")
    [sigma, mu] = pickle.load(fgz)
    fgz.close()
    return(sigma, mu)
#-------------------------------------------------------------------------------

################################################################################
##                      UNBATCHED FID CALCULATION                             ##
################################################################################
# In the unbatched version the images are fed individually as jpeg into the jpeg
# layer of the inception net. The conversion to and from jpeg slightly changes
# the RGB values. The experiments where performed with this version.

def get_query_tensor_unbatched(sess):
    return sess.graph.get_tensor_by_name('pool_3:0')
#-------------------------------------------------------------------------------


def FID_unbatched( images, querry_tensor, mu_trn, sigma_trn, sess):
    """Calculates  the Frechet Inception Distance (FID) on generated images
    with respect to precalculated statistics.
    FID is a distance measure between two multidimensional
    Gaussians X_1 ~ N(mu_1, C_1) and X_2 ~ N(mu_2, C_2):
    FID = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Params:
    -- images      : Numpy array of dimension (n_images, hi, wi, 3). The values
                     must lie between 0 and 256.
    -- query_tensor: the tensor returned by the function 'get_query_tensor_unbatched'
    -- mu_trn      : The sample mean over activations of the pool_3 layer, precalcualted
                     on an representive data set.
    -- sigma_trn   : The covariance matrix over activations of the pool_3 layer,
                     precalcualted on an representive data set.
    -- sess        : Current session.

    Returns:
    -- FID  : The Frechet Inception Distance. If an exception occures, FID=500 is returned.
    """
    d0 = images.shape[0]
    pred_arr = np.zeros((n_used_imgs,2048))
    for i, img in enumerate(images):
        if i % 100 == 0:
            print("\rprop incept %d/%d" % (i, len(samples)), end="", flush=True)
        #img = (image + 1.) * 127.5
        image_jpeg = self.sess.run(self.encode_jpeg, feed_dict={self.image_enc_data: img})
        predictions = self.sess.run(query_tensor, {'DecodeJpeg/contents:0': image_jpeg})
        predictions = np.squeeze(predictions)
        pred_arr[i] = predictions

    print("mu sigma incept prop", end=" ", flush=True)
    mu_query = np.mean(pred_arr, axis=0)
    sigma_query = np.cov(pred_arr, rowvar=False)
    print("ok")
    print("FID", end=" ", flush=True)
    try:
        s2srn = sqrtm(sigma_query)
        s2srn = sqrtm(np.dot(np.dot(s2srn, sigma_trn), s2srn))
        fid = np.square(norm(mu_query - mu_trn)) + np.trace(sigma_query + sigma_trn - 2 * s2srn)
    except Exception as e:
        print()
        print("Exception:")
        print(e)
        print("FID", end=" ")
        fid = 500
    return fid
#-------------------------------------------------------------------------------




################################################################################
##                      BATCHED FID CALCULATION                               ##
################################################################################
# In the batched version the images are fed into the ExpandDims layer of the
# inception net. Since the conversion into jpeg is circumvented, the values of
# the images are not changed. But this slightly changes the FID compared to the
# unbatched version.

def get_Fid_query_tensor(sess):
    """Prepares inception net for batched usage and returns pool_3 layer.
    Function is mostly copied from:
    https://github.com/openai/improved-gan/blob/master/inception_score/model.py
    """
    pool3 = sess.graph.get_tensor_by_name('pool_3:0')
    ops = pool3.graph.get_operations()
    for op_idx, op in enumerate(ops):
        for o in op.outputs:
            shape = o.get_shape()
            shape = [s.value for s in shape]
            new_shape = []
            for j, s in enumerate(shape):
                if s == 1 and j == 0:
                    new_shape.append(None)
                else:
                    new_shape.append(s)
            o._shape = tf.TensorShape(new_shape)
    return pool3
#-------------------------------------------------------------------------------


def get_predictions( images, query_tensor, sess, batch_size=50,
                     hi=64, wi=64, chan=3, verbous=False):
    """Calculates the activations of the pool_3 layer for all images.

    Params:
    -- images      : Numpy array of dimension (n_images, hi, wi, 3). The values
                     must lie between 0 and 256.
    -- query_tensor: the tensor returned by the function 'get_Fid_query_tensor'
    -- sess        : current session
    -- batch_size  : the images numpy array is split into batches with batch size
                     batch_size. A reasonable batch size depends on the disposable hardware.
    -- hi, wi, chan: image hight, image with and number of chanels
    -- verbous     : If set to True the number of calculated batches is reported.
    Returns:
    -- pred_arr: A numpy array of dimension (num images, 2048) that contains the
                 activations of the pool_3:0 layer.
    """
    d0 = images.shape[0]
    if batch_size > d0:
        print("warning: batch size is bigger than the data size. setting batch size to data size")
        batch_size = d0
    n_batches = d0//batch_size
    n_used_imgs = n_batches*batch_size
    pred_arr = np.zeros((n_used_imgs,2048))
    for i in range(n_batches):
        if verbous:
                print("\rPropagating batch %d/%d" % (i+1, n_batches), end="", flush=True)
        start = i*batch_size
        end = start + batch_size
        batch = images[start:end]
        pred = sess.run(query_tensor, {'ExpandDims:0': batch})
        pred_arr[start:end] = pred.reshape(batch_size,-1)
    if verbous:
        print(" done")
    return pred_arr
#-------------------------------------------------------------------------------


def FID( pred_arr, mu_trn, sigma_trn, sess):
    """Numpy implementation of the Frechet Inception Distance (FID)
       between two multidimensional Gaussians X_1 ~ N(mu_1, C_1) and X_2 ~ N(mu_2, C_2):
       FID = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Params:
    -- pred_arr : Numpy array containing the activations of the pool_3 layer of the
                  inception net ( like returned by the function 'get_predictions')
    -- mu_trn   : The sample mean over activations of the pool_3 layer, precalcualted
                  on an representive data set.
    -- sigma_trn: The covariance matrix over activations of the pool_3 layer,
                  precalcualted on an representive data set.
    -- sess     : Current session.

    Returns:
    -- fid  : The Frechet Inception Distance. If an exception occures, FID=500 is returned.
    -- mean : The squared norm of the difference of the means: ||mu_1 - mu_2||^2.
    -- trace: The trace-part of the FID: Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    """
    mu_query = np.mean(pred_arr, axis=0)
    sigma_query = np.cov(pred_arr, rowvar=False)
    fid, mean, trace = None, None, None
    try:
        s2srn = sp.linalg.sqrtm(sigma_query)
        s2srn = sp.linalg.sqrtm(np.dot(np.dot(s2srn, sigma_trn), s2srn))
        mean = np.square(np.linalg.norm(mu_query - mu_trn))
        trace = np.trace(sigma_query) + np.trace(sigma_trn) - 2 * np.trace(s2srn)
        fid = mean + trace
    except Exception as e:
        print(e)
        print("exception occured. FID is set to 500")
        fid = 500
    return fid, mean, trace
#-------------------------------------------------------------------------------
