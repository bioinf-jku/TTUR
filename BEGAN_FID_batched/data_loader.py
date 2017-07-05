import os
from PIL import Image
from glob import glob
import tensorflow as tf

def get_loader(root, batch_size, scale_size, data_format, split=None, is_grayscale=False, seed=None):
    dataset_name = os.path.basename(root)
    if dataset_name in ['CelebA'] and split:
        root = os.path.join(root, 'splits', split)

    if dataset_name == "lsun_cropped":
      #print("scan files... ")
      paths = []
      for i in range(304):
        print("\rscan directories %d" % (i + 1), end="", flush=True)
        paths += glob(os.path.join(root,  str(i), "*.jpg"))
      print()
      print("%d files found" % len(paths))
    else:
      print("scan files... ", end="", flush=True)
      paths = glob(os.path.join(root, "*.jpg"))
      print(" %d files found" % len(paths))

    tf_decode = tf.image.decode_jpeg

    with Image.open(paths[0]) as img:
        w, h = img.size
        shape = [h, w, 3]

    filename_queue = tf.train.string_input_producer(list(paths), shuffle=True, seed=seed)
    reader = tf.WholeFileReader()
    filename, data = reader.read(filename_queue)
    image = tf_decode(data, channels=3)

    if is_grayscale:
        image = tf.image.rgb_to_grayscale(image)
    image.set_shape(shape)

    min_after_dequeue = 5000
    capacity = min_after_dequeue + 3 * batch_size

    queue = tf.train.shuffle_batch(
        [image], batch_size=batch_size,
        num_threads=4, capacity=capacity,
        min_after_dequeue=min_after_dequeue, name='synthetic_inputs')

    if dataset_name in ['CelebA']:
        queue = tf.image.crop_to_bounding_box(queue, 50, 25, 128, 128)
        queue = tf.image.resize_nearest_neighbor(queue, [scale_size, scale_size])
    else:
        pass
        #queue = tf.image.resize_nearest_neighbor(queue, [scale_size, scale_size])

    if data_format == 'NCHW':
        queue = tf.transpose(queue, [0, 3, 1, 2])
    elif data_format == 'NHWC':
        pass
    else:
        raise Exception("[!] Unkown data_format: {}".format(data_format))

    return tf.to_float(queue)
