import numpy as np
import scipy.misc
import time
import os
from glob import glob

def make_generator(path, batch_size, dataset):
    print("scan files", end=" ", flush=True)
    if dataset == "celeba":
      files = glob(os.path.join(path, "*.jpg"))
      dim = 64
    if dataset == "svhn" or dataset == "cifar10":
      files = glob(os.path.join(path, "*.png"))
      dim = 32
    if dataset == "lsun":
      # It's assumed the lsun images are splitted
      # into subdirectories named 0, 1, .., 304
      files = []
      for i in range(304):
        print("\rscan files %d" % i, end="", flush=True)
        files += glob(os.path.join(path, str(i), "*.jpg"))
      dim = 64
    n_files = len(files)
    print()
    print("%d images found" % n_files)
    def get_epoch():
        images = np.zeros((batch_size, 3, dim, dim), dtype='int32')
        files_idx = list(range(n_files))
        random_state = np.random.RandomState()
        random_state.shuffle(files_idx)
        for n, i in enumerate(files_idx):
            image = scipy.misc.imread(files[i])
            images[n % batch_size] = image.transpose(2,0,1)
            if n > 0 and n % batch_size == 0:
                yield (images,)
    return get_epoch

def load(batch_size, data_dir, dataset):
    return (
        make_generator(data_dir, batch_size, dataset),
        make_generator(data_dir, batch_size, dataset)
    )

if __name__ == '__main__':
    train_gen, valid_gen = load(64)
    t0 = time.time()
    for i, batch in enumerate(train_gen(), start=1):
        print("s\t%d" % (str(time.time() - t0), batch[0][0,0,0,0]))
        if i == 1000:
            break
        t0 = time.time()
