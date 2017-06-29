import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import collections
import time
import pickle
import math

_since_beginning = collections.defaultdict(lambda: {})
_since_last_flush = collections.defaultdict(lambda: {})

_iter = [0]
def tick():
  _iter[0] += 1

def plot(name, value):
  _since_last_flush[name][_iter[0]] = value

def flush():
  prints = []

  for name, vals in _since_last_flush.items():
    #prints.append("{}\t{}" % (name, np.mean(vals.values())))
    v = vals.values()
    sv = sum(v)
    prints.append("%s\t%f" % (name, sv / len(v)))
    _since_beginning[name].update(vals)

    x_vals = sorted(_since_beginning[name].keys())
    y_vals = [_since_beginning[name][x] for x in x_vals]

    #plt.clf()
    #plt.plot(x_vals, y_vals)
    #plt.xlabel('iteration')
    #plt.ylabel(name)
    #plt.savefig(name.replace(' ', '_')+'.jpg')

  #print("iter %d\t%s" % (_iter[0], "\t".join(prints)))
  print("iter %d" % (_iter[0]))
  for p in prints:
    print(p)
  _since_last_flush.clear()

  #with open('log.pkl', 'wb') as f:
  #  pickle.dump(dict(_since_beginning), f, pickle.HIGHEST_PROTOCOL)
