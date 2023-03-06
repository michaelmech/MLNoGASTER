import numpy as np
import inspect
import rapidfuzz
from sklearn.metrics import pairwise_distances
import numba

def euclidean(x,y):
  return np.sqrt((x-y)**2)

def nextFibonacci(n):
    a = n*(1 + np.sqrt(5))/2.0
    return round(a)

def prevFibonacci(n):
  a = n/((1 + np.sqrt(5))/2.0)
  return round(a)

def JWdistance(x,y):
  return rapidfuzz.distance.JaroWinkler.distance(x.flatten(),y.flatten())

def pairwise_emd(X):
  return pairwise_distances(X,metric=numba_emd)

def alt_cdf(y):
    uniques, counts = np.unique(y, return_counts=True)
    value_counts = np.cumsum(counts) / np.sum(counts)
    return value_counts[np.searchsorted(uniques, y)]

@numba.njit(fastmath=True)
def numba_emd(u_values, v_values):

    u_sorter = np.argsort(u_values)
    v_sorter = np.argsort(v_values)
    all_values = np.concatenate((u_values, v_values))
    all_values=np.sort(all_values)
    deltas = np.diff(all_values)
    u_cdf_indices=np.searchsorted(u_values[u_sorter],all_values[:-1], 'right')
    v_cdf_indices=np.searchsorted(v_values[v_sorter],all_values[:-1], 'right')
    u_cdf = u_cdf_indices / u_values.size
    v_cdf = v_cdf_indices / v_values.size
   
    return np.sum(np.multiply(np.abs(u_cdf - v_cdf), deltas))

def p_div(x1, x2):
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(np.abs(x2) > 0, np.divide(x1, x2), 0)
  
def inject_repr(cls):
    def generic_repr(that):
      
      return f'{that.__class__.__name__}'
   
    cls.__repr__ = generic_repr
    return cls
