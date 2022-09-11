import numpy as np
import inspect

def euclidean(x,y):

  return np.sqrt((x-y)**2)

def nextFibonacci(n):
    a = n*(1 + np.sqrt(5))/2.0
    return round(a)

def prevFibonacci(n):
  a = n/((1 + np.sqrt(5))/2.0)
  return round(a)

def p_div(x1, x2):
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(np.abs(x2) > 0, np.divide(x1, x2), 0)
  
def inject_repr(cls):
    def generic_repr(that):
      
      return f'{that.__class__.__name__}'
   
    cls.__repr__ = generic_repr
    return cls