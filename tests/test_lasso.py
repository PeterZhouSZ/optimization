#!/usr/bin/env python

'''
Description of files goes here.
'''

# System imports
import os
import sys
import time
import cPickle

# Scientific computing
import numpy as np
import scipy.linalg as lin
import scipy.sparse as sps

# Plotting
import matplotlib.pyplot as plt

sys.path.append('../optimization');

# Custom modules import.
import lasso
if __name__ == '__main__':
	# Test lasso.
    m = 10
    n = 100

    x = sps.random(n, 1)
    A = np.random.randn(m, n)
    b = (x.T.dot(A.T)).T
    l = 1
    niters = 20
    debug = True

    xhat = lasso.lasso(A, b, l, niters, debug)
