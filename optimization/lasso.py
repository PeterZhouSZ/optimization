#!/usr/bin/env python

# System imports
import os
import sys
import time
import cPickle
import pdb

# Scientific computing
import numpy as np
import scipy as sp
import scipy.linalg as lin
import scipy.ndimage as ndim

# Plotting
import matplotlib.pyplot as plt

def lasso(A, b, l, niters=100, debug=True):
    '''
        Function to solve the LASSO problem,

        min_x l\|x\|_1 + 1/2 \|Ax - b\|^2
        using ADMM.

        Inputs:
            A: Features matrix.
            b: Coefficients matrix.
            l: Lambda, the weight for L1 penalty.
            niters: Number of iterations to run the ADMM for.
            debug: If True, print debug messages.

        Outputs:
            x: Solution of the LASSO problem.
    '''
    # We solve the augmented langrangian problem:
    # min_(x, z, L) \|x\|_1 + 1/2 \|Ax - b\|^2 + beta/2\|z - x - L\|^2

    # Initialize the variables. We will use the split z = x.
    [m, n] = A.shape

    x = np.zeros((n, 1))        # Primary variable.
    z = np.zeros((n, 1))        # Dummy variable.
    L = np.random.randn(n, 1)   # Dual variable.

    # Constants
    beta = 10       # Weight of augmented lagrangian.
    eta = 1.5       # Ascent rate for dual variable.

    # Precomputation.
    Atb = A.T.dot(b)
    AtA = A.T.dot(A)
    Amat = AtA + beta*np.eye(n)

    # Start the iterations
    for idx in range(niters):
        # Step 1: solve for x -- Thresholding with l/beta.
        x = shrink(z - L, l/beta)

        # Step 2: Solve for z -- Least squares.
        z = lin.solve(Amat, Atb + beta*(x + L))

        # Step 3: Solve for L -- Dual ascent.
        L -= eta*(z - x)

        # Print debug message.
        if debug:
            lasso_debug(A, b, x, z, l, idx)

    return x

def shrink(x, l):
    '''
        Function to perform soft thresholding.
    '''
    return np.sign(x)*np.maximum(abs(x)-l, 0)

def lasso_debug(A, b, x, z, l, idx):
    '''
        Debug printing utility for LASSO

        Inputs:
            A: Features matrix.
            b: RHS.
            x: Solution at this iteration.
            z: Dummy variable.
            l: Lambda, the constant for L1 penalty.
            idx: Current iteration number.

        Outputs:
            None.
    '''
    obj = l*sum(abs(x)) + 0.5*pow(lin.norm(A.dot(x) - b), 2)
    zx = lin.norm(z - x)

    if idx == 0:
        print '%3s\t%10s\t%10s'%('Iter', 'Objective', '||z-x||')
    else:
        print '%3d\t%10.4f\t%10.4f'%(idx+1, obj, zx)

if __name__ == '__main__':
	pass
