#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions for generating data using Gaussian Random Fields
(adapted from the code in https://github.com/PredictiveIntelligenceLab/Physics-informed-DeepONets )
"""
import numpy as np
np.random.seed(42)
# Define RBF kernel
def RBF(x1, x2, length_scales):
    diffs = np.expand_dims(x1 / length_scales, 1) - \
            np.expand_dims(x2 / length_scales, 0)
    r2 = np.sum(diffs**2, axis=2)
    return np.exp(-0.5 * r2)

def GRF(sensor_pts, length_scale = 0.1):
    """
    The antiderivative operator for Gaussian Random Fields
    Parameters
    ----------
    sensor_pts : (ndarray)
        location of the sensor points (for the input function)

    length_scale : (float), optional
        length scale for the Gaussian Random Field (GRF). The default is 0.1.

    Returns
    -------
    f : (ndarray)
        values of the GRF at the sensor points
    """
    
    # Sample GP prior at a fine grid    
    n = 512
    jitter = 1e-10
    x = np.linspace(0, 1, n)[:,None]
    k = RBF(x, x, length_scale)
    l = np.linalg.cholesky(k + jitter*np.eye(n))
    gp_sample = np.dot(l, np.random.normal(size=n))
    f = np.interp(sensor_pts, x.flatten(), gp_sample)
    

    return f
    
def generate_inputs(N, m=1024, length_scale = 0.1, scaling = 1., randomize_scale=False):
    """
    Generates all the N inputs

    Parameters
    ----------
    N : (int)
        Number of training functions u.

    m : (integer), optional
        number of sensor points. The default is 1024.

    length_scale : (float), optional
        length scale for the Gaussian Random Field. The default is 0.1

    scaling : (float), optional
        scales the inputs by a constant factor. The default is 1

    randomize_scale : (bool), optional
        multiply the scale of each input by a random number between 0 and 1.

    Returns
    -------
    f_all : (2D arrays)
        the values of the input function at the sensor points, one input function per row

    """
    f_all = np.zeros((N, m))
    sensor_pts = np.linspace(0, 1, m+1, endpoint=False)[1:]
    for i in range(N):
        in_data = GRF(sensor_pts, length_scale=length_scale) * scaling
        if randomize_scale:
            in_data *= np.random.rand()
        f_all[i] = in_data
        
    return f_all

def generate_midpoint_inputs(N, m=1024, length_scale = 0.1, scaling = 1., randomize_scale=False):
    """
    Generates all the N inputs at midpoints of equally spaced points

    Parameters
    ----------
    N : (int)
        Number of training functions u.

    m : (integer), optional
        number of sensor points. The default is 1024.

    length_scale : (float), optional
        length scale for the Gaussian Random Field. The default is 0.1

    scaling : (float), optional
        scales the inputs by a constant factor. The default is 1

    randomize_scale : (bool), optional
        multiply the scale of each input by a random number between 0 and 1.

    Returns
    -------
    f_all : (2D arrays)
        the values of the input function at the sensor points, one input function per row

    """
    f_all = np.zeros((N, m))
    sensor_pts = np.linspace(0, 1, m+1)
    sensor_pts = 1/2*(sensor_pts[:-1]+sensor_pts[1:])
    for i in range(N):
        in_data = GRF(sensor_pts, length_scale=length_scale)*scaling
        if randomize_scale:
            in_data *= np.random.rand()
        f_all[i] = in_data
        
    return f_all

