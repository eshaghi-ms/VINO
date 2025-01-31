#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions for generating data using Gaussian Random Fields

"""
import numpy as np

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from scipy.interpolate import griddata
#import matplotlib.pyplot as plt

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
    
    # Define the spatial grid in 2D
    n = 16
    x = np.linspace(0, 1, n)[:,None]
    y = np.linspace(0, 1, n)[:,None]
    x_grid, y_grid = np.meshgrid(x, y)
    
    # Create a mean-zero Gaussian kernel (RBF kernel)
    kernel = RBF(length_scale)
    
    # Generate random samples from a Gaussian Process
    gp = GaussianProcessRegressor(kernel=kernel)
    f = gp.sample_y(np.column_stack((x_grid.ravel(), y_grid.ravel())), random_state=np.random.randint(int(2**31)))
    #f = f.reshape(x_grid.shape)
    
    sensor_grid_x, sensor__grid_y = np.meshgrid(sensor_pts, sensor_pts)
    f = griddata((x_grid.flatten(), y_grid.flatten()), f.flatten(), (sensor_grid_x, sensor__grid_y), method='cubic')

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
    f_all : (3D arrays)
        the values of the input function at the sensor points, one input function per row and per column

    """
    f_all = np.zeros((N, m, m))
    sensor_pts = np.linspace(0, 1, m+1, endpoint=False)[1:]
    # x_new_grid, y_new_grid = np.meshgrid(sensor_pts, sensor_pts)
    for i in range(N):
        if i % 100 == 0:
            print(f"Generating the {i}th input")
        in_data = GRF(sensor_pts, length_scale=length_scale)*scaling
        if randomize_scale:
            in_data *= np.random.rand()
        f_all[i] = in_data
        
        #fig = plt.figure()
        #ax = fig.add_subplot(111, projection='3d')
        #ax.plot_surface(x_new_grid, y_new_grid, f_all[i], cmap='viridis')
        #ax.set_xlabel('x')
        #ax.set_ylabel('y')
        #ax.set_zlabel('Predicted u(x, y)')
        #ax.set_title('Predicted Random Field using Gaussian Process (2D)')
        #plt.show()

    return f_all
