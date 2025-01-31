#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 15:27:50 2023

@author: cosmin + ChatGPT
Prompt: https://chat.openai.com/c/421ff524-3240-4fcd-ad47-fee95c387353
"""
import numpy as np
from scipy.fftpack import idct


# import matplotlib.pyplot as plt

def GRF(alpha, tau, s):
    # Random variables in KL expansion
    xi = np.random.randn(s, s)

    # Define the (square root of) eigenvalues of the covariance operator
    k1, k2 = np.meshgrid(np.arange(s), np.arange(s))
    coef = tau ** (alpha - 1) * (np.pi ** 2 * (k1 ** 2 + k2 ** 2) + tau ** 2) ** (-alpha / 2)

    # Construct the KL coefficients
    l = s * coef * xi
    l[0, 0] = 0

    # 2D inverse discrete cosine transform
    u = idct(idct(l, axis=0, norm='ortho'), axis=1, norm='ortho')
    return u


def generate_inputs(N, m, alpha, tau):
    """
    Generates all the N inputs

    Parameters
    ----------
    N : (int)
        Number of training functions u.

    m : (integer)
        number of sensor points.

    tau, alpha : (float)
        parameters of covariance

    Returns
    -------
    f_all : (3D arrays)
        the values of the input function at the sensor points, one input function per row and per column

    """
    f_all = np.zeros((N, m, m))
    # sensor_pts = np.linspace(0, 1, m+1, endpoint=False)[1:]
    # x_new_grid, y_new_grid = np.meshgrid(sensor_pts, sensor_pts)
    for i in range(N):
        if i % 100 == 0:
            print(f"Generating the {i}th input")
        in_data = GRF(alpha, tau, m)
        in_data = np.where(in_data >= 0, np.float64(12.0), np.float64(4.0))
        f_all[i] = in_data
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # ax.plot_surface(x_new_grid, y_new_grid, f_all[i], cmap='viridis')
        # ax.set_xlabel('x')
        # ax.set_ylabel('y')
        # ax.set_zlabel('Predicted u(x, y)')
        # ax.set_title('Predicted Random Field using Gaussian Process (2D)')
        # plt.show()

    return f_all
