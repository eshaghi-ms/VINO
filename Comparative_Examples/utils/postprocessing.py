#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions for plotting and post-processing
"""
import os
import numpy as np
import matplotlib.pyplot as plt


def plot_pred(x, u_exact, u_pred, title):
    fig_font = "DejaVu Serif"
    fig_size = (5, 3)
    plt.rcParams["font.family"] = fig_font
    plt.figure(figsize=fig_size)
    plt.plot(x, u_exact, 'b', label='Ground Truth', linewidth=1.5)
    plt.plot(x, u_pred, '-.r', label='Prediction', linewidth=1.5)
    plt.legend()
    plt.title(title)
    title = title.replace('\\', '').replace('$', '')
    os.makedirs('plots', exist_ok=True)
    plt.savefig('plots/' + title + '.png', dpi=300)
    plt.show()
    rel_l2_error = np.linalg.norm(u_exact - u_pred) / np.linalg.norm(u_exact)
    print("Relative L2 error is ", rel_l2_error)

    error_title = f"Relative $L^2$ error = {rel_l2_error:.4f}"
    print(error_title)
    plt.rcParams["font.family"] = fig_font
    plt.figure(figsize=fig_size)
    plt.plot(x, u_exact - u_pred, 'b', label='Ground Truth', linewidth=1.5)
    # plt.legend()
    plt.title(error_title)
    # plt.savefig('error_' + title + '.png', dpi=300)
    plt.show()


def plot_pred1(x, y, f, title):
    fig_font = "DejaVu Serif"
    plt.rcParams["font.family"] = fig_font
    plt.figure()
    plt.contourf(x, y, f, levels=2, cmap='Purples')
    plt.colorbar()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title('Input ' + title)
    plt.savefig('Input ' + title + '.png', dpi=600)
    plt.show()


def plot_pred2(x, y, u_exact, u_pred, title, saved_title):
    saved_title = saved_title.replace('\\', '').replace('$', '')
    fig_font = "DejaVu Serif"
    plt.rcParams["font.family"] = fig_font
    plt.figure()
    plt.contourf(x, y, u_pred, levels=500, cmap='jet')
    plt.colorbar()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title(title)
    plt.savefig(saved_title + ' - Approximate solution' + '.png', dpi=600)
    plt.show()

    plt.figure()
    plt.contourf(x, y, u_exact, levels=500, cmap='jet')
    plt.colorbar()
    plt.gca().set_aspect('equal', adjustable='box')
    # plt.title('Exact solution - ' + title)
    plt.savefig(saved_title + ' - Exact solution' + '.png', dpi=600)
    plt.show()

    plt.figure()
    plt.contourf(x, y, u_exact - u_pred, levels=500, cmap='bwr')
    plt.colorbar()
    plt.gca().set_aspect('equal', adjustable='box')
    # plt.title('Error - ' + title)
    plt.savefig(saved_title + ' - Error' + '.png', dpi=600)
    rel_l2_error = np.linalg.norm(u_exact - u_pred) / np.linalg.norm(u_exact)
    print("Relative L2 error is ", rel_l2_error)
    plt.show()


def plot_field_2d(F, x_pts, y_pts, title, folder=None, file=None):
    """
    Plots a 2D field stored in a 1D tensor F

    Parameters
    ----------
    F : (1D tensor of size num_pts_v*num_pts_u)
        fields values at each point
    x_pts : (2D tensor of size num_pts_v x num_pts_u)
        x-coordinates of each field point to be plotted
    y_pts : (2d tensor of size num_pts_v x num_pts_u)
        y-coordinates of each field points to be plotted
    title : (string)
        title of the plot
    folder : (None or string)
        directory where to save the plot
    file : (None or string)
        file name for the plot

    Returns
    -------
    None.

    """
    plt.contourf(x_pts, y_pts, F, 255, cmap=plt.cm.jet)
    plt.colorbar()
    plt.title(title)
    plt.axis('equal')
    if folder is not None:
        full_name = folder + '/' + file
        plt.savefig(full_name)
    plt.show()
