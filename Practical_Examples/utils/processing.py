#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Files related to taking derivatives using finite differences and convolutions
"""
import jax.numpy as jnp
from jax import lax


def diff_x(U, L):
    """
    Compute the derivative with respect to x of function U whose values are
    given on a grid with num_pts_x * num_pts_y equally spaced points using the
    stencil [[-1/2, 0, 1/2]] in the interior and [-3/2, 2, -1/2] at the left edge
    and [1/2, -2, 3/2] at the right edge

    Parameters
    ----------
    U : (4d tensor) (batch_size * num_pts_y * num_pts_x * 1)
        the function to be differentiated evaluated at uniformly spaced points on a grid.
    L : (float)
        length of the domain (in the x direction)

    Returns
    -------
    dudx_conv : (4d tensor of shape batch_size * num_pts_y * num_pts_x * 1)
        the values of the x-derivative evaluated at the points on the grid
    """
    num_pts_x = U.shape[2]
    num_pts_y = U.shape[1]
    batch_size = U.shape[0]

    filter_dx = jnp.array([[-1 / 2, 0, 1 / 2]])[jnp.newaxis, jnp.newaxis, :, :]
    u_2d = jnp.reshape(U, (batch_size, 1, num_pts_y, num_pts_x))
    dudx_conv = lax.conv(u_2d, filter_dx, (1, 1), padding='SAME')
    dudx_left = -3 / 2 * u_2d[:, :, :, 0:1] + 2 * u_2d[:, :, :, 1:2] - 1 / 2 * u_2d[:, :, :, 2:3]
    dudx_right = 1 / 2 * u_2d[:, :, :, -3:-2] - 2 * u_2d[:, :, :, -2:-1] + 3 / 2 * u_2d[:, :, :, -1:]
    dudx_conv = dudx_conv.at[:, :, :, 0:1].set(dudx_left)
    dudx_conv = dudx_conv.at[:, :, :, -1:].set(dudx_right)
    dudx_conv = jnp.reshape(dudx_conv, (batch_size, num_pts_y, num_pts_x, 1))
    dudx_conv *= (num_pts_x - 1) / L
    return dudx_conv


def diff_y(U, W):
    """
    Compute the derivative with respect to y of function U whose values are
    given on a grid with num_pts_x * num_pts_y equally spaced points using the
    stencil [[-1/2], [0], [1/2]] in the interior and [-3/2, 2, -1/2] at the bottom edge
    and [1/2, -2, 3/2] at the top edge

    Parameters
    ----------
    U : (4d tensor) (batch_size * num_pts_y * num_pts_x * 1)
        the function to be differentiated evaluated at uniformly spaced points on a grid.
    W : (float)
        width of the domain (in the y direction)

    Returns
    -------
    dudy_conv : (4d tensor of shape batch_size * num_pts_y * num_pts_x * 1)
        the values of the x-derivative evaluated at the points on the grid
    """
    num_pts_x = U.shape[2]
    num_pts_y = U.shape[1]
    batch_size = U.shape[0]

    filter_dy = jnp.array([[-1 / 2], [0], [1 / 2]])[jnp.newaxis, jnp.newaxis, :, :]
    u_2d = jnp.reshape(U, (batch_size, 1, num_pts_y, num_pts_x))
    dudy_conv = lax.conv(u_2d, filter_dy, (1, 1), padding='SAME')
    dudy_bottom = -3 / 2 * u_2d[:, :, 0:1, :] + 2 * u_2d[:, :, 1:2, :] - 1 / 2 * u_2d[:, :, 2:3, :]
    dudy_top = 1 / 2 * u_2d[:, :, -3:-2, :] - 2 * u_2d[:, :, -2:-1, :] + 3 / 2 * u_2d[:, :, -1:, :]
    dudy_conv = dudy_conv.at[:, :, 0:1, :].set(dudy_bottom)
    dudy_conv = dudy_conv.at[:, :, -1:, :].set(dudy_top)
    dudy_conv = jnp.reshape(dudy_conv, (batch_size, num_pts_y, num_pts_x, 1))
    dudy_conv *= (num_pts_y - 1) / W
    return dudy_conv


def difference_x(U):
    num_pts_x = U.shape[2]
    num_pts_y = U.shape[1]
    batch_size = U.shape[0]

    filter_dx = jnp.array([[-1., 1]])[jnp.newaxis, jnp.newaxis, :, :]
    u_2d = jnp.reshape(U, (batch_size, 1, num_pts_y, num_pts_x))
    dudx_conv = lax.conv(u_2d, filter_dx, (1, 1), padding='SAME')
    dudx_right = -1 * u_2d[:, :, :, -2:-1] + 1 * u_2d[:, :, :, -1:]
    dudx_conv = dudx_conv.at[:, :, :, -1:].set(dudx_right)
    dudx_conv = jnp.reshape(dudx_conv, (batch_size, num_pts_y, num_pts_x, 1))
    return dudx_conv


def difference_y(U):
    num_pts_x = U.shape[2]
    num_pts_y = U.shape[1]
    batch_size = U.shape[0]

    filter_dy = jnp.array([[-1.], [1]])[jnp.newaxis, jnp.newaxis, :, :]
    u_2d = jnp.reshape(U, (batch_size, 1, num_pts_y, num_pts_x))
    dudy_conv = lax.conv(u_2d, filter_dy, (1, 1), padding='SAME')
    dudy_top = - 1 * u_2d[:, :, -2:-1, :] + 1 * u_2d[:, :, -1:, :]
    dudy_conv = dudy_conv.at[:, :, -1:, :].set(dudy_top)
    dudy_conv = jnp.reshape(dudy_conv, (batch_size, num_pts_y, num_pts_x, 1))
    return dudy_conv
