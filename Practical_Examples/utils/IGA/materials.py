#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Classes for material properties
"""
import numpy as np


# Define RBF kernel
def RBF(x1, x2, lengthScales):
    diffs = np.expand_dims(x1 / lengthScales, 1) - \
            np.expand_dims(x2 / lengthScales, 0)
    r2 = np.sum(diffs ** 2, axis=2)
    return np.exp(-0.5 * r2)


def GRF(N, h, mean=0, variance=1, length_scale=0.1):
    jitter = 1e-10
    X = np.linspace(0, h, N)[:, None]
    K = RBF(X, X, length_scale)
    L = np.linalg.cholesky(K + jitter * np.eye(N))
    gp_sample = variance * np.dot(L, np.random.normal(size=N)) + mean
    return X.flatten(), gp_sample


class MaterialElast2D:
    """
    Class for 2D linear elastic materials
    Input
    -----
    Emod : (float) Young's modulus
    nu : (float) Poisson ratio
    plane_type : (string) stress or strain
    """
    def __init__(self, Emod=None, nu=None, plane_type="stress"):
        self.Emod = Emod
        self.nu = nu
        if plane_type == "stress":
            self.Cmat = Emod / (1 - nu ** 2) * np.array([[1, nu, 0], [nu, 1, 0],
                                                         [0, 0, (1 - nu) / 2]])
        elif plane_type == "strain":
            self.Cmat = Emod / ((1 + nu) * (1 - 2 * nu)) * np.array([[1 - nu, nu, 0], [nu, 1 - nu, 0],
                                                                     [0, 0, (1 - 2 * nu) / 2]])


class MaterialElast2D_FGM:
    def __init__(self, Emod=None, nu=None, vertices=None, gp=None, N=None, plane_type="stress"):
        self.E = None
        self.nu = nu
        self.Emod = Emod
        if plane_type == "stress":
            self.Cmat = 1 / (1 - nu ** 2) * np.array([[1, nu, 0], [nu, 1, 0], [0, 0, (1 - nu) / 2]])
        elif plane_type == "strain":
            self.Cmat = 1 / ((1 + nu) * (1 - 2 * nu)) * np.array([[1 - nu, nu, 0], [nu, 1 - nu, 0],
                                                                  [0, 0, (1 - 2 * nu) / 2]])
        # b = vertices[3][0]
        h = vertices[3][1]
        self.Y = np.linspace(0, h, N)[:, None].flatten()
        self.gp = gp

    def elasticity(self, coordinates, mesh=None):
        # xPhys = coordinates[0]
        yPhys = coordinates[1]
        self.E = np.interp(yPhys, self.Y, self.gp)
        return self.E


class MaterialElast2D_RandomFGM:
    """
    Class for 2D linear elastic materials
    Input
    -----
    Emod : (float) Young's modulus
    nu : (float) Poisson ratio
    plane_type : (string) stress or strain
    """
    def __init__(self, N_material, Emod=None, e0_bot=None, e0_top=None, nu=None, vertices=None,
                 length_scale=None, plane_type="stress"):
        self.E = None
        self.nu = nu
        self.Emod = Emod
        if plane_type == "stress":
            self.Cmat = 1 / (1 - nu ** 2) * np.array([[1, nu, 0], [nu, 1, 0], [0, 0, (1 - nu) / 2]])
        elif plane_type == "strain":
            self.Cmat = 1 / ((1 + nu) * (1 - 2 * nu)) * np.array([[1 - nu, nu, 0], [nu, 1 - nu, 0],
                                                                  [0, 0, (1 - 2 * nu) / 2]])
        # b = vertices[3][0]
        h = vertices[3][1]
        self.Y, self.gp = GRF(N_material, h, length_scale=length_scale)
        Emin = (1 - e0_bot) * Emod
        Emax = (1 + e0_top) * Emod
        gmax = np.max(self.gp)
        gmin = np.min(self.gp)
        self.gp = (Emax - Emin) / (gmax - gmin) * (self.gp - gmin) + Emin

    def elasticity(self, coordinates, mesh=None):
        """
        Retrieves the elasticity modulus for FGM beam based on physical coordinates (xPhys, yPhys) of points in the beam.

        Parameters
        ----------
        coordinates : ndarray
            An array representing the physical coordinates (xPhys, yPhys) of points in the beam.
        mesh
        Returns
        -------
        E : float
            The elasticity modulus for the given point in the beam.
        """
        # xPhys = coordinates[0]
        yPhys = coordinates[1]
        self.E = np.interp(yPhys, self.Y, self.gp)
        return self.E


class MaterialElast2D_Hole:
    def __init__(self, Emod=None, nu=None, vertices=None, elasticity_fun=None, plane_type="stress"):
        self.E = None
        self.nu = nu
        self.Emod = Emod
        if plane_type == "stress":
            self.Cmat = 1 / (1 - nu ** 2) * np.array([[1, nu, 0], [nu, 1, 0], [0, 0, (1 - nu) / 2]])
        elif plane_type == "strain":
            self.Cmat = 1 / ((1 + nu) * (1 - 2 * nu)) * np.array([[1 - nu, nu, 0], [nu, 1 - nu, 0],
                                                                  [0, 0, (1 - 2 * nu) / 2]])
        self.elasticity_fun = elasticity_fun
        self.vertices = vertices

    def elasticity(self, coordinates, mesh=None):
        L = self.vertices[3][0]
        W = self.vertices[3][1]
        x = coordinates[0] if 0 <= coordinates[0] <= L else (0 if coordinates[0] < 0 else L)
        y = coordinates[1] if 0 <= coordinates[1] <= W else (0 if coordinates[1] < 0 else W)
        self.E = self.elasticity_fun((x, y))
        if np.isnan(np.min(self.E)):
            print("self.E is nan")
        return self.E


