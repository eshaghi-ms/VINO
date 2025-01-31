import os
import numpy as np
from typing import Tuple
from scipy.fftpack import idct
from timeit import default_timer
from torch.utils.data import Dataset

from .pde_solvers import darcy2D_solver, FGMBeam_solver, PlateHole_solver


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
    f_all = np.zeros((N, m, m), dtype=np.float64)
    # sensor_pts = np.linspace(0, 1, m + 1, endpoint=False, dtype=np.float64)[1:]
    # x_new_grid, y_new_grid = np.meshgrid(sensor_pts, sensor_pts)
    for i in range(N):
        if (i+1) % 200 == 0:
            print(f"Generating the {i+1}th input")
        in_data = GRF(alpha, tau, m)
        in_data = np.where(in_data >= 0, 12., 4.)
        f_all[i] = in_data
    return f_all


def gaussian_normalize(x: np.ndarray, eps=0.00001) -> Tuple[np.ndarray, float, float]:
    """
  Adapted from
  https://github.com/zongyi-li/fourier_neural_operator/blob/c13b475dcc9bcd855d959851104b770bbcdd7c79/utilities3.py#L73

  Attributes:
    x (np.ndarray): Input array
    eps (float): Small number to avoid division by zero

  Returns:
    Tuple[np.ndarray, float, float]: Normalized array, mean and standard deviation
  """
    mean = np.mean(x, 0, keepdims=True)
    std = np.std(x, 0, keepdims=True)
    x = (x - mean) / (std + eps)
    return x, mean, std


class GaussianNormalizer:
    def __init__(self, x, eps=1e-5):
        self.mean = np.mean(x, 0, keepdims=True)
        self.std = np.std(x, 0, keepdims=True)
        self.eps = eps

    def encode(self, x):
        return (x - self.mean) / (self.std + self.eps)

    def decode(self, x):
        std = self.std + self.eps
        mean = self.mean
        return (x * std) + mean


def ElementsToNodes(elementValues):
    temp = np.pad(elementValues, ((0, 0), (0, 1), (0, 1), (0, 0)), mode='edge')
    return (temp[:, :-1, :-1, :] + temp[:, 1:, :-1, :] + temp[:, :-1, 1:, :] + temp[:, 1:, 1:, :]) / 4


class DarcyDataset(Dataset):
    def __init__(self, model_data):
        if not os.path.exists(model_data["dir"]):
            os.makedirs(model_data["dir"])

        if os.path.exists(model_data["path"]):
            print("Found saved dataset at", model_data["path"])
            loaded_data = np.load(model_data["path"])
            inputs = loaded_data['inputs']
            targets = loaded_data['targets']
        else:
            inputs = generate_inputs(
                model_data["n_data"], model_data["grid_point_num"], model_data["GRF"]["alpha"], model_data["GRF"]["tau"])
            targets = darcy2D_solver(inputs)
            np.savez(model_data.path, inputs=inputs, targets=targets)

        # Extract data and put batch dimension in front
        self.x = inputs
        self.y = targets

        # Add channel dimension at the end
        self.x = self.x[..., np.newaxis]
        self.y = self.y[..., np.newaxis]

        # Normalize data
        if model_data["normalized"]:
            #self.x, _, _ = gaussian_normalize(self.x)
            #self.y, _, _ = gaussian_normalize(self.y)

            self.normalizer_x = GaussianNormalizer(self.x)
            self.normalizer_y = GaussianNormalizer(self.y)
            self.x = self.normalizer_x.encode(self.x)
            self.y = self.normalizer_y.encode(self.y)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class FGM_dataset(Dataset):
    def __init__(self, model_data):
        n = model_data["n_train"] + model_data["n_test"]
        n_u = model_data['beam']['numPtsU']
        n_v = model_data['beam']['numPtsV']
        if not os.path.exists(model_data['dir']):
            os.makedirs(model_data['dir'])

        if os.path.exists(model_data['path']):
            print("Found saved dataset at", model_data['path'])
            loaded_data = np.load(model_data['path'])
            traction = loaded_data['traction']
            material = loaded_data['material']
            disp2d = loaded_data['disp2D']
        else:
            print("Creating database")
            traction = np.zeros([n, n_u])
            material = np.zeros([n, n_v])
            disp2d = np.zeros([n, n_v, n_u, 2])
            for i in range(n):
                t1 = default_timer()
                traction[i, :], material[i, :], disp2d[i, :, :, :] = FGMBeam_solver(model_data['beam'])
                print(i, default_timer() - t1)
            np.savez(model_data['path'], traction=traction, material=material, disp2D=disp2d)

        # fix dimensions
        traction = traction.reshape(model_data['n_data'], 1, n_u, 1)
        material = material.reshape(model_data['n_data'], n_v, 1, 1)

        # repeat inputs
        traction = np.tile(traction, (1, n_v, 1, 1))
        material = np.tile(material, (1, 1, n_u, 1))
        # material = ElementsToNodes(material)
        # traction = ElementsToNodes(traction)

        # repeat inputs
        self.x = np.concatenate([traction, material], axis=-1)
        self.y = disp2d

        if model_data['normalized']:
            # self.x, _, _ = gaussian_normalize(self.x)
            # self.y, _, _ = gaussian_normalize(self.y)
            self.normalizer_x = GaussianNormalizer(self.x)
            self.normalizer_y = GaussianNormalizer(self.y)
            self.x = self.normalizer_x.encode(self.x)
            self.y = self.normalizer_y.encode(self.y)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class Hyperelasticity_dataset(Dataset):
    def __init__(self, model_data):
        n = model_data["fno"]["n_data"]
        n_u = model_data["beam"]["numPtsU"]
        n_v = model_data["beam"]["numPtsV"]
        if not os.path.exists(model_data["dir"]):
            os.makedirs(model_data["dir"])
        print(model_data)
        if os.path.isfile(model_data["path"]):
            loaded_data = np.load(model_data["path"])
            traction = loaded_data['traction']
            disp2d = loaded_data['disp2D']
            print("Found saved dataset at", model_data["path"])
        else:
            print("NOT Found saved dataset at", model_data["path"])
            raise ValueError("Please run the python code for generating dataset or download the database.")

        # fix dimensions
        traction = traction.reshape(n, n_v, 1, 1)

        # repeat inputs
        traction = np.tile(traction, (1, 1, n_u, 1))

        # repeat inputs
        self.x = traction
        self.y = disp2d

        if model_data["normalized"]:
            #self.x, _, _ = gaussian_normalize(self.x)
            #self.y, _, _ = gaussian_normalize(self.y)
            self.normalizer_x = GaussianNormalizer(self.x)
            self.normalizer_y = GaussianNormalizer(self.y)
            self.x = self.normalizer_x.encode(self.x)
            self.y = self.normalizer_y.encode(self.y)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class PlateHole_dataset(Dataset):
    def __init__(self, model_data):
        self.y = None
        self.x = None
        self.normalizer_x = None
        self.normalizer_y = None
        self.model_data = model_data
        n = model_data["n_train"] + model_data["n_test"]
        n_u = model_data['beam']['numPtsU']
        n_v = model_data['beam']['numPtsV']
        if not os.path.exists(model_data['dir']):
            os.makedirs(model_data['dir'])

        if os.path.exists(model_data['path']):
            print("Found saved dataset at", model_data['path'])
            loaded_data = np.load(model_data['path'])
            self.material = loaded_data['material']
            self.disp2D = loaded_data['disp2D']
            self.stress2D = loaded_data['stress2D']
        else:
            print("Creating database")
            self.material = np.zeros([n, n_v, n_u, 1])
            self.disp2D = np.zeros([n, n_v, n_u, 2])
            self.stress2D = np.zeros([n, n_v, n_u, 3])
            for i in range(n):
                t1 = default_timer()
                self.material[i, :, :, 0], self.disp2D[i, :, :, :], self.stress2D[i, :, :, :] = PlateHole_solver(model_data['beam'], model_data["GRF"])
                print(i, default_timer() - t1)
            np.savez(model_data['path'], material=self.material, disp2D=self.disp2D, stress2D=self.stress2D)
        self.set_data()

    def set_data(self):
        self.x = self.material
        self.y = self.disp2D
        # self.y = self.stress2D
        if self.model_data['normalized']:
            self.make_normal()

    def make_normal(self):
        self.normalizer_x = GaussianNormalizer(self.x)
        self.normalizer_y = GaussianNormalizer(self.y)
        self.x = self.normalizer_x.encode(self.x)
        # self.x = self.x - 0.5
        self.y = self.normalizer_y.encode(self.y)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]



