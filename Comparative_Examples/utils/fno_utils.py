#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions for the Fourier Neural Operator
Adapted from https://github.com/neuraloperator/neuraloperator/blob/master/utilities3.py
"""
import torch
import torch.nn.functional as F
import numpy as np
import operator
from functools import reduce
from timeit import default_timer

from torch.nn import functional as F


# loss function with rel/abs Lp loss
class LpLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()

        # Dimension and Lp-norm type are positive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]

        # Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h ** (self.d / self.p)) * torch.norm(x.view(num_examples, -1) - y.view(num_examples, -1), self.p,
                                                          1)

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]

        diff_norms = torch.norm(x.reshape(num_examples, -1) - y.reshape(num_examples, -1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples, -1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms / y_norms)
            else:
                return torch.sum(diff_norms / y_norms)

        return diff_norms / y_norms

    def __call__(self, x, y):
        return self.rel(x, y)


# print the number of parameters
def count_params(model):
    c = 0
    for p in list(model.parameters()):
        c += reduce(operator.mul,
                    list(p.size() + (2,) if p.is_complex() else p.size()))
    return c


def du_FD(U, device):
    """
    Computes the first derivative of U using the finite difference
    stencil [-1/2, 0, 1/2] in the interior, the left edge stencil
    [-1, 1], and the right edge stencil [-1, 1]

    Parameters
    ----------
    U : (tensor of dimension batch_size x 1 x num_nodes)
        function u(X) evaluated at equally spaced nodes in each row

    device : (device type)
        cuda (for GPU ) or cpu


    Returns
    -------
    F_fd : (tensor of dimension batch_size x 1 x(num_nodes+2))
        function f(x) = u\'(x) evaluated at the nodes and endpoints of the interval

    """
    num_nodes = U.shape[2]
    filters = torch.tensor([-1 / 2, 0., 1 / 2]).unsqueeze(0).unsqueeze(0).to(device)
    f_fd = F.conv1d(U, filters, padding=0, groups=1)
    left_col = -1 * U[:, :, 0:1] + 1 * U[:, :, 1:2]
    right_col = -1 * U[:, :, -2:-1] + 1 * U[:, :, -1:]
    f_fd = torch.cat((left_col, f_fd, right_col), dim=2)
    f_fd *= (num_nodes - 1)
    return f_fd


def d2u_FD(U, device):
    """
    Computes the (negative) second derivative of U using the finite difference
    stencil [-1, 2, -1]

    Parameters
    ----------
    U : (tensor array) dimension batch_size x 1 x num_nodes
        function u(X) evaluated at equally spaced nodes in each row

    device : (device type)
        cuda (for GPU ) or cpu


    Returns
    -------
    F_fd : (tensor of same dimension as U, padded with zeros)
        function f(x) = -u''(x) evaluated at the nodes

    """
    num_nodes = U.shape[2]
    filters = torch.tensor([-1., 2., -1.]).unsqueeze(0).unsqueeze(0).to(device)
    f_fd = F.conv1d(U, filters, padding=0, groups=1)
    f_fd *= (num_nodes + 1) ** 2
    return f_fd


def d4u_FD(U, device):
    """
    Computes the (negative) fourth derivative of U using the finite difference
    stencil [1, -4, 6, -4, 1]

    Parameters
    ----------
    U : (tensor array) dimension batch_size x 1 x num_nodes
        Function u(X) evaluated at equally spaced nodes in each row

    device : (device type)
        cuda (for GPU) or cpu

    Returns
    -------
    F_fd : (tensor of same dimension as U, padded with zeros)
        Function f(x) = -u''''(x) evaluated at the nodes
    """
    num_nodes = U.shape[2]
    filters = torch.tensor([1., -4., 6., -4., 1.]).unsqueeze(0).unsqueeze(0).to(device)
    f_fd = F.conv1d(U, filters, padding=0, groups=1)
    f_fd *= (num_nodes + 1) ** 4  # Scale by the factor based on spacing
    return f_fd


def Laplacian(U, device):
    """
    Computes the (negative) laplacian of U using the finite difference
    stencil [ 0, -1,  0
             -1,  4, -1
              0, -1,  0]

    Parameters
    ----------
    U : (tensor of dimension batch_size x 1 x num_nodes x num_nodes)

    device : (device type)
        cuda (for GPU ) or cpu


    Returns
    -------
    F_fd : (tensor of same dimension as U, padded with zeros)
        function f(x) = -\nabla^2 u(x, y) evaluated at the nodes

    """
    num_nodes = U.shape[2]
    filters = torch.tensor([[0., -1., 0.], [-1., 4., -1.], [0., -1., 0.]]).unsqueeze(0).unsqueeze(0).to(device)
    f_fd = F.conv2d(U, filters, groups=1)
    f_fd *= (num_nodes - 1) ** 2
    return f_fd


def diff_x(U, device, L=1):
    num_nodes = U.shape[3]
    filter_dx = torch.tensor([[-1 / 2, 0., 1 / 2]]).unsqueeze(0).unsqueeze(0).to(device)
    dudx = F.conv2d(U, filter_dx)
    dudx_left = -3 / 2 * U[:, :, :, 0:1] + 2 * U[:, :, :, 1:2] - 1 / 2 * U[:, :, :, 2:3]
    dudx_right = 1 / 2 * U[:, :, :, -3:-2] - 2 * U[:, :, :, -2:-1] + 3 / 2 * U[:, :, :, -1:]
    dudx = torch.cat((dudx_left, dudx, dudx_right), dim=3)
    dudx *= (num_nodes - 1) / L
    return dudx


def diff_y(U, device, W=1):
    num_nodes = U.shape[2]
    filter_dy = torch.tensor([[-1 / 2], [0.], [1 / 2]]).unsqueeze(0).unsqueeze(0).to(device)
    dudy = F.conv2d(U, filter_dy)
    dudy_bottom = -3 / 2 * U[:, :, 0:1, :] + 2 * U[:, :, 1:2, :] - 1 / 2 * U[:, :, 2:3, :]
    dudy_top = 1 / 2 * U[:, :, -3:-2, :] - 2 * U[:, :, -2:-1, :] + 3 / 2 * U[:, :, -1:, :]
    dudy = torch.cat((dudy_bottom, dudy, dudy_top), dim=2)
    dudy *= (num_nodes - 1) / W
    return dudy


def difference_x(U, device, data_type=torch.float32):
    filter_dx = torch.tensor([[-1., 1]], dtype=data_type).unsqueeze(0).unsqueeze(0).to(device)
    dudx = F.conv2d(U, filter_dx)
    dudx_right = -1 * U[:, :, :, -2:-1] + 1 * U[:, :, :, -1:]
    dudx = torch.cat((dudx, dudx_right), dim=3)
    return dudx


def difference_y(U, device, data_type=torch.float32):
    filter_dy = torch.tensor([[-1], [1]], dtype=data_type).unsqueeze(0).unsqueeze(0).to(device)
    dudy = F.conv2d(U, filter_dy)
    dudy_top = -1 * U[:, :, -2:-1, :] + 1 * U[:, :, -1:, :]
    dudy = torch.cat((dudy, dudy_top), dim=2)
    return dudy


def simpsons_integral_4d(tensor, dx, dy):
    simpson_coeff_x = torch.ones_like(tensor[:, :, 0, 0])
    simpson_coeff_y = torch.ones_like(tensor[:, 0, :, 0])

    simpson_coeff_x[1:-1:2] = 4
    simpson_coeff_x[2:-1:2] = 2
    simpson_coeff_y[1:-1:2] = 4
    simpson_coeff_y[2:-1:2] = 2

    simpson_coeff_x = simpson_coeff_x.unsqueeze(2).unsqueeze(3)
    simpson_coeff_y = simpson_coeff_y.unsqueeze(1).unsqueeze(3)

    integral_tensor = tensor * simpson_coeff_x * simpson_coeff_y
    integral = integral_tensor.sum(dim=1).sum(dim=1)
    integral *= dx * dy / 9.0
    return integral


def train_fno(model, train_loader, test_loader, loss_func, optimizer,
              scheduler, device, epochs, batch_size):
    model.train()
    n_train = len(train_loader)
    n_test = len(test_loader)
    train_mse_log = np.zeros(epochs)
    train_l2_log = np.zeros(epochs)
    test_l2_log = np.zeros(epochs)
    for ep in range(epochs):
        t1 = default_timer()
        train_mse = 0
        train_l2 = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)

            mse = F.mse_loss(out.view(batch_size, -1), y.view(batch_size, -1), reduction='mean')
            l2 = loss_func(out.view(batch_size, -1), y.view(batch_size, -1)) / len(x)
            l2.backward()  # use the l2 relative loss

            optimizer.step()
            train_mse += mse.item()
            train_l2 += l2.item()

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(train_mse)
        else:
            scheduler.step()
        model.eval()
        test_l2 = 0.0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                test_l2 += loss_func(out.view(batch_size, -1), y.view(batch_size, -1)).item() / len(x)

        train_mse /= n_train
        train_l2 /= n_train
        test_l2 /= n_test

        train_mse_log[ep] = train_mse
        train_l2_log[ep] = train_l2
        test_l2_log[ep] = test_l2
        t2 = default_timer()
        if ep % 50 == 0:
            print(ep, t2 - t1, train_mse, train_l2, test_l2)

    return train_mse_log, train_l2_log, test_l2_log


def train_pino_antiderivative(model, train_loader, test_loader, loss_func, optimizer, scheduler,
                              device, epochs, batch_size, data=False):
    model.train()
    n_train = len(train_loader)
    n_test = len(test_loader)
    train_mse_log = np.zeros(epochs)
    train_l2_log = np.zeros(epochs)
    test_l2_log = np.zeros(epochs)
    for ep in range(epochs):
        t1 = default_timer()
        train_mse = 0
        train_l2 = 0
        train_l2_pde = 0
        train_l2_data = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            u_pred = F.pad(out, (0, 0, 1, 1), mode='constant', value=0)
            f_pred = d2u_FD(u_pred.permute(0, 2, 1), device).permute(0, 2, 1)

            mse_pde = F.mse_loss(x.view(batch_size, -1), f_pred.view(batch_size, -1), reduction='mean') / len(x)
            l2_pde = loss_func(x.view(batch_size, -1), f_pred.view(batch_size, -1)) / len(x)

            mse_data = F.mse_loss(out.view(batch_size, -1), y.view(batch_size, -1), reduction='mean')
            l2_data = loss_func(out.view(batch_size, -1), y.view(batch_size, -1)) / len(x)

            if data:
                mse = mse_data + mse_pde
                l2 = l2_data + l2_pde
            else:
                mse = mse_pde
                l2 = l2_pde

            l2.backward()

            optimizer.step()
            train_mse += mse.item()
            train_l2 += l2.item()
            train_l2_data += l2_data.item()
            train_l2_pde += l2_pde.item()

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(train_mse)
        else:
            scheduler.step()
        model.eval()
        test_l2 = 0.0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)

                out = model(x)
                u_pred = F.pad(out, (0, 0, 1, 1), mode='constant', value=0)
                f_pred = d2u_FD(u_pred.permute(0, 2, 1), device).permute(0, 2, 1)
                test_l2_pde = loss_func(x.view(batch_size, -1), f_pred.view(batch_size, -1)).item() / len(x)
                test_l2_data = loss_func(out.view(batch_size, -1), y.view(batch_size, -1)).item() / len(x)
                if data:
                    test_l2 += test_l2_pde + test_l2_data
                else:
                    test_l2 += test_l2_pde

        train_mse /= n_train
        train_l2 /= n_train
        train_l2_data /= n_train
        train_l2_pde /= n_train
        test_l2 /= n_test

        train_mse_log[ep] = train_mse
        train_l2_log[ep] = train_l2
        test_l2_log[ep] = test_l2
        t2 = default_timer()
        if ep % 50 == 0:
            print(ep, t2 - t1, train_mse, train_l2, test_l2, train_l2_data, train_l2_pde)

    return train_mse_log, train_l2_log, test_l2_log


def train_vino_antiderivative(model, train_loader, test_loader, loss_func, optimizer, scheduler,
                              device, epochs, batch_size, data=False):
    model.train()
    n_train = len(train_loader)
    n_test = len(test_loader)
    train_l2_log = np.zeros(epochs)
    test_l2_log = np.zeros(epochs)
    print("ep,      t2 - t1,        train_l2,       test_l2,        train_l2_data,      train_l2_pde")
    for ep in range(epochs):
        t1 = default_timer()
        l2 = 0
        train_l2 = 0
        train_l2_pde = 0
        train_l2_data = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            mask = torch.ones_like(x)
            optimizer.zero_grad()
            out = model(x)
            u_pred = F.pad(out, (0, 0, 1, 1), mode='constant', value=0)

            x = F.pad(x, (0, 0, 1, 1), mode='constant', value=0)
            mask = F.pad(mask, (0, 0, 1, 1), mode='constant', value=0.5)

            du_pred = du_FD(u_pred.permute(0, 2, 1), device).permute(0, 2, 1)
            integrand = 1 / 2 * du_pred ** 2 - u_pred * x
            l2_pde = torch.mean(integrand * mask) / len(x)

            #integrand_1 = u_pred * x
            #loss_1 = torch.mean(integrand_1 * mask) / len(x)
            #dx = 1 / u_pred.shape[1]
            #loss_2 = torch.mean((u_pred[:, 0:-1, :] - u_pred[:, 1:, :]) ** 2) / (2 * dx ** 2) / len(x)
            #l2_pde = loss_2 - loss_1

            #U_0 = u_pred[:, 0:-1, :]
            #U_1 = u_pred[:, 1:, :]

            #l2_pde = torch.sum(
            #    (U_0 ** 2 / 2 - U_0 * U_1 + U_1 ** 2 / 2) / dx ** 2 + dx * (U_0 / 6 + U_1 / 3)
            #) / len(x)

            if data:
                l2_data = loss_func(out.view(batch_size, -1), y.view(batch_size, -1)) / len(x)
                l2 = l2_data + l2_pde
                train_l2_data += l2_data.item()
            else:
                l2 = l2_pde
            l2.backward()

            optimizer.step()
            train_l2 += l2.item()
            train_l2_pde += l2_pde.item()

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(l2)
        else:
            scheduler.step()
        model.eval()
        test_l2 = 0.0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)

                out = model(x)
                u_pred = F.pad(out, (0, 0, 1, 1), mode='constant', value=0)
                f_pred = d2u_FD(u_pred.permute(0, 2, 1), device).permute(0, 2, 1)
                test_l2_pde = loss_func(x.reshape(batch_size, -1), f_pred.reshape(batch_size, -1)).item() / len(x)
                if data:
                    test_l2_data = loss_func(out.view(batch_size, -1), y.view(batch_size, -1)).item() / len(x)
                    test_l2 += test_l2_pde + test_l2_data
                else:
                    test_l2 += test_l2_pde
        train_l2 /= n_train
        test_l2 /= n_test

        train_l2_log[ep] = train_l2
        test_l2_log[ep] = test_l2
        t2 = default_timer()
        if ep % 50 == 0:
            print(ep, t2 - t1, train_l2, test_l2, train_l2_data, train_l2_pde)

    return train_l2_log, test_l2_log


def train_pino_poisson(model, train_loader, test_loader, loss_func, optimizer, scheduler,
                       device, epochs, step_size, batch_size, data=False):
    model.train()
    n_train = len(train_loader)
    n_test = len(test_loader)
    train_mse_log = np.zeros(epochs)
    train_l2_log = np.zeros(epochs)
    test_l2_log = np.zeros(epochs)
    for ep in range(epochs):
        t1 = default_timer()
        train_mse = 0
        train_l2 = 0
        train_l2_pde = 0
        train_l2_data = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            u_pred = F.pad(out, (0, 0, 1, 1, 1, 1), mode='constant', value=0)
            f_pred = Laplacian(u_pred.permute(0, 3, 1, 2), device).permute(0, 2, 3, 1)

            mse_pde = F.mse_loss(x.view(batch_size, -1), f_pred.view(batch_size, -1), reduction='mean') / len(x)
            l2_pde = loss_func(x.view(batch_size, -1), f_pred.view(batch_size, -1)) / len(x)

            mse_data = F.mse_loss(out.view(batch_size, -1), y.view(batch_size, -1), reduction='mean')
            l2_data = loss_func(out.view(batch_size, -1), y.view(batch_size, -1)) / len(x)

            mse = mse_data + mse_pde
            if data:
                l2 = l2_data + l2_pde
            else:
                l2 = l2_pde
            l2.backward()  # use the l2 relative loss

            optimizer.step()
            train_mse += mse.item()
            train_l2 += l2.item()
            train_l2_data += l2_data.item()
            train_l2_pde += l2_pde.item()

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(train_mse)
        else:
            scheduler.step()
        model.eval()
        test_l2 = 0.0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)

                out = model(x)
                u_pred = F.pad(out, (0, 0, 1, 1, 1, 1), mode='constant', value=0)
                f_pred = Laplacian(u_pred.permute(0, 3, 1, 2), device).permute(0, 2, 3, 1)
                test_l2_pde = loss_func(x.view(batch_size, -1), f_pred.view(batch_size, -1)).item() / len(x)
                test_l2_data = loss_func(out.view(batch_size, -1), y.view(batch_size, -1)).item() / len(x)
                test_l2 += test_l2_pde + test_l2_data
        train_mse /= n_train
        train_l2 /= n_train
        train_l2_data /= n_train
        train_l2_pde /= n_train
        test_l2 /= n_test

        train_mse_log[ep] = train_mse
        train_l2_log[ep] = train_l2
        test_l2_log[ep] = test_l2
        t2 = default_timer()
        if ep % step_size == 0:
            print(ep, t2 - t1, train_mse, train_l2, test_l2, train_l2_data, train_l2_pde)

    return train_mse_log, train_l2_log, test_l2_log


def train_vino_poisson(model, train_loader, test_loader, loss_func, optimizer, scheduler,
                       device, epochs, step_size, batch_size, data=False):
    model.train()
    n_train = len(train_loader)
    n_test = len(test_loader)
    train_l2_log = np.zeros(epochs)
    test_l2_log = np.zeros(epochs)
    for ep in range(epochs):
        t1 = default_timer()
        train_l2 = 0
        train_l2_pde = 0
        train_l2_data = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            mask = torch.ones_like(x)
            optimizer.zero_grad()
            out = model(x)
            u_pred = F.pad(out, (0, 0, 1, 1, 1, 1), mode='constant', value=0)

            du_dx = difference_x(u_pred.permute(0, 3, 1, 2), device).permute(0, 2, 3, 1)
            du_dy = difference_y(u_pred.permute(0, 3, 1, 2), device).permute(0, 2, 3, 1)

            x = F.pad(x, (0, 0, 1, 1, 1, 1), mode='replicate')
            mask = F.pad(mask, (0, 0, 1, 1, 1, 1), mode='constant', value=0.5)
            mask[:, 0, 0, 0] = 0.25
            mask[:, 0, -1, 0] = 0.25
            mask[:, -1, 0, 0] = 0.25
            mask[:, -1, -1, 0] = 0.25

            num_nodes_x = u_pred.shape[2]
            num_nodes_y = u_pred.shape[1]

            loss_grad = 1 / 2 * torch.sum(
                2 / 3 * du_dx[:, 0:-1, 0:-1, :] ** 2 + 2 / 3 * du_dx[:, 1:, 0:-1, :] ** 2 + du_dy[:, 0:-1, 0:-1,
                                                                                            :] ** 2 -
                1 / 3 * du_dx[:, 0:-1, 0:-1, :] * du_dx[:, 1:, 0:-1, :] - du_dx[:, 0:-1, 0:-1, :] * du_dy[:, 0:-1, 0:-1,
                                                                                                    :]
                + du_dx[:, 1:, 0:-1, :] * du_dy[:, 0:-1, 0:-1, :]) / len(x)
            loss_int = torch.sum((x * u_pred) * mask) / len(x) / ((num_nodes_x - 1) * (num_nodes_y - 1))
            l2_pde = loss_grad - loss_int
            l2_data = loss_func(out.view(batch_size, -1), y.view(batch_size, -1)) / len(x)
            if data:
                l2 = l2_data + l2_pde
            else:
                l2 = l2_pde

            l2.backward()

            optimizer.step()
            train_l2 += l2.item()
            train_l2_data += l2_data.item()
            train_l2_pde += l2_pde.item()

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(l2)
        else:
            scheduler.step()
        model.eval()
        test_l2 = 0.0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)

                out = model(x)
                u_pred = F.pad(out, (0, 0, 1, 1, 1, 1), mode='constant', value=0)
                f_pred = Laplacian(u_pred.permute(0, 3, 1, 2), device).permute(0, 2, 3, 1)
                test_l2_pde = loss_func(x.view(batch_size, -1), f_pred.view(batch_size, -1)).item() / len(x)
                test_l2_data = loss_func(out.view(batch_size, -1), y.view(batch_size, -1)).item() / len(x)
                if data:
                    test_l2 += test_l2_pde + test_l2_data
                else:
                    test_l2 += test_l2_pde

        train_l2 /= n_train
        train_l2_data /= n_train
        train_l2_pde /= n_train
        test_l2 /= n_test
        train_l2_log[ep] = train_l2
        test_l2_log[ep] = test_l2
        t2 = default_timer()
        if ep % step_size == 0:
            print(ep, t2 - t1, train_l2, test_l2, train_l2_data, train_l2_pde)

    return train_l2_log, test_l2_log


def train_pino_darcy(model, train_loader, test_loader, loss_func, optimizer, scheduler,
                     device, epochs, step_size, batch_size, data=False):
    model.train()
    n_train = len(train_loader)
    n_test = len(test_loader)
    train_mse_log = np.zeros(epochs)
    train_l2_log = np.zeros(epochs)
    test_l2_log = np.zeros(epochs)
    for ep in range(epochs):
        t1 = default_timer()
        train_mse = 0
        train_l2 = 0
        train_l2_pde = 0
        train_l2_data = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            a = x

            du_dx = diff_x(out.permute(0, 3, 1, 2), device).permute(0, 2, 3, 1)
            du_dy = diff_y(out.permute(0, 3, 1, 2), device).permute(0, 2, 3, 1)

            f_pred_1 = diff_x((a * du_dx).permute(0, 3, 1, 2), device).permute(0, 2, 3, 1)
            f_pred_2 = diff_y((a * du_dy).permute(0, 3, 1, 2), device).permute(0, 2, 3, 1)

            f_pred = -(f_pred_1 + f_pred_2)
            f_ground = torch.ones_like(a)

            mse_pde = F.mse_loss(f_ground.view(batch_size, -1), f_pred.reshape(batch_size, -1), reduction='mean') / len(
                x)
            l2_pde = loss_func(f_ground.view(batch_size, -1), f_pred.reshape(batch_size, -1)) / len(x)

            mse_data = F.mse_loss(out.view(batch_size, -1), y.view(batch_size, -1), reduction='mean')
            l2_data = loss_func(out.view(batch_size, -1), y.view(batch_size, -1)) / len(x)

            mse = mse_data + mse_pde
            if data:
                l2 = l2_data + l2_pde
            else:
                l2 = l2_pde

            l2.backward()  # use the l2 relative loss

            optimizer.step()
            train_mse += mse.item()
            train_l2 += l2.item()
            train_l2_data += l2_data.item()
            train_l2_pde += l2_pde.item()

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(train_mse)
        else:
            scheduler.step()
        model.eval()
        test_l2 = 0.0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                a = x

                du_dx = diff_x(out.permute(0, 3, 1, 2), device).permute(0, 2, 3, 1)
                du_dy = diff_y(out.permute(0, 3, 1, 2), device).permute(0, 2, 3, 1)

                f_pred_1 = diff_x((a * du_dx).permute(0, 3, 1, 2), device).permute(0, 2, 3, 1)
                f_pred_2 = diff_y((a * du_dy).permute(0, 3, 1, 2), device).permute(0, 2, 3, 1)

                f_pred = -(f_pred_1 + f_pred_2)
                f_ground = torch.ones_like(a)

                test_l2_pde = loss_func(f_ground.view(batch_size, -1), f_pred.reshape(batch_size, -1)).item() / len(x)
                test_l2_data = loss_func(out.view(batch_size, -1), y.view(batch_size, -1)).item() / len(x)
                if data:
                    test_l2 += test_l2_pde + test_l2_data
                else:
                    test_l2 += test_l2_pde

        train_mse /= n_train
        train_l2 /= n_train
        train_l2_data /= n_train
        train_l2_pde /= n_train
        test_l2 /= n_test

        train_mse_log[ep] = train_mse
        train_l2_log[ep] = train_l2
        test_l2_log[ep] = test_l2
        t2 = default_timer()
        if ep % step_size == 0:
            print(ep, t2 - t1, train_mse, train_l2, test_l2, train_l2_data, train_l2_pde)

    return train_mse_log, train_l2_log, test_l2_log


def train_vino_darcy(model, train_loader, test_loader, loss_func, optimizer,
                     scheduler, device, epochs, step_size, batch_size, data=False):
    model.train()
    n_train = len(train_loader)
    n_test = len(test_loader)
    train_l2_log = np.zeros(epochs)
    test_l2_log = np.zeros(epochs)
    for ep in range(epochs):
        t1 = default_timer()
        train_l2 = 0
        train_l2_pde = 0
        train_l2_data = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            mask = torch.ones_like(x[:, 1:-1, 1:-1, :])
            mask = F.pad(mask, (0, 0, 1, 1, 1, 1), mode='constant', value=0.5)
            mask[:, 0, 0, 0] = 0.25
            mask[:, 0, -1, 0] = 0.25
            mask[:, -1, 0, 0] = 0.25
            mask[:, -1, -1, 0] = 0.25
            optimizer.zero_grad()
            out = model(x)
            a = x
            num_x = x.shape[2]
            num_y = x.shape[1]
            du_dx = difference_x(out.permute(0, 3, 1, 2), device).permute(0, 2, 3, 1)
            du_dy = difference_y(out.permute(0, 3, 1, 2), device).permute(0, 2, 3, 1)

            loss_grad = 1 / 2 * torch.sum(
                2 / 3 * a[:, 0:-1, 0:-1, :] * du_dx[:, 0:-1, 0:-1, :] ** 2 +
                2 / 3 * a[:, 1:, 0:-1, :] * du_dx[:, 1:, 0:-1, :] ** 2 +
                a[:, 0:-1, 0:-1, :] * du_dy[:, 0:-1, 0:-1, :] ** 2 -
                1 / 3 * 1 / 2 * (a[:, 0:-1, 0:-1, :] + a[:, 1:, 0:-1, :]) * du_dx[:, 0:-1, 0:-1, :] * du_dx[:, 1:, 0:-1,
                                                                                                      :] -
                a[:, 0:-1, 0:-1, :] * du_dx[:, 0:-1, 0:-1, :] * du_dy[:, 0:-1, 0:-1, :] +
                1 / 2 * (a[:, 0:-1, 0:-1, :] + a[:, 1:, 0:-1, :]) * du_dx[:, 1:, 0:-1, :] * du_dy[:, 0:-1, 0:-1,
                                                                                            :]) / len(x)

            loss_int = torch.sum(out * mask) / len(x) / ((num_x - 1) * (num_y - 1))
            l2_pde = loss_grad - loss_int
            l2_data = loss_func(out.view(batch_size, -1), y.view(batch_size, -1)) / len(x)
            if data:
                l2 = l2_data + l2_pde
            else:
                l2 = l2_pde

            l2.backward()
            optimizer.step()
            train_l2 += l2.item()
            train_l2_data += l2_data.item()
            train_l2_pde += l2_pde.item()

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(train_l2)
        else:
            scheduler.step()
        model.eval()
        test_l2 = 0.0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                a = x
                du_dx = diff_x(out.permute(0, 3, 1, 2), device).permute(0, 2, 3, 1)
                du_dy = diff_y(out.permute(0, 3, 1, 2), device).permute(0, 2, 3, 1)
                f_pred_1 = diff_x((a * du_dx).permute(0, 3, 1, 2), device).permute(0, 2, 3, 1)
                f_pred_2 = diff_y((a * du_dy).permute(0, 3, 1, 2), device).permute(0, 2, 3, 1)
                f_pred = -(f_pred_1 + f_pred_2)
                f_ground = torch.ones_like(a)

                test_l2_pde = loss_func(f_ground.view(batch_size, -1), f_pred.reshape(batch_size, -1)).item() / len(x)
                test_l2_data = loss_func(out.view(batch_size, -1), y.view(batch_size, -1)).item() / len(x)
                if data:
                    test_l2 += test_l2_pde + test_l2_data
                else:
                    test_l2 += test_l2_pde

        train_l2 /= n_train
        train_l2_data /= n_train
        train_l2_pde /= n_train
        test_l2 /= n_test

        train_l2_log[ep] = train_l2
        test_l2_log[ep] = test_l2
        t2 = default_timer()
        if ep % step_size == 0:
            print(ep, t2 - t1, train_l2, test_l2, train_l2_data, train_l2_pde)

    return train_l2_log, test_l2_log


def train_vino_beam(model, train_loader, test_loader, loss_func, optimizer, scheduler,
                    device, epochs, batch_size):
    print("ep", "t2 - t1           ", "train_l2           ", "train_l2_data           ", "test_l2           ")
    model.train()
    nTrain = len(train_loader)
    nTest = len(test_loader)
    train_l2_log = np.zeros(epochs)
    train_l2_data_log = np.zeros(epochs)
    test_l2_log = np.zeros(epochs)

    # Add lists to store batch losses
    train_l2_batch_log = []
    test_l2_batch_log = []

    for ep in range(epochs):
        t1 = default_timer()
        train_l2 = 0
        train_l2_data = 0

        epoch_train_l2_batch = []  # Store batch losses for this epoch (training)
        epoch_test_l2_batch = []  # Store batch losses for this epoch (testing)

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            mask = torch.ones_like(x)
            optimizer.zero_grad()
            out = model(x)

            u_pred = out
            # u_pred = F.pad(out[:, 1:-1, :], (0, 0, 1, 1), mode='constant', value=0)
            # du_pred = du_FD(u_pred.permute(0, 2, 1), device).permute(0, 2, 1)
            # du2_pred = d2u_FD(u_pred.permute(0, 2, 1), device).permute(0, 2, 1)
            # du2_pred = F.pad(du2_pred, (0, 0, 1, 1), mode='replicate')

            # x = F.pad(x, (0, 0, 1, 1), mode='constant', value=0)
            mask = F.pad(mask[:, 1:-1, :], (0, 0, 1, 1), mode='constant', value=0.5)
            # integrand = 1/2*du_pred.view(batch_size, -1)**2-u_pred.view(batch_size, -1)*x.view(batch_size, -1)
            E = 1e1
            I = 1 / 12
            f = -x
            # integrand = E * I / 2 * du2_pred ** 2 - u_pred * f

            delta_x = 1 / (u_pred.shape[1] - 1)
            integral = (u_pred[:, 0:-2, :] - 2 * u_pred[:, 1:-1, :] + u_pred[:, 2:, :]) ** 2
            Loss_grad = E * I / 2 * torch.sum(integral) / (delta_x ** 3) / len(x)

            U_0 = u_pred[:, 0:-2, :]
            U_1 = u_pred[:, 1:-1, :]
            U_2 = u_pred[:, 2:, :]
            P_0 = f[:, 0:-2, :]
            P_1 = f[:, 1:-1, :]
            P_2 = f[:, 2:, :]

            Loss_int = (delta_x * (31 * P_0 * U_0 + 23 * P_0 * U_1 + 23 * P_1 * U_0 - 4 * P_0 * U_2 + 64 * P_1 * U_1 - 4 * P_2 * U_0 - 7 * P_1 * U_2 - 7 * P_2 * U_1 + P_2 * U_2)) / 120
            Loss_int = torch.sum(Loss_int) / len(x)
            # Loss_int = torch.sum(u_pred * f * mask) / len(x) * delta_x
            l2 = Loss_grad - Loss_int

            # u_pred = 1/2*(u_pred[:,:-1,:] + u_pred[:,1:,:])
            # l2 = torch.mean(integrand * mask) / len(x)
            l2_data = loss_func(out.view(batch_size, -1), y.view(batch_size, -1)) / len(x)
            l2.backward()  # use the l2 relative loss

            # Store the batch loss
            epoch_train_l2_batch.append(l2.item())

            optimizer.step()
            train_l2 += l2.item()
            train_l2_data += l2_data.item()
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(l2)
        else:
            scheduler.step()

        # Append the batch losses for the epoch
        train_l2_batch_log.append(epoch_train_l2_batch)

        model.eval()
        test_l2 = 0.0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                batch_l2 = loss_func(out.view(batch_size, -1), y.view(batch_size, -1)).item() / len(x)
                test_l2 += batch_l2
                # Store the batch loss
                epoch_test_l2_batch.append(batch_l2)

        # Append the test losses for the epoch
        test_l2_batch_log.append(epoch_test_l2_batch)

        train_l2 /= nTrain
        train_l2_data /= nTrain
        test_l2 /= nTest

        train_l2_log[ep] = train_l2
        train_l2_data_log[ep] = train_l2_data
        test_l2_log[ep] = test_l2
        t2 = default_timer()
        if ep % 50 == 0:
            print(ep, t2 - t1, train_l2, train_l2_data, test_l2)

    return train_l2_log, train_l2_data_log, test_l2_log, train_l2_batch_log, test_l2_batch_log


def train_pino_beam(model, train_loader, test_loader, loss_func, optimizer, scheduler,
                    device, epochs, batch_size):
    print("ep", "t2 - t1           ", "train_l2           ", "train_l2_data           ", "test_l2           ")
    model.train()
    nTrain = len(train_loader)
    nTest = len(test_loader)
    train_l2_log = np.zeros(epochs)
    train_l2_data_log = np.zeros(epochs)
    test_l2_log = np.zeros(epochs)
    for ep in range(epochs):
        t1 = default_timer()
        train_l2 = 0
        train_l2_data = 0
        for x, y in train_loader:
            cor = (torch.linspace(0, 1, x.shape[1]).to(device),)
            x, y = x.to(device), y.to(device)
            mask = torch.ones_like(x)
            optimizer.zero_grad()
            out = model(x)

            u_pred = out
            du1_pred = torch.gradient(u_pred, spacing=cor, axis=1)[0]
            du2_pred = torch.gradient(du1_pred, spacing=cor, axis=1)[0]
            du3_pred = torch.gradient(du2_pred, spacing=cor, axis=1)[0]
            du4_pred = torch.gradient(du3_pred, spacing=cor, axis=1)[0]
            f = x

            # mse_pde = F.mse_loss(f.view(batch_size, -1), du4_pred.view(batch_size, -1), reduction='mean') / len(x)
            l2_pde = loss_func(f[:, 4:-4, :].view(batch_size, -1), du4_pred[:, 4:-4, :].view(batch_size, -1)) / len(x)
            l2_data = loss_func(out.view(batch_size, -1), y.view(batch_size, -1)) / len(x)

            #l2 = l2_pde
            l2 = l2_data
            l2.backward()  # use the l2 relative loss

            optimizer.step()
            #train_l2 += l2.item()
            train_l2 += l2_pde.item()
            train_l2_data += l2_data.item()
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(l2)
        else:
            scheduler.step()
        model.eval()
        test_l2 = 0.0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                test_l2 += loss_func(out.view(batch_size, -1), y.view(batch_size, -1)).item() / len(x)

        train_l2 /= nTrain
        train_l2_data /= nTrain
        test_l2 /= nTest

        train_l2_log[ep] = train_l2
        train_l2_data_log[ep] = train_l2_data
        test_l2_log[ep] = test_l2
        t2 = default_timer()
        if ep % 50 == 0:
            print(ep, t2 - t1, train_l2, train_l2_data, test_l2)

    return train_l2_log, train_l2_data_log, test_l2_log
