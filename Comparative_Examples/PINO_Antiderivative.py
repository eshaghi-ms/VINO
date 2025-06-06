#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Physics Informed Neural Operator example for Poisson 1D problem

Problem statement:
    Solve the equation -u''(x) = f(x) for x \in (0, 1) with Dirichlet boundary
    conditions u(0) = u(1) = 0
    where f(x) is a given function and u(x) is the unknown function    
"""

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from timeit import default_timer

from utils.generate_data_antiderivative import generate_inputs
from Comparative_Examples.utils.solvers import Poisson1D_solve
from utils.postprocessing import plot_pred
from utils.fno_1d import FNO1d
from utils.fno_utils import count_params, LpLoss, train_pino_antiderivative

torch.manual_seed(42)
np.random.seed(42)
device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

################################################################
#  configurations
################################################################
nTrain = 1000
nTest = 100
L = 1.
scaling = 100.  # scaling parameter related to the expected range of the inputs
length_scale_train = 0.1
length_scale_test = 0.1

s = 2 ** 8
batch_size = 20#200
learning_rate = 0.001

epochs = 500#2000
step_size = 50
gamma = 0.5

modes = 16
width = 64

use_data = False

if s // 2 + 1 < modes:
    raise ValueError("Warning: modes should be bigger than (s//2+1)")

#################################################################
# generate the data
#################################################################
F_train = torch.from_numpy(generate_inputs(nTrain, s, scaling=scaling,
                                           length_scale=length_scale_train)).float()
U_train = torch.from_numpy(Poisson1D_solve(F_train)).to(device).float()
F_train = F_train.reshape(nTrain, -1, 1)
U_train = U_train.reshape(nTrain, -1, 1)
F_test = torch.from_numpy(generate_inputs(nTest, s, scaling=scaling,
                                          length_scale=length_scale_test)).float()
U_test = torch.from_numpy(Poisson1D_solve(F_test)).to(device).float()
F_test = F_test.reshape(nTest, -1, 1)
U_test = U_test.reshape(nTest, -1, 1)

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(F_train, U_train),
                                           batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(F_test, U_test),
                                          batch_size=batch_size, shuffle=False)

# model
model = FNO1d(modes, width).to(device)
n_params = count_params(model)
print(f'\nOur model has {n_params} parameters.')

################################################################
# training and evaluation
################################################################
optimizer = torch.optim.Adam(model.parameters(),
                             lr=learning_rate)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=gamma, patience=100)
myLoss = LpLoss(d=1, size_average=False)
t1 = default_timer()
train_mse_log, train_l2_log, test_l2_log = train_pino_antiderivative(model, train_loader, test_loader, myLoss, optimizer,
                                                                     scheduler, device, epochs, batch_size, use_data)
print("Training time: ", default_timer() - t1)
# plot the convergence of the losses
plt.semilogy(train_mse_log, label='Train MSE')
plt.semilogy(train_l2_log, label='Train L2')
plt.semilogy(test_l2_log, label='Test L2')
plt.legend()
plt.show()

test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(F_test,
                                                                         U_test), batch_size=1, shuffle=False)
pred = torch.zeros(U_test.squeeze().shape)
index = 0
x_test2 = torch.zeros(F_test.reshape(nTest, s).shape)
y_test2 = torch.zeros(U_test.shape)
test_l2_set = []
with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        x_test2[index] = x.reshape(1, s)
        y_test2[index] = y
        out = model(x).view(-1)
        pred[index] = out
        test_l2 = myLoss(out.view(1, -1), y.view(1, -1)).item()
        test_l2_set.append(test_l2)
        print(index, test_l2)
        index = index + 1

test_l2_set = torch.tensor(test_l2_set)
test_l2_avg = torch.mean(test_l2_set)
test_l2_std = torch.std(test_l2_set)

print("The average testing error is", test_l2_avg.item())
print("Std. deviation of testing error is", test_l2_std.item())
print("Min testing error is", torch.min(test_l2_set).item())
print("Max testing error is", torch.max(test_l2_set).item())

# Plotting a random function from the test data generated by GRF
index = 90  # random.randrange(0, nTest)
x_test_plot = np.linspace(0., L, s + 2).astype('float32')
u_exact = F.pad(y_test2[index, :].squeeze(), (1, 1), mode='constant', value=0)
u_pred = F.pad(pred[index, :], (1, 1), mode='constant', value=0)
plot_pred(x_test_plot, u_exact, u_pred, 'GRF')

# testing y=sin(2*pi*x)
u_test_sin = 4 * np.pi ** 2 * np.sin(2 * np.pi * x_test_plot).astype('float32').reshape(1, s + 2, 1)
y_test_sin = np.sin(2 * np.pi * x_test_plot).astype('float32')
u_test_sin = torch.tensor(u_test_sin[:, 1:-1, :]).to(device)
pred_sin = model(u_test_sin).view(-1).cpu().detach().numpy()
pred_sin = np.concatenate(([0.], pred_sin, [0.]), axis=0)
plot_pred(x_test_plot, y_test_sin, pred_sin, 'sinusoidal')

# testing y=x**3-x
u_test_poly = -6 * x_test_plot
u_test_poly = u_test_poly.reshape(1, s + 2, 1)
y_test_poly = x_test_plot ** 3 - x_test_plot.astype('float32')
u_test_poly = torch.tensor(u_test_poly[:, 1:-1, :]).to(device)
pred_poly = model(u_test_poly).view(-1).cpu().detach().numpy()
pred_poly = np.concatenate(([0.], pred_poly, [0.]), axis=0)
plot_pred(x_test_plot, y_test_poly, pred_poly, 'polynomial')

# plotting y=(x-1)*(exp(x)-1)
u_test_exp = -np.exp(x_test_plot) * (x_test_plot + np.ones_like(x_test_plot))
u_test_exp = u_test_exp.reshape(1, s + 2, 1)
y_test_exp = (x_test_plot - np.ones_like(x_test_plot)) * (np.exp(x_test_plot) - np.ones_like(x_test_plot))
u_test_exp = torch.tensor(u_test_exp[:, 1:-1, :]).to(device)
pred_exp = model(u_test_exp).view(-1).cpu().detach().numpy()
pred_exp = np.concatenate(([0.], pred_exp, [0.]), axis=0)
plot_pred(x_test_plot, y_test_exp, pred_exp, 'exponential')