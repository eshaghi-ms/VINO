#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 11:58:07 2023

@author: Mohammad Sadegh

Physics Informed Neural Operator example for Poisson 2D problem

Problem statement:
    Solve the equation -\Delta u = f(x, y) for x, y \in (0, 1) with Dirichlet boundary
    conditions u(0,y) = 0, u(x,0) = 0, u(1,y) = 0, u(x,1) = 0,
    where f(x,y) is a given function and u(x,y) is the unknown function   
"""
import numpy as np
import torch
import matplotlib.pyplot as plt
from timeit import default_timer
import time

from utils.generate_data_poisson import generate_inputs
from Comparative_Examples.utils.solvers import Poisson2D_solve
from utils.postprocessing import plot_pred2
from utils.fno_2d import FNO2d
from utils.fno_utils import count_params, LpLoss, train_pino_poisson

torch.manual_seed(42)
np.random.seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("this is the device name:")
print(device)
if torch.cuda.is_available():
    torch.cuda.set_device(0)
################################################################
#  configurations
################################################################
nTrain = 1000
nTest = 100
L = 1.
scaling = 100.  # scaling parameter related to the expected range of the inputs
length_scale_train = 0.1
length_scale_test = 0.1

s = 2 ** 4
batch_size = 100
learning_rate = 0.001

epochs = 500
step_size = 10
gamma = 0.5

modes = 8
width = 32

patience = 50

use_data = False

if s // 2 + 1 < modes:
    raise ValueError("Warning: modes should be bigger than (s//2+1)")

t_start = time.time()
#################################################################
# generate the data
#################################################################
F_train = torch.from_numpy(generate_inputs(nTrain, s, scaling=scaling,
                                           length_scale=length_scale_train)).float()
U_train = torch.from_numpy(Poisson2D_solve(F_train)).to(device).float()
F_train = F_train.reshape(nTrain, s, s, 1)
U_train = U_train.reshape(nTrain, s, s, 1)

F_test = torch.from_numpy(generate_inputs(nTest, s, scaling=scaling,
                                          length_scale=length_scale_test)).float()
U_test = torch.from_numpy(Poisson2D_solve(F_test)).to(device).float()
F_test = F_test.reshape(nTest, s, s, 1)
U_test = U_test.reshape(nTest, s, s, 1)

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(F_train, U_train),
                                           batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(F_test, U_test),
                                          batch_size=batch_size, shuffle=False)

# model
model = FNO2d(modes, modes, width).to(device)
n_params = count_params(model)
print(f'\nOur model has {n_params} parameters.')

t_data_gen = time.time()
print("Time taken for generation data is: ", t_data_gen - t_start)
print("epoch,      t2-t1,     train_mse,      train_l2,      test_l2")
################################################################
# training
################################################################
optimizer = torch.optim.Adam(model.parameters(),
                             lr=learning_rate)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=gamma, patience=patience)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=gamma)
myLoss = LpLoss(d=1, size_average=False)
t1 = default_timer()
train_mse_log, train_l2_log, test_l2_log = train_pino_poisson(model, train_loader, test_loader, myLoss, optimizer,
                                                              scheduler, device, epochs, step_size, batch_size, use_data)
print("Training time: ", default_timer() - t1)
# plot the convergence of the losses
plt.semilogy(train_mse_log, label='Train MSE')
plt.semilogy(train_l2_log, label='Train L2')
plt.semilogy(test_l2_log, label='Test L2')
plt.legend()
# plt.savefig( 'TrainingTrend.png', dpi=300)
plt.show()

test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(F_test,
                                                                         U_test), batch_size=1, shuffle=False)
pred = torch.zeros(U_test.squeeze().shape)
index = 0
x_test2 = torch.zeros(F_test.reshape(nTest, s, s).shape)
y_test2 = torch.zeros(U_test.shape)
test_l2_set = []

with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        x_test2[index] = x.reshape(1, s, s)
        y_test2[index] = y
        out = model(x).view(s, s)
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
print("Index of maximum error is", torch.argmax(test_l2_set).item())
################################################################
# evaluation
################################################################

# Plotting a random function from the test data generated by GRF
index = 77
x_test_plot = np.linspace(0., L, s).astype('float32')
y_test_plot = np.linspace(0., L, s).astype('float32')
x_plot_grid, y_plot_grid = np.meshgrid(x_test_plot, y_test_plot)

x_input = x_test2[index, :]

# fig_font = "DejaVu Serif"
# plt.rcParams["font.family"] = fig_font
# plt.figure()
# plt.contourf(x_test_plot, y_test_plot, x_input, levels=500, cmap='hsv')
# plt.colorbar()
# plt.gca().set_aspect('equal', adjustable='box')
# plt.title('Input Function')
# plt.show()

u_exact = y_test2[index, :].squeeze()
u_pred = pred[index, :]
plot_pred2(x_plot_grid, y_plot_grid, u_exact, u_pred, 'Test with GRF', 'Ex1')
