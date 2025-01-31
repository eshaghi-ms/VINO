#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Variational Physics-informed Neural Operator example for Forth-order equation

Problem statement:
    Solve the equation k u''''(x) = f(x) for x in (0, 1) with Dirichlet boundary
    conditions u(0) = u(1) = u'(0) = u'(1) = 0,
    where f(x) is a given function and u(x) is the unknown function
"""
import os
import numpy as np
import torch

import matplotlib.pyplot as plt
# import random
from timeit import default_timer

from utils.generate_data_antiderivative import generate_inputs
from utils.solvers import Beam1D_solve
from utils.postprocessing import plot_pred
from utils.fno_1d import FNO1d
from utils.fno_utils import count_params, LpLoss, train_vino_beam

torch.manual_seed(42)
np.random.seed(42)
device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')


class FNO_Beam1D(FNO1d):
    def forward(self, x):
        x = super().forward(x)
        x[:, 0, :] = 0
        x[:, -1, :] = 0
        return x


################################################################
#  configurations
################################################################
nTrain = 1000
nTest = 100
b = h = L = 1.
E = 1e1
I = 1 / 12 * b * h ** 3  # Moment of inertia (m^4)

scaling = 100.  # scaling parameter related to the expected range of the inputs
length_scale_train = 0.1
length_scale_test = 0.1

s = 2 ** 6
batch_size = 10
learning_rate = 0.001

epochs = 2000
gamma = 0.5

modes = 32
width = 64

if s // 2 + 1 < modes:
    raise ValueError("Warning: modes should be bigger than (s//2+1)")

#################################################################
# generate the data
#################################################################
F_train = torch.from_numpy(generate_inputs(nTrain, s, scaling=scaling,
                                           length_scale=length_scale_train)).float()
U_train = torch.from_numpy(Beam1D_solve(F_train)).to(device).float()  #  Beam
F_train = F_train.reshape(nTrain, -1, 1)
U_train = U_train.reshape(nTrain, -1, 1)
F_test = torch.from_numpy(generate_inputs(nTest, s, scaling=scaling,
                                          length_scale=length_scale_test)).float()

U_test = torch.from_numpy(Beam1D_solve(F_test)).to(device).float()
F_test = F_test.reshape(nTest, -1, 1)
U_test = U_test.reshape(nTest, -1, 1)

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(F_train, U_train),
                                           batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(F_test, U_test),
                                          batch_size=batch_size, shuffle=False)

# model
model = FNO_Beam1D(modes, width).to(device)
n_params = count_params(model)
print(f'the model has {n_params} parameters.')

################################################################
# training and evaluation
################################################################
optimizer = torch.optim.Adam(model.parameters(),
                             lr=learning_rate)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=gamma, patience=100)
myLoss = LpLoss(d=1, size_average=False)

t1 = default_timer()
train_l2_log, train_l2_data_log, test_l2_log, train_l2_batch_log, test_l2_batch_log = (
    train_vino_beam(model, train_loader, test_loader, myLoss, optimizer, scheduler, device, epochs, batch_size))
print("Training time: ", default_timer() - t1)

epochs_range = np.arange(1, epochs + 1)
# Prepare training loss bands
train_l2_batch_min = [min(epoch) for epoch in train_l2_batch_log]
train_l2_batch_max = [max(epoch) for epoch in train_l2_batch_log]
# Prepare test loss bands
test_l2_batch_min = [min(epoch) for epoch in test_l2_batch_log]
test_l2_batch_max = [max(epoch) for epoch in test_l2_batch_log]

folder = "plots"
fig_font = "DejaVu Serif"
plt.rcParams["font.family"] = fig_font
plt.rcParams.update({'font.size': 12})
os.makedirs(folder, exist_ok=True)

plt.figure(figsize=(10, 6))
# Plot training loss band
plt.fill_between(epochs_range[1:], train_l2_batch_min[1:], train_l2_batch_max[1:],
                 color='blue', alpha=0.2, label='Training Loss Band')
plt.plot(epochs_range[1:], train_l2_log[1:], color='blue', label='Average Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss Trend')
plt.legend(loc='best')
plt.grid(alpha=0.5)
plt.ylim((-40, 20))
full_name = folder + '/' + "TrainingLoss"
plt.savefig(full_name + '.png', dpi=600, bbox_inches='tight')
plt.show()


plt.figure(figsize=(10, 6))
# Plot testing loss band
plt.fill_between(epochs_range[1:], test_l2_batch_min[1:], test_l2_batch_max[1:],
                 color='orange', alpha=0.2, label='Testing Loss Band')
plt.plot(epochs_range[1:], test_l2_log[1:], color='orange', label='Average Testing Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Test Dataset Error')
plt.legend(loc='best')
plt.grid(alpha=0.5)
plt.ylim((0, 1.1))
full_name = folder + '/' + "TestErrorTrend"
plt.savefig(full_name + '.png', dpi=600, bbox_inches='tight')
plt.show()

# plot the convergence of the losses
# plt.plot(train_l2_log, label='Train L2')
# plt.plot(train_l2_data_log, label='Train L2 Data')
# plt.plot(test_l2_log, label='Test L2')
# plt.legend()
# plt.show()

# plt.figure()
# plt.plot(test_l2_log)
# plt.legend()
# plt.title("Test error trend during training")
# os.makedirs(folder, exist_ok=True)
# full_name = folder + '/' + "ErrorTrend"
# plt.savefig(full_name + '.png', dpi=600, bbox_inches='tight')
# plt.show()

# plt.figure()
# plt.plot(train_l2_log[1:])
# plt.legend()
# plt.title("Loss function convergence")
# os.makedirs(folder, exist_ok=True)
# full_name = folder + '/' + "ErrorTrend"
# plt.savefig(full_name + '.png', dpi=600, bbox_inches='tight')
# plt.show()

test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(F_test, U_test), batch_size=1, shuffle=False)
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
test_l2_min, min_idx = torch.min(test_l2_set), torch.argmin(test_l2_set)
test_l2_max, max_idx = torch.max(test_l2_set), torch.argmax(test_l2_set)
test_l2_mode, mode_count = torch.mode(test_l2_set)
mode_indices = torch.nonzero(test_l2_set == test_l2_mode).squeeze().tolist()

print("The average testing error is", test_l2_avg.item())
print("Std. deviation of testing error is", test_l2_std.item())
print("Min testing error is", torch.min(test_l2_set).item())
print("Max testing error is", torch.max(test_l2_set).item())
print("Min testing error is", test_l2_min.item(), "at index", min_idx.item())
print("Max testing error is", test_l2_max.item(), "at index", max_idx.item())
print("Mode of testing errors is", test_l2_mode.item(), "appearing", mode_count.item(),
      "times at indices", mode_indices)

# Plotting a random function from the test data generated by GRF
index = 90  # random.randrange(0, nTest)
x_test_plot = np.linspace(0, L, s).astype('float32')
u_exact = y_test2[index, :]
u_pred = pred[index, :]
plot_pred(x_test_plot, u_exact, u_pred, 'GRF_90')

index = 41  # random.randrange(0, nTest)
x_test_plot = np.linspace(0, L, s).astype('float32')
u_exact = y_test2[index, :]
u_pred = pred[index, :]
plot_pred(x_test_plot, u_exact, u_pred, 'GRF_41')

index = 98  # random.randrange(0, nTest)
x_test_plot = np.linspace(0, L, s).astype('float32')
u_exact = y_test2[index, :]
u_pred = pred[index, :]
plot_pred(x_test_plot, u_exact, u_pred, 'GRF_98')

index = 42  # random.randrange(0, nTest)
x_test_plot = np.linspace(0, L, s).astype('float32')
u_exact = y_test2[index, :]
u_pred = pred[index, :]
plot_pred(x_test_plot, u_exact, u_pred, 'GRF_42')

index = 43  # random.randrange(0, nTest)
x_test_plot = np.linspace(0, L, s).astype('float32')
u_exact = y_test2[index, :]
u_pred = pred[index, :]
plot_pred(x_test_plot, u_exact, u_pred, 'GRF_43')

index = 44  # random.randrange(0, nTest)
x_test_plot = np.linspace(0, L, s).astype('float32')
u_exact = y_test2[index, :]
u_pred = pred[index, :]
plot_pred(x_test_plot, u_exact, u_pred, 'GRF_44')

index = 45  # random.randrange(0, nTest)
x_test_plot = np.linspace(0, L, s).astype('float32')
u_exact = y_test2[index, :]
u_pred = pred[index, :]
plot_pred(x_test_plot, u_exact, u_pred, 'GRF_45')

# Testing for a uniform load p(x) = constant
# w = 20.0  # distributed load
# u_test_uniform = np.full_like(x_test_plot, w).reshape(1, s, 1)  # Uniformly distributed load
# y_test_uniform = -((w / (24 * E * I)) * (L ** 3 * x_test_plot - 2 * L * x_test_plot ** 3 + x_test_plot ** 4)).astype('float32')
# u_test_uniform = torch.tensor(u_test_uniform).to(device)
# pred_uniform = model(u_test_uniform).view(-1).cpu().detach().numpy()
# plot_pred(x_test_plot, y_test_uniform, pred_uniform, 'uniform_load')
