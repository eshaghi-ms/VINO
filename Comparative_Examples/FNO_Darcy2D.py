#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fourier Neural Operator example for Darcy 2D problem

Problem statement:
    Solve the equation \nabla \\cdot (a(x, y) \\nabla u(x, y)) = f(x, y) for x, y \\in (0, 1) with Dirichlet boundary
    conditions u(0,y) = 0, u(x,0) = 0, 
    where f(x,y) is the forcing function and kept fixed f(x) = 1
    and a(x,y) is diffusion coefficient 
    The network learns the operator mapping the diffusion coefficient to the solution (a ↦ u)
  
"""
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from timeit import default_timer
import time
import os

from utils.generate_data_darcy import generate_inputs
from Comparative_Examples.utils.solvers import Darcy2D_solve
from utils.postprocessing import plot_pred1, plot_pred2
from utils.fno_2d import FNO2d
from utils.fno_utils import count_params, LpLoss, train_fno
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
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
class FNO2dSpecific(FNO2d):

    def forward(self, _x, **kwargs):
        grid = self.get_grid(_x.shape, _x.device)
        # x_min = -100
        # x_max = 100
        # x = (2 * x - x_max - x_min) / (x_max - x_min)

        _x = torch.cat((_x, grid), dim=-1)
        _x = self.fc0(_x)
        _x = _x.permute(0, 3, 1, 2)
        # x = F.pad(x, [0, self.padding])  # pad the domain if input is non-periodic

        x1 = self.conv0(_x)
        x1 = self.mlp0(x1)
        x2 = self.w0(_x)
        _x = x1 + x2
        _x = F.gelu(_x)

        x1 = self.conv1(_x)
        x1 = self.mlp1(x1)
        x2 = self.w1(_x)
        _x = x1 + x2
        _x = F.gelu(_x)

        x1 = self.conv2(_x)
        x1 = self.mlp2(x1)
        x2 = self.w2(_x)
        _x = x1 + x2
        _x = F.gelu(_x)

        x1 = self.conv3(_x)
        x1 = self.mlp3(x1)
        x2 = self.w3(_x)
        _x = x1 + x2

        _x = _x.permute(0, 2, 3, 1)
        _x = self.fc1(_x)
        _x = F.gelu(_x)
        _x = self.fc2(_x)
        return _x


nTrain = 1000
nTest = 100
L = 1.0

s = 2 ** 6
batch_size = 100
learning_rate = 0.001

epochs = 500
step_size = 10
gamma = 0.5

modes = 12
width = 32

patience = 50

if s // 2 + 1 < modes:
    raise ValueError("Warning: modes should be bigger than (s//2+1)")

t_start = time.time()
#################################################################
# generate the data
#################################################################

alpha = 2.
tau = 3.

dataset_name = 'darcy2d'
parent_dir = './data/'
dataset_filename = parent_dir + dataset_name + '_s' + str(s) + '.pt'

if not os.path.exists(parent_dir):
    os.makedirs(parent_dir)

if os.path.exists(dataset_filename):
    print("Found saved dataset at", dataset_filename)
    loaded_data = torch.load(dataset_filename)
    F_train = loaded_data['F_train']
    U_train = loaded_data['U_train']
    F_test = loaded_data['F_test']
    U_test = loaded_data['U_test']
else:
    F_train = torch.from_numpy(generate_inputs(nTrain, s, alpha, tau)).float()
    U_train = torch.from_numpy(Darcy2D_solve(F_train)).to(device).float()
    F_train = F_train.reshape(nTrain, s, s, 1)
    U_train = U_train.reshape(nTrain, s, s, 1)

    F_test = torch.from_numpy(generate_inputs(nTest, s, alpha, tau)).float()
    U_test = torch.from_numpy(Darcy2D_solve(F_test)).to(device).float()
    F_test = F_test.reshape(nTest, s, s, 1)
    U_test = U_test.reshape(nTest, s, s, 1)
    torch.save({'F_train': F_train, 'U_train': U_train,
                'F_test': F_test, 'U_test': U_test}, dataset_filename)

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(F_train, U_train),
                                           batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(F_test, U_test),
                                          batch_size=batch_size, shuffle=False)

# model
model = FNO2dSpecific(modes, modes, width).to(device)
n_params = count_params(model)
print(f'\nOur model has {n_params} parameters.')

t_data_gen = time.time()
print("Time taken for generation data is: ", t_data_gen - t_start)
################################################################
# training
################################################################
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=gamma)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=gamma, patience=patience)
myLoss = LpLoss(d=1, size_average=False)
t1 = default_timer()
train_mse_log, train_l2_log, test_l2_log = train_fno(model, train_loader,
                                                     test_loader, myLoss, optimizer,
                                                     scheduler, device, epochs, batch_size)
print("Training time: ", default_timer() - t1)
# plot the convergence of the losses
plt.semilogy(train_mse_log, label='Train MSE')
plt.semilogy(train_l2_log, label='Train L2')
plt.semilogy(test_l2_log, label='Test L2')
plt.legend()
# plt.savefig('TrainingTrend.png', dpi=300)
plt.show()


train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(F_train, U_train), batch_size=1, shuffle=False)
pred = torch.zeros(U_train.squeeze().shape)
index = 0
x_train2 = torch.zeros(F_train.reshape(nTrain, s, s).shape)
y_train2 = torch.zeros(U_train.shape)
train_l2_set = []

with torch.no_grad():
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        x_train2[index] = x.reshape(1, s, s)
        y_train2[index] = y
        out = model(x).view(s, s)
        pred[index] = out
        train_l2 = myLoss(out.view(1, -1), y.view(1, -1)).item()
        train_l2_set.append(train_l2)
        print(index, train_l2)
        index = index + 1

train_l2_set = torch.tensor(train_l2_set)
train_l2_avg = torch.mean(train_l2_set)
train_l2_std = torch.std(train_l2_set)
train_l2_argmax = torch.argmax(train_l2_set).item()

print("The average training error is", train_l2_avg.item())
print("Std. deviation of training error is", train_l2_std.item())
print("Min training error is", torch.min(train_l2_set).item())
print("Max training error is", torch.max(train_l2_set).item())
print("Index of maximum error is", train_l2_argmax)

test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(F_test, U_test), batch_size=1, shuffle=False)
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
test_l2_argmax = torch.argmax(test_l2_set).item()

print("The average testing error is", test_l2_avg.item())
print("Std. deviation of testing error is", test_l2_std.item())
print("Min testing error is", torch.min(test_l2_set).item())
print("Max testing error is", torch.max(test_l2_set).item())
print("Index of maximum error is", test_l2_argmax)

################################################################
# evaluation
################################################################

# Plotting a random function from the test data generated by GRF
index = 19  # random.randrange(0, nTest)

x_test_plot = np.linspace(0., L, s).astype('float32')
y_test_plot = np.linspace(0., L, s).astype('float32')
x_plot_grid, y_plot_grid = np.meshgrid(x_test_plot, y_test_plot)

coefficient = x_test2[index, :].squeeze()
u_exact = y_test2[index, :].squeeze()
u_pred = pred[index, :]
plot_pred1(x_plot_grid, y_plot_grid, coefficient, 'GRF')
plot_pred2(x_plot_grid, y_plot_grid, u_exact, u_pred, 'Test with GRF', 'Ex1')

# synthetic test
# coefficient = torch.where(torch.from_numpy(x_plot_grid <= 0.5), 4., 12.)
# coefficients = torch.unsqueeze(coefficient, dim=0)
# u_comp = Darcy2D_solve(coefficients).squeeze()
# u_pred = model(torch.unsqueeze(coefficients.to(device), dim=-1)).squeeze().cpu().detach().numpy()
# plot_pred1(x_plot_grid, y_plot_grid, coefficient, 'half plate example')
# plot_pred2(x_plot_grid, y_plot_grid, u_comp, u_pred, 'Half plate', 'Ex2')
