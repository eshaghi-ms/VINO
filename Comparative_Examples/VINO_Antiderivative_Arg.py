#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The script is written to be used in Hyperparameters_Check.py
Variational Physics-informed Neural Operator example for Second-order antiderivative

Problem statement:
    Solve the equation u''(x) + f(x) = 0 for x in (0, 1) with Dirichlet boundary
    conditions u(0) = u(1) = 0
    where f(x) is a given function and u(x) is the unknown function
"""

import argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from timeit import default_timer

from utils.generate_data_antiderivative import generate_inputs
from utils.solvers import Poisson1D_solve
from utils.postprocessing import plot_pred
from utils.fno_1d import FNO1d
from utils.fno_utils import count_params, LpLoss, train_vino_antiderivative


def main(args):
    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')

    nTrain = args.nTrain
    nTest = args.nTest
    scaling = args.scaling
    s = args.s
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    epochs = args.epochs
    gamma = args.gamma
    modes = args.modes
    width = args.width
    n_layer = args.n_layer
    L = 1.
    length_scale_train = 0.1
    length_scale_test = 0.1
    use_data = False

    if s // 2 + 1 < modes:
        raise ValueError("Warning: modes should be bigger than (s//2+1)")

    # Data generation
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

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(F_train, U_train),
        batch_size=batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(F_test, U_test),
        batch_size=batch_size, shuffle=False
    )

    # Model definition
    model = FNO1d(modes, width, n_layer).to(device)
    n_params = count_params(model)
    print(f'The model has {n_params} parameters.')

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=gamma, patience=100)
    myLoss = LpLoss(d=1, size_average=False)

    t1 = default_timer()
    train_l2_log, test_l2_log = train_vino_antiderivative(
        model, train_loader, test_loader, myLoss, optimizer, scheduler,
        device, epochs, batch_size, use_data
    )
    print("Training time: ", default_timer() - t1)

    # plt.plot(train_l2_log, label='Train L2')
    # plt.plot(test_l2_log, label='Test L2')
    # plt.legend()
    # plt.show()

    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(F_test, U_test), batch_size=1, shuffle=False)
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
            index += 1

    test_l2_set = torch.tensor(test_l2_set)
    test_l2_avg = torch.mean(test_l2_set)
    test_l2_std = torch.std(test_l2_set)

    print("The average testing error is", test_l2_avg.item())
    print("Std. deviation of testing error is", test_l2_std.item())
    print("Min testing error is", torch.min(test_l2_set).item())
    print("Max testing error is", torch.max(test_l2_set).item())

    return test_l2_set


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run FNO 1D Poisson Solver Experiment")
    parser.add_argument("--nTrain", type=int, default=1000, help="Number of training samples")
    parser.add_argument("--nTest", type=int, default=100, help="Number of testing samples")
    parser.add_argument("--scaling", type=float, default=100.0, help="Scaling parameter")
    parser.add_argument("--s", type=int, default=256, help="Resolution of the grid")
    parser.add_argument("--batch_size", type=int, default=20, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for optimizer")
    parser.add_argument("--epochs", type=int, default=1000, help="Number of training epochs")
    parser.add_argument("--gamma", type=float, default=0.5, help="Learning rate scheduler decay factor")
    parser.add_argument("--modes", type=int, default=16, help="Number of Fourier modes")
    parser.add_argument("--width", type=int, default=64, help="Width of the neural network layers")
    parser.add_argument("--n_layer", type=int, default=4, help="The number of Fourier layers")

    args = parser.parse_args()
    main(args)
