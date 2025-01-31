"""
Fourier Neural Operator for a 2D elasticity problem for an FGM beam with
random distribution of Elasticity modulus under different random tensions

Problem statement:
    \\Omega = (0,0.1)x(0,2)
    Fixed BC: x = 0 and x = 2
    Traction \\tau = GRF at y=0.1 in the vertical direction
"""

import time
import torch
import numpy as np
from jax import random
from jax import numpy as jnp
from matplotlib import pyplot as plt
from jax.flatten_util import ravel_pytree
from jax.example_libraries import optimizers
from torch.utils.data import DataLoader, random_split
from tensorflow_probability.substrates.jax.optimizer import lbfgs_minimize

from utils.fno_2d import FNO2D
from utils.postprocessing import plot_field_2d
from utils.database_makers import PlateHole_dataset
from utils.jax_tfp_loss import jax_tfp_function_factory
from utils.fno_utils import count_params, LpLoss, train_fno, model_evaluation, collate_fn

# jax.default_device = jax.devices("cuda")[1]
# jax.config.update("jax_enable_x64", True)
torch.manual_seed(42)
np.random.seed(42)


################################################################
#  configurations
################################################################
class FNO2d_PlateHole(FNO2D):
    model_data: dict = None

    def get_grid(self, _x):
        grid = super().get_grid(_x)
        gridy = grid[..., 0:1] * self.model_data['width']
        gridx = grid[..., 1:2] * self.model_data['length']
        return jnp.concatenate((gridy, gridx), -1)


plate_length = 5.
plate_width = 5.
num_pts_x = 200
num_pts_y = 200
# model_data dictionary
model_data = dict()
model_data["beam"] = dict()
model_data["beam"]["length"] = plate_length
model_data["beam"]["width"] = plate_width
model_data["beam"]["num_refinements"] = 6
model_data["beam"]["numPtsU"] = num_pts_x
model_data["beam"]["numPtsV"] = num_pts_y
model_data["beam"]["traction"] = jnp.ones_like(jnp.linspace(0, plate_width, num_pts_y))
model_data["beam"]["E"] = 100
model_data["beam"]["nu"] = 0.25
model_data["n_train"] = 200  # 1000
model_data["n_test"] = 20  # 100
model_data["n_data"] = model_data["n_train"] + model_data["n_test"]
model_data["n_dataset"] = 1100
model_data["batch_size"] = 50  # 100
model_data["batch_size_BFGS"] = 25  # 50
model_data["num_epoch"] = 300  # 1000
model_data["num_epoch_LBFGS"] = 300  # 1000
model_data["nrg"] = random.PRNGKey(0)
model_data["fno"] = dict()
model_data["fno"]["mode1"] = 8
model_data["fno"]["mode2"] = 8
model_data["fno"]["width"] = 32
model_data["fno"]["depth"] = 4
model_data["fno"]["channels_last_proj"] = 128
model_data["fno"]["padding"] = 0
model_data["fno"]["learning_rate"] = 0.001
model_data["fno"]["weight_decay"] = 1e-4
model_data["GRF"] = dict()
model_data["GRF"]["alpha"] = 4.
model_data["GRF"]["tau"] = 15.
model_data["normalized"] = False  # True
model_data["dir"] = "./data/"
model_data["path"] = (model_data["dir"] +
                      'PlateHole_LxW_' + str(plate_length) + "x" + str(plate_width) +
                      '_s' + str(model_data["beam"]["numPtsU"]) + "x" + str(model_data["beam"]["numPtsV"]) +
                      '_n' + str(model_data["n_dataset"]) + '.npz')

# Checking that the modes are not more than the input size
assert model_data["fno"]["mode1"] <= model_data["beam"]["numPtsU"] // 2 + 1
assert model_data["fno"]["mode2"] <= model_data["beam"]["numPtsV"] // 2 + 1
assert model_data["beam"]["numPtsU"] % 2 == 0  # Only tested for even-sized inputs
assert model_data["beam"]["numPtsV"] % 2 == 0  # Only tested for even-sized inputs

#################################################################
# generate the data
#################################################################
# Loading and splitting dataset
t_start = time.time()
dataset = PlateHole_dataset(model_data)
train_dataset, test_dataset, rest_dataset = random_split(dataset, [model_data["n_train"], model_data["n_test"], model_data["n_dataset"] - model_data["n_data"]])
normalizers = [dataset.normalizer_x, dataset.normalizer_y] if model_data["normalized"] is True else None

X_train = []
Y_train = []
X_test = []
Y_test = []
for x, y in train_dataset:
    X_train.append(x)
    # plt.plot(y[:, 100, 0])
    # plt.show()
    Y_train.append(y)
for x, y in test_dataset:
    X_test.append(x)
    Y_test.append(y)

X_train = jnp.array(X_train)
Y_train = jnp.array(Y_train)
X_test = jnp.array(X_test)
Y_test = jnp.array(Y_test)

# Making dataloaders
train_loader = DataLoader(
    train_dataset,
    batch_size=model_data["batch_size"],
    shuffle=True,
    collate_fn=collate_fn,
    drop_last=True
)
test_loader = DataLoader(
    test_dataset,
    batch_size=model_data["batch_size"],
    shuffle=True,
    collate_fn=collate_fn
)

t_data_gen = time.time()
print("Time taken for generation data is: ", t_data_gen - t_start)
################################################################
# training
################################################################
# Initialize model
model = FNO2d_PlateHole(
    modes1=model_data["fno"]["mode1"],
    modes2=model_data["fno"]["mode2"],
    width=model_data["fno"]["width"],
    depth=model_data["fno"]["depth"],
    channels_last_proj=model_data["fno"]["channels_last_proj"],
    padding=model_data["fno"]["padding"],
    out_channels=Y_train.shape[-1],
    model_data=model_data["beam"]
)
_x, _ = train_dataset[0]
_x = jnp.expand_dims(_x, axis=0)
_, model_params = model.init_with_output(model_data["nrg"], _x)
del _x

n_params = count_params(model_params)
print(f'\nOur model has {n_params} parameters.')

# Initialize optimizers
init_fun, update_fun, get_params = optimizers.adam(model_data["fno"]["learning_rate"])
opt_state = init_fun(model_params)

# Define loss function
loss_fn = LpLoss(model, model_data, d=1, p=1, size_average=False)

print("Training (ADAM)...")
t0 = time.time()
model_params, train_losses, test_losses = train_fno(model, train_loader, test_loader, loss_fn,
                                                    get_params, update_fun, opt_state)
t1 = time.time()
print("Training (ADAM) time: ", t1 - t0)

print("Training (TFP-BFGS)...")
train_op2 = "TFP-BFGS"

t2 = time.time()
num_bfgs_iterations = 0

train_loader = DataLoader(
    train_dataset,
    batch_size=model_data["batch_size_BFGS"],
    shuffle=True,
    collate_fn=collate_fn,
    drop_last=True
)

for X_batch, Y_batch in train_loader:
    # Update the loss function for the current batch
    loss_func = jax_tfp_function_factory(model, model_params, loss_fn, X_batch, Y_batch)

    initial_pos = loss_func.init_params_1d
    current_loss, _ = loss_func(initial_pos)

    results = lbfgs_minimize(loss_func, initial_position=initial_pos, parallel_iterations=1,
                             max_iterations=model_data["num_epoch_LBFGS"], num_correction_pairs=100, tolerance=1e-09)

    num_bfgs_iterations += results.num_iterations
    print(f"Iteration: {num_bfgs_iterations}, Batch Loss: {results.objective_value}")
    train_losses.append(results.objective_value)
    # Update model parameters with the results from L-BFGS
    _, unflatten_params = ravel_pytree(model_params)
    model_params = unflatten_params(results.position)

t3 = time.time()
print("Total iterations: ", num_bfgs_iterations)
print("Time taken (BFGS)", t3 - t2, "seconds")
print("Time taken (all)", t3 - t_data_gen, "seconds")

# plot the convergence of the losses
plt.semilogy(train_losses, label='Train L2')
plt.semilogy(test_losses, label='Test L2')
plt.legend()
plt.show()
################################################################
# evaluation
################################################################
test_loader = DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=False,
    collate_fn=collate_fn
)

model_evaluation(model, model_data, model_params, loss_fn, test_loader, normalizers)

index = 2
x_train, y_train = train_dataset[index]
x_train, y_train = jnp.expand_dims(x_train, axis=0), jnp.expand_dims(y_train, axis=0)
y_pred = model.apply(model_params, x_train)
if model_data["normalized"] is True:
    x_train = dataset.normalizer_x.decode(x_train)
    x_train = np.where(np.array(x_train) < 0.5, False, True)
    y_train = dataset.normalizer_y.decode(y_train)
    y_pred = dataset.normalizer_y.decode(y_pred)
field_names = 0
if y_train.shape[3] == 2:
    field_names = ['disp_x', 'disp_y']
if y_train.shape[3] == 3:
    field_names = ['stress_xx', 'stress_yy', 'stress_xy']

for field_index, field_name in enumerate(field_names):
    plot_field_2d(y_train[index, :, :, field_index], plate_length, plate_width, "Train - Exact - " + field_name, mask=x_train)
    plot_field_2d(y_pred[index, :, :, field_index], plate_length, plate_width, "Train - Predicted - " + field_name, mask=x_train)

x_test, y_test = test_dataset[index]
x_test, y_test = jnp.expand_dims(x_test, axis=0), jnp.expand_dims(y_test, axis=0)
y_pred = model.apply(model_params, x_test)
if model_data["normalized"] is True:
    x_test = dataset.normalizer_x.decode(x_test)
    x_test = np.where(np.array(x_test) < 0.5, False, True)
    y_test = dataset.normalizer_y.decode(y_test)
    y_pred = dataset.normalizer_y.decode(y_pred)

for field_index, field_name in enumerate(field_names):
    plot_field_2d(y_test[index, :, :, field_index], plate_length, plate_width, "Test - Exact - " + field_name, mask=x_test)
    plot_field_2d(y_pred[index, :, :, field_index], plate_length, plate_width, "Test - Predicted - " + field_name, mask=x_test)
