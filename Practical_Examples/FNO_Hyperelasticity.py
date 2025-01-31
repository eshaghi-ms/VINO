"""
Fourier Neural Operator for a 2D hyperelasticity problem for a 2D beam with
under different random tensions
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

import utils.config as cf
from utils.fno_2d import FNO2D
from utils.postprocessing import plot_field_2d
from utils.jax_tfp_loss import jax_tfp_function_factory
from utils.database_makers import Hyperelasticity_dataset
from utils.fno_utils import count_params, LpLoss, train_fno, model_evaluation, collate_fn

# jax.config.update("jax_enable_x64", True)
torch.manual_seed(42)
np.random.seed(42)


################################################################
#  configurations
################################################################
class FNO2d_HyperElasticity(FNO2D):
    model_data: dict = None

    def get_grid(self, _x):
        grid = super().get_grid(_x)
        gridy = grid[..., 0:1] * self.model_data['width']
        gridx = grid[..., 1:2] * self.model_data['length']
        return jnp.concatenate((gridy, gridx), -1)


model_data = cf.HyperElasticity
beam_length = model_data["plate_length"]
beam_width = model_data["plate_width"]
data_type = model_data["data_type"]  # float64
energy_type = model_data["FEM_data"]["energy_type"]
dim = model_data["FEM_data"]["dimension"]
model_data["nrg"] = random.PRNGKey(0)

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
dataset = Hyperelasticity_dataset(model_data)
train_dataset, test_dataset = random_split(dataset, [model_data["fno"]["n_train"], model_data["fno"]["n_test"]])
normalizers = [dataset.normalizer_x, dataset.normalizer_y] if model_data["normalized"] is True else None

X_train = []
Y_train = []
X_test = []
Y_test = []
for x, y in train_dataset:
    X_train.append(x)
    Y_train.append(y)
for x, y in test_dataset:
    X_test.append(x)
    Y_test.append(y)

X_train = jnp.array(X_train)
Y_train = jnp.array(Y_train)
X_test = jnp.array(X_test)
Y_test = jnp.array(Y_test)

# Making dataloaders
t_start = time.time()
train_loader = DataLoader(
    train_dataset,
    batch_size=model_data["fno"]["batch_size"],
    shuffle=True,
    collate_fn=collate_fn,
    drop_last=True
)
test_loader = DataLoader(
    test_dataset,
    batch_size=model_data["fno"]["batch_size"],
    shuffle=True,
    collate_fn=collate_fn
)

t_data_gen = time.time()
print("Time taken for generation data is: ", t_data_gen - t_start)
################################################################
# training
################################################################
# Initialize model
model = FNO2d_HyperElasticity(
    modes1=model_data["fno"]["mode1"],
    modes2=model_data["fno"]["mode2"],
    width=model_data["fno"]["width"],
    depth=model_data["fno"]["depth"],
    channels_last_proj=model_data["fno"]["channels_last_proj"],
    padding=model_data["fno"]["padding"],
    out_channels=2,
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
loss_func = jax_tfp_function_factory(model, model_params, loss_fn, X_train, Y_train)

initial_pos = loss_func.init_params_1d
tolerance = 1e-5
current_loss, _ = loss_func(initial_pos)
num_bfgs_iterations = 0
t2 = time.time()
results = lbfgs_minimize(loss_func, initial_position=initial_pos, parallel_iterations=1,
                         max_iterations=model_data["num_epoch_LBFGS"], num_correction_pairs=100, tolerance=1e-20)
t3 = time.time()
num_bfgs_iterations += results.num_iterations
print("Iteration: ", num_bfgs_iterations, " loss: ", results.objective_value)
_, unflatten_params = ravel_pytree(model_params)
model_params = unflatten_params(results.position)

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

index = 1
x_train, y_train = train_dataset[index]
x_train, y_train = jnp.expand_dims(x_train, axis=0), jnp.expand_dims(y_train, axis=0)
y_pred = model.apply(model_params, x_train)
if model_data["normalized"] is True:
    y_train = dataset.normalizer_y.decode(y_train)
    y_pred = dataset.normalizer_y.decode(y_pred)
# plot_field_2d(x_train[0, :, :, 0], beam_length, beam_width, "GRF-Random Input")
plot_field_2d(y_train[0, :, :, 0], beam_length, beam_width, "GRF-Exact Solution - U")
plot_field_2d(y_pred[0, :, :, 0], beam_length, beam_width, "GRF-Predicted Solution - U")
plot_field_2d(y_train[0, :, :, 0] - y_pred[0, :, :, 0], beam_length, beam_width, "GRF-Error - U")
plot_field_2d(y_train[0, :, :, 1], beam_length, beam_width, "GRF-Exact Solution - V")
plot_field_2d(y_pred[0, :, :, 1], beam_length, beam_width, "GRF-Predicted Solution - V")
plot_field_2d(y_train[0, :, :, 1] - y_pred[0, :, :, 1], beam_length, beam_width, "GRF-Error - V")

x_test, y_test = test_dataset[index]
x_test, y_test = jnp.expand_dims(x_test, axis=0), jnp.expand_dims(y_test, axis=0)
y_pred = model.apply(model_params, x_test)
if model_data["normalized"] is True:
    y_test = dataset.normalizer_y.decode(y_test)
    y_pred = dataset.normalizer_y.decode(y_pred)
plot_field_2d(y_test[0, :, :, 0], beam_length, beam_width, "GRF-Exact Solution - U")
plot_field_2d(y_pred[0, :, :, 0], beam_length, beam_width, "GRF-Predicted Solution - U")
plot_field_2d(y_test[0, :, :, 0] - y_pred[0, :, :, 0], beam_length, beam_width, "GRF-Error - U")
plot_field_2d(y_test[0, :, :, 1], beam_length, beam_width, "GRF-Exact Solution - V")
plot_field_2d(y_pred[0, :, :, 1], beam_length, beam_width, "GRF-Predicted Solution - V")
plot_field_2d(y_test[0, :, :, 1] - y_pred[0, :, :, 1], beam_length, beam_width, "GRF-Error - V")
