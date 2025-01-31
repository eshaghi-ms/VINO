import os
import time
import torch
import numpy as np
from jax import random
from jax import numpy as jnp
from matplotlib import pyplot as plt
from jax.example_libraries import optimizers
from torch.utils.data import DataLoader, random_split
# from tensorflow_probability.substrates.jax.optimizer import lbfgs_minimize

from utils.fno_2d import FNO2D
from utils.database_makers import DarcyDataset
from utils.postprocessing import plot_field_2d
# from .utils.jax_tfp_loss import jax_tfp_function_factory
from utils.fno_utils import count_params, train_fno, model_evaluation, collate_fn, VinoDarcyLoss


class FNO2d_darcy(FNO2D):
    def __call__(self, x_: jnp.ndarray) -> jnp.ndarray:
        x_ = super().__call__(x_)
        # apply Neumann boundary condition
        x_ = x_.at[:, :, 0, :].set(0)
        x_ = x_.at[:, :, -1, :].set(0)
        x_ = x_.at[:, 0, :, :].set(0)
        x_ = x_.at[:, -1, :, :].set(0)

        # grid = self.get_grid(x)
        # m = grid[..., 0:1] * grid[..., 1:2] * (grid[..., 0:1] - 1) * (grid[..., 1:2] - 1)
        # x *= m
        return x_


# jax.config.update("jax_enable_x64", True)
torch.manual_seed(42)
np.random.seed(42)
################################################################
#  configurations
################################################################
# Settings dictionary
model_data = dict()
model_data["n_train"] = 1000
model_data["n_test"] = 100
model_data["n_data"] = model_data["n_train"] + model_data["n_test"]
model_data["batch_size"] = 100
model_data["learning_rate"] = 0.001
model_data["num_epoch"] = 500
model_data["num_epoch_LBFGS"] = 100
model_data["nrg"] = random.PRNGKey(0)
model_data["fno"] = dict()
model_data["fno"]["weight_decay"] = 1e-4
model_data["fno"]["modes"] = 12
model_data["fno"]["width"] = 32
model_data["fno"]["depth"] = 4
model_data["fno"]["channels_last_proj"] = 128
model_data["fno"]["padding"] = 0
model_data["grid_point_num"] = 2 ** 6
model_data["GRF"] = dict()
model_data["GRF"]["alpha"] = 2.
model_data["GRF"]["tau"] = 3.
model_data["normalized"] = True
model_data["dir"] = os.path.abspath(os.path.join(os.path.dirname(__file__), './data/'))
model_data["path"] = os.path.join(model_data["dir"],
                                  'darcy2d_s' + str(model_data["grid_point_num"]) +
                                  '_n' + str(model_data["n_data"]) + '.npz')
print(model_data["path"])

assert model_data["grid_point_num"] // 2 + 1 >= model_data["fno"]["modes"]
#################################################################
# generate the data
#################################################################
# Loading and splitting dataset
dataset = DarcyDataset(model_data)
train_dataset, test_dataset = random_split(
    dataset,
    [model_data["n_train"], model_data["n_test"]]
)
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
model = FNO2d_darcy(
    modes1=model_data["fno"]["modes"],
    modes2=model_data["fno"]["modes"],
    width=model_data["fno"]["width"],
    depth=model_data["fno"]["depth"],
    channels_last_proj=model_data["fno"]["channels_last_proj"],
    padding=model_data["fno"]["padding"],
)
_x, _ = train_dataset[0]
_x = jnp.expand_dims(_x, axis=0)
_, model_params = model.init_with_output(model_data["nrg"], _x)
del _x

n_params = count_params(model_params)
print(f'\nOur model has {n_params} parameters.')

# Initialize optimizers
init_fun, update_fun, get_params = optimizers.adam(model_data["learning_rate"])
opt_state = init_fun(model_params)

# Define loss function
loss_fn = VinoDarcyLoss(model, model_data, normalizers, d=1, p=1, size_average=False)

print("Training (ADAM)...")
t0 = time.time()
model_params, train_losses, test_losses = train_fno(model, train_loader, test_loader, loss_fn,
                                                    get_params, update_fun, opt_state)
t1 = time.time()
print("Training (ADAM) time: ", t1 - t0)
"""
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
_, unflatten_params = jax.flatten_util.ravel_pytree(model_params)
model_params = unflatten_params(results.position)

print("Time taken (BFGS)", t3 - t2, "seconds")
print("Time taken (all)", t3 - t_data_gen, "seconds")
"""
# plot the convergence of the losses
plt.plot(train_losses, label='Train L2')
plt.plot(test_losses, label='Test L2')
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

index = 10
x_train, y_train = train_dataset[index]
x_train, y_train = jnp.expand_dims(x_train, axis=0), jnp.expand_dims(y_train, axis=0)
y_pred = model.apply(model_params, x_train)
if model_data["normalized"] is True:
    y_train = dataset.normalizer_y.decode(y_train)
    y_pred = dataset.normalizer_y.decode(y_pred)
plot_field_2d(x_train[0, :, :, 0], 1, 1, "GRF-Random Input")
plot_field_2d(y_train[0, :, :, 0], 1, 1, "GRF-Exact Solution")
plot_field_2d(y_pred[0, :, :, 0], 1, 1, "GRF-Predicted Solution")
plot_field_2d(y_train[0, :, :, 0] - y_pred[0, :, :, 0], 1, 1, "GRF-Error")

x_test, y_test = test_dataset[index]
x_test, y_test = jnp.expand_dims(x_test, axis=0), jnp.expand_dims(y_test, axis=0)
y_pred = model.apply(model_params, x_test)
if model_data["normalized"] is True:
    y_test = dataset.normalizer_y.decode(y_test)
    y_pred = dataset.normalizer_y.decode(y_pred)
plot_field_2d(x_test[0, :, :, 0], 1, 1, "GRF-Random Input")
plot_field_2d(y_test[0, :, :, 0], 1, 1, "GRF-Exact Solution")
plot_field_2d(y_pred[0, :, :, 0], 1, 1, "GRF-Predicted Solution")
plot_field_2d(y_test[0, :, :, 0] - y_pred[0, :, :, 0], 1, 1, "GRF-Error")
