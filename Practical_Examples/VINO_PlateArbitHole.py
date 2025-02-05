"""
Energy-based Physics-Informed Fourier Neural Operator for a 2D elasticity problem for a plate with random voids

Problem statement:
    \\Omega = (0,5)x(0,5)
    Fixed BC: x = 0
    Traction \\tau = 1 at y=5 in the horizontal direction

Note: To achieve the result as same as the paper, use the default hyperparameters and run the code, 
and then, change num_epoch = 2000 and learning_rate = 0.001 and run the code to continue the training process. 
The other option is using the scheduled Adam optimizer. 
"""
import os
# import sys
import jax
import time
import torch
import pickle
# import optax
import numpy as np
from tqdm import tqdm
from jax import random
from jax import numpy as jnp
from matplotlib import pyplot as plt
# from jax.flatten_util import ravel_pytree
from jax.example_libraries import optimizers
from torch.utils.data import DataLoader, random_split
# from tensorflow_probability.substrates.jax.optimizer import lbfgs_minimize

from utils.fno_2d import FNO2D
# from utils.fno_utils import train_fno_scheduled
from utils.database_makers import PlateHole_dataset
# from utils.jax_tfp_loss import jax_tfp_function_factory
from utils.postprocessing import plot_field_2d, plot_fields_2d
from utils.fno_utils import (count_params, VinoPlateHoleLoss,
                             train_fno, model_evaluation, collate_fn)

os.environ['CUDA_VISIBLE_DEVICES'] = '3'
# jax.default_device = jax.devices("cpu")
torch.manual_seed(42)
np.random.seed(42)


################################################################
#  configurations
################################################################
def save_training_state(params, _train_losses, _test_losses, model_path, losses_path):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    os.makedirs(os.path.dirname(losses_path), exist_ok=True)
    with open(model_path, 'wb') as f:
        pickle.dump(params, f)
    with open(losses_path, 'wb') as f:
        pickle.dump({'train_losses': _train_losses, 'test_losses': _test_losses}, f)
    print(f"Model parameters saved to {model_path} and losses saved to {losses_path}")


def load_training_state(model_path, losses_path):
    params = None
    _train_losses = []
    _test_losses = []

    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            params = pickle.load(f)
        print(f"Model parameters loaded from {model_path}")
    if os.path.exists(losses_path):
        with open(losses_path, 'rb') as f:
            losses_data = pickle.load(f)
            _train_losses = losses_data['train_losses']
            _test_losses = losses_data['test_losses']
        print(f"Training and testing losses loaded from {losses_path}")
    return params, _train_losses, _test_losses


def compute_validation_loss(_model_params, _test_loader, _loss_fn):
    total_loss = 0
    total_samples = 0
    with tqdm(_test_loader, unit="batch") as tVal:
        for batch in tVal:
            x_, y_ = batch
            loss_val = _loss_fn(_model_params, x_, y_)
            total_loss += loss_val * len(x_)
            total_samples += len(x_)
    avg_loss = total_loss / total_samples
    return avg_loss


class FNO2d_PlateHole(FNO2D):
    model_data: dict = None

    def __call__(self, x_: jnp.ndarray) -> jnp.ndarray:
        x_ = super().__call__(x_)
        x_ = x_.at[:, :, 0, :].set(0)
        return x_

    def get_grid(self, x_):
        grid = super().get_grid(x_)
        gridy = grid[..., 0:1] * self.model_data['width']
        gridx = grid[..., 1:2] * self.model_data['length']
        return jnp.concatenate((gridy, gridx), -1)


class PlateHoleDEM_dataset(PlateHole_dataset):
    def set_data(self):
        self.x = self.material
        self.y = self.disp2D
        if self.model_data['normalized']:
            self.make_normal()


plate_length = 5.
plate_width = 5.
num_pts_x = 200
num_pts_y = 200
# model_data dictionary
model_data = dict()
model_data["beam"] = dict()
model_data["beam"]["length"] = plate_length
model_data["beam"]["width"] = plate_width
model_data["beam"]["num_refinements"] = 5
model_data["beam"]["numPtsU"] = num_pts_x
model_data["beam"]["numPtsV"] = num_pts_y
model_data["beam"]["traction"] = jnp.ones_like(jnp.linspace(0, plate_width, num_pts_y))
model_data["beam"]["E"] = 100
model_data["beam"]["nu"] = 0.25
model_data["beam"]["state"] = "plane stress"

model_data["n_train"] = 1000
model_data["n_test"] = 200
model_data["n_data"] = model_data["n_train"] + model_data["n_test"]
model_data["n_dataset"] = 1200
model_data["batch_size"] = 20
model_data["batch_size_BFGS"] = 50
model_data["num_epoch"] = 50  # 2000
model_data["num_epoch_LBFGS"] = 100
model_data["nrg"] = random.PRNGKey(0)
model_data["data_type"] = 'float32'
if model_data["data_type"] == 'float64':
    jax.config.update("jax_enable_x64", True)

model_data["fno"] = dict()
model_data["fno"]["mode1"] = 8
model_data["fno"]["mode2"] = 8
model_data["fno"]["width"] = 32
model_data["fno"]["depth"] = 8
model_data["fno"]["channels_last_proj"] = 128
model_data["fno"]["padding"] = 56
model_data["fno"]["learning_rate"] = 0.01  # 0.001
model_data["fno"]["weight_decay"] = 1e-5
model_data["fno"]["scheduled"] = False

model_data["GRF"] = dict()
model_data["GRF"]["alpha"] = 10.
model_data["GRF"]["tau"] = 7.
model_data["normalized"] = False
model_data["dir"] = os.path.abspath(os.path.join(os.path.dirname(__file__), './data/'))
model_data["filename"] = ('PlateHole_LxW_' + str(plate_length) + "x" + str(plate_width) +
                          '_s' + str(model_data["beam"]["numPtsU"]) + "x" + str(model_data["beam"]["numPtsV"]) +
                          '_n' + str(model_data["n_dataset"]))
model_data["path"] = os.path.join(model_data["dir"], model_data["filename"] + '.npz')
model_save_path = './model/model_params_' + model_data["filename"] + '.pkl'
losses_save_path = './model/losses_' + model_data["filename"] + '.pkl'

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
dataset = PlateHoleDEM_dataset(model_data)
train_dataset, test_dataset, rest_dataset = random_split(dataset, [model_data["n_train"], model_data["n_test"],
                                                                   model_data["n_dataset"] - model_data["n_data"]])
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

# Define loss function
loss_fn = VinoPlateHoleLoss(model, model_data, normalizers, d=1, p=1, size_average=False)

loaded_model_params, train_losses, test_losses = load_training_state(model_save_path, losses_save_path)
if loaded_model_params is not None:
    model_params = loaded_model_params
    best_loss = compute_validation_loss(model_params, train_loader, loss_fn)
    print("Best loss test data from previous model: ", best_loss)
else:
    _x, _ = train_dataset[0]
    _x = jnp.expand_dims(_x, axis=0)
    _, model_params = model.init_with_output(model_data["nrg"], _x)
    del _x
    best_loss = float('inf')
    train_losses = []
    test_losses = []

n_params = count_params(model_params)
print(f'\nOur model has {n_params} parameters.')

print("Training (ADAM)...")
t0 = time.time()

if model_data["fno"]["scheduled"]:
    start_lr = 5e-3 # 1e-2
    end_lr = 1e-5
    steps_lr = 20  #10
    schedule = optax.linear_schedule(init_value=start_lr, end_value=end_lr, transition_steps=steps_lr)
    # start_lr = 1e-3
    # decay_rate = 0.5  # 0.9
    # transition_steps = 100  # 1000
    # schedule = optax.exponential_decay(init_value=start_lr, transition_steps=transition_steps, decay_rate=decay_rate)

    optimizer = optax.adam(learning_rate=schedule)
    opt_state = optimizer.init(model_params)

    model_params, train_losses, test_losses = train_fno_scheduled(
        model, train_loader, test_loader, loss_fn, opt_state, optimizer, model_params)
else:
    init_fun, update_fun, get_params = optimizers.adam(model_data["fno"]["learning_rate"])
    opt_state = init_fun(model_params)
    model_params, train_losses, test_losses = train_fno(model, train_loader, test_loader, loss_fn, get_params,
                                                        update_fun, opt_state, model_params, best_loss, train_losses, test_losses)

save_training_state(model_params, train_losses, test_losses, model_save_path, losses_save_path)

t1 = time.time()
print("Training (ADAM) time: ", t1 - t0)
"""
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

check_loader = DataLoader(
    train_dataset,
    batch_size=model_data["n_train"],
    shuffle=True,
    collate_fn=collate_fn,
    drop_last=True
)
X, Y = next(iter(check_loader))

n_stop = 1
best_loss = loss_fn(model_params, X, Y)
print(f"Patch loss: {best_loss}")
previous_results = None
for X_batch, Y_batch in train_loader:
    # Update the loss function for the current batch
    loss_func = jax_tfp_function_factory(model, model_params, loss_fn, X_batch, Y_batch)
    # initial_pos = loss_func.init_params_1d
    initial_pos = ravel_pytree(model_params)[0]
    current_loss, _ = loss_func(initial_pos)

    if previous_results is None:
        results = lbfgs_minimize(loss_func, initial_position=initial_pos, parallel_iterations=1,
                                 max_iterations=model_data["num_epoch_LBFGS"], num_correction_pairs=100,
                                 tolerance=1e-09)
    else:
        results = lbfgs_minimize(loss_func, initial_position=None, previous_optimizer_results=previous_results,
                                 parallel_iterations=1, max_iterations=model_data["num_epoch_LBFGS"],
                                 num_correction_pairs=100, tolerance=1e-20)

    num_bfgs_iterations += results.num_iterations
    print(
        f"Iteration: {num_bfgs_iterations}, Inner Iteration: {results.num_iterations}, Batch Loss: {results.objective_value}")
    train_losses.append(results.objective_value)

    _, unflatten_params = ravel_pytree(model_params)

    loss_step = loss_fn(model_params, X, Y)
    print(f"Patch loss: {loss_step}")
    if loss_step < best_loss:
        best_loss = loss_step
        model_params = unflatten_params(results.position)
        previous_results = results
        print(f"New best loss: {best_loss}")

t3 = time.time()
print("Total iterations: ", num_bfgs_iterations)
print("Time taken (BFGS)", t3 - t2, "seconds")
print("Time taken (all)", t3 - t_data_gen, "seconds")
#"""
# plot the convergence of the losses
folder = "VINO_Plate"
fig_font = "DejaVu Serif"
plt.rcParams["font.family"] = fig_font
plt.figure(figsize=(3.5, 4))
plt.plot(train_losses[5:], label='Train L2')
plt.plot(test_losses[5:], label='Test L2')
plt.legend()
if not os.path.exists(folder):
    os.makedirs(folder)
full_name = folder + '/' + "LossTrend"
plt.savefig(full_name + '.png', dpi=600, bbox_inches='tight')
plt.show()
################################################################
# evaluation
################################################################
train_loader = DataLoader(
    train_dataset,
    batch_size=1,
    shuffle=False,
    collate_fn=collate_fn,
    drop_last=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=False,
    collate_fn=collate_fn
)

# print("Evaluating Train Data")
# model_evaluation(model, model_data, model_params, loss_fn, train_loader, normalizers)
print("Evaluating Test Data")
model_evaluation(model, model_data, model_params, loss_fn, test_loader, normalizers)

# Train Dataset

index = 56  # 24
x_train, y_train = train_dataset[index]
x_train, y_train = jnp.expand_dims(x_train, axis=0), jnp.expand_dims(y_train, axis=0)
y_pred = model.apply(model_params, x_train)
if model_data["normalized"] is True:
    x_train = dataset.normalizer_x.decode(x_train)
    x_train = np.where(np.array(x_train) < 0.5, False, True)
    y_train = dataset.normalizer_y.decode(y_train)
    y_pred = dataset.normalizer_y.decode(y_pred)
# plot_field_2d(x_train[0, :, :, 0], beam_length, beam_width, "GRF-Random Input")
field_names = ['disp_x', 'disp_y']
for field_index, field_name in enumerate(field_names):
    plot_field_2d(y_train[0, :, :, field_index], plate_length, plate_width, "Train - Exact - " + field_name,
                  mask=x_train, folder=folder, file="TrainExact_" + field_name)
    plot_field_2d(y_pred[0, :, :, field_index], plate_length, plate_width, "Train - Predicted - " + field_name,
                  mask=x_train, folder=folder, file="TrainPredicted_" + field_name)
    plot_field_2d(y_pred[0, :, :, field_index] - y_train[0, :, :, field_index],
                  plate_length, plate_width, "Train - error - " + field_name,
                  mask=x_train, folder=folder, file="TrainError_" + field_name, isError=True)

# Test Dataset

index = 96  # 59
x_test, y_test = test_dataset[index]
x_test, y_test = jnp.expand_dims(x_test, axis=0), jnp.expand_dims(y_test, axis=0)
y_pred = model.apply(model_params, x_test)
if model_data["normalized"] is True:
    x_test = dataset.normalizer_x.decode(x_test)
    x_test = np.where(np.array(x_test) < 0.5, False, True)
    y_test = dataset.normalizer_y.decode(y_test)
    y_pred = dataset.normalizer_y.decode(y_pred)

for field_index, field_name in enumerate(field_names):
    plot_field_2d(y_test[0, :, :, field_index], plate_length, plate_width, "Test - Exact - " + field_name,
                  mask=x_test, folder=folder, file="TestExact_" + field_name)
    plot_field_2d(y_pred[0, :, :, field_index], plate_length, plate_width, "Test - Predicted - " + field_name,
                  mask=x_test, folder=folder, file="TestPredicted_" + field_name)
    plot_field_2d(y_pred[0, :, :, field_index] - y_test[0, :, :, field_index],
                  plate_length, plate_width, "Test - Error - " + field_name,
                  mask=x_test, folder=folder, file="TestError_" + field_name, isError=True)
print("Finish")
