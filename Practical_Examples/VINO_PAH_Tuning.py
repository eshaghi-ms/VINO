import jax
import time
import torch
import optax
import numpy as np
from jax import random
from jax import numpy as jnp
from matplotlib import pyplot as plt
# from jax.flatten_util import ravel_pytree
from jax.example_libraries import optimizers
from torch.utils.data import DataLoader, random_split
# from tensorflow_probability.substrates.jax.optimizer import lbfgs_minimize

from utils.fno_2d import FNO2D
# from utils.postprocessing import plot_field_2d
from utils.fno_utils import train_fno_scheduled
from utils.database_makers import PlateHole_dataset
# from utils.jax_tfp_loss import jax_tfp_function_factory
from utils.fno_utils import count_params, VinoPlateHoleLoss, train_fno, model_evaluation, collate_fn

# jax.default_device = jax.devices("cuda")[1]
torch.manual_seed(42)
np.random.seed(42)


################################################################
#  configurations
################################################################
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
model_data["beam"]["num_refinements"] = 6
model_data["beam"]["numPtsU"] = num_pts_x
model_data["beam"]["numPtsV"] = num_pts_y
model_data["beam"]["traction"] = jnp.ones_like(jnp.linspace(0, plate_width, num_pts_y))
model_data["beam"]["E"] = 100
model_data["beam"]["nu"] = 0.25
model_data["beam"]["state"] = "plane stress"

model_data["n_train"] = 200  # 1000
model_data["n_test"] = 50  # 100
model_data["n_data"] = model_data["n_train"] + model_data["n_test"]
model_data["n_dataset"] = 1100
model_data["batch_size"] = 50 #  100
model_data["num_epoch"] = 200 # 500 #  1000
model_data["nrg"] = random.PRNGKey(0)
model_data["data_type"] = 'float32'
if model_data["data_type"] == 'float64':
    jax.config.update("jax_enable_x64", True)

model_data["fno"] = dict()
model_data["fno"]["mode1"] = 8
model_data["fno"]["mode2"] = 8
model_data["fno"]["width"] = 32
model_data["fno"]["depth"] = 4
model_data["fno"]["channels_last_proj"] = 128
model_data["fno"]["padding"] = 0
#model_data["fno"]["learning_rate"] = 0.001
model_data["fno"]["weight_decay"] = 1e-4
model_data["fno"]["scheduled"] = True

model_data["GRF"] = dict()
model_data["GRF"]["alpha"] = 4.
model_data["GRF"]["tau"] = 15.
model_data["normalized"] = False
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

num_ind = 5
start_lr_s = [5e-3, 3e-3, 1e-3, 5e-4, 1e-4, 5e-5]
end_lr_s = [5e-4, 3e-4, 1e-4, 5e-5, 1e-5, 5e-6]
steps_lr_s = [5, 10, 20, 50, 100]

errors = jnp.zeros((len(start_lr_s), len(end_lr_s), len(steps_lr_s))) + jnp.inf
losses = jnp.zeros((len(start_lr_s), len(end_lr_s), len(steps_lr_s))) + jnp.inf

for ind1, start_lr in enumerate(start_lr_s):
    for ind2, end_lr in enumerate(end_lr_s):
        if start_lr > end_lr:
            for ind3, steps_lr in enumerate(steps_lr_s):
                print("start_ls = ", start_lr)
                print("end_ls = ", end_lr)
                print("steps_ls = ", steps_lr)

                t0 = time.time()

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

                # Define loss function
                loss_fn = VinoPlateHoleLoss(model, model_data, normalizers, d=1, p=1, size_average=False)

                print("Training (ADAM)...")



                if model_data["fno"]["scheduled"]:

                    schedule = optax.linear_schedule(init_value=start_lr, end_value=end_lr, transition_steps=steps_lr)
                    #schedule = optax.exponential_decay(init_value=base_lr, transition_steps=1000, decay_rate=0.9)
                    optimizer = optax.adam(learning_rate=schedule)
                    opt_state = optimizer.init(model_params)
                    model_params, train_losses, test_losses = train_fno_scheduled(model, train_loader, test_loader,
                                                                                  loss_fn, opt_state, optimizer, model_params)
                else:
                    init_fun, update_fun, get_params = optimizers.adam(model_data["fno"]["learning_rate"])
                    opt_state = init_fun(model_params)
                    model_params, train_losses, test_losses = train_fno(model, train_loader, test_loader, loss_fn,
                                                                        get_params, update_fun, opt_state)

                t1 = time.time()
                print("Training (ADAM) time: ", t1 - t0)

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
                errors = errors.at[ind1, ind2, ind3].set(
                    model_evaluation(model, model_data, model_params, loss_fn, test_loader, normalizers, True))
                losses = losses.at[ind1, ind2, ind3].set(min(train_losses))

                t1 = time.time()
                print("Loop time: ", t1 - t0)

print("'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''")
print("'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''")
print("'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''")

index1, index2, index3 = jnp.unravel_index(jnp.argmin(errors), errors.shape)
index4, index5, index6 = jnp.unravel_index(jnp.argmin(losses), errors.shape)

print("Minimum Error")
print("start_ls = ", start_lr_s[index1])
print("end_ls = ", end_lr_s[index2])
print("steps_ls = ", steps_lr_s[index3])
print("finish")

print("Minimum Loss")
print("start_ls = ", start_lr_s[index4])
print("end_ls = ", end_lr_s[index5])
print("steps_ls = ", steps_lr_s[index6])

