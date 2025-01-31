import time

import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm
import optax

from .processing import diff_x, diff_y, difference_x, difference_y


def collate_fn(batch):
    x, y = zip(*batch)
    x = np.stack(x, axis=0)
    y = np.stack(y, axis=0)
    return x, y


def count_params(params):
    flat_params, _ = jax.tree_util.tree_flatten(params)
    return sum(param.size for param in flat_params)


def model_evaluation(model, model_data, model_params, loss_fn, test_loader, normalizers=None, output=False, mask=None):
    pred = jnp.zeros(test_loader.dataset.dataset.y.shape)
    index = 0
    # x_test = jnp.zeros(test_loader.dataset.dataset.x.shape)
    y_test = jnp.zeros(pred.shape)
    test_losses_set = []
    for x, y in test_loader:
        if model_data["normalized"] is True:
            normalizer_x = normalizers[0]
            normalizer_y = normalizers[1]
            x = x if mask is None else normalizer_x.decode(x)
            y = normalizer_y.decode(y)
            t0 = time.time()
            y_pred = normalizer_y.decode(model.apply(model_params, x))[0, :, :, :]
        else:
            t0 = time.time()
            y_pred = model.apply(model_params, x)[0, :, :, :]
        print("time for one sample: ", time.time()-t0)
        # x_test = x_test.at[index].set(x[0, :, :, :])
        if mask is None:
            y_test = y_test.at[index].set(y[0, :, :, :])
            pred = pred.at[index].set(y_pred)
        else:
            y_test = y_test.at[index].set((y * (1-x))[0, :, :, :])
            pred = pred.at[index].set(y_pred * (1-x)[0, :, :, :])
        loss_val = loss_fn(y_test[index].reshape(1, -1), pred[index].reshape(1, -1)).item()
        test_losses_set.append(loss_val)
        print(index, loss_val)
        index = index + 1

    test_losses_array = jnp.array(test_losses_set)
    test_losses_avg = jnp.mean(test_losses_array)
    test_losses_std = jnp.std(test_losses_array)
    test_losses_argmax = jnp.argmax(test_losses_array)
    print("The average testing error is", test_losses_avg)
    print("Std. deviation of testing error is", test_losses_std)
    print("Min testing error is", jnp.min(test_losses_array).item())
    print("Max testing error is", jnp.max(test_losses_array).item())
    print("Index of maximum error is", test_losses_argmax.item())
    if output:
        return test_losses_avg


def train_fno(model, train_loader, test_loader, loss_fn, get_params, update_fun,
              opt_state, model_params=None, best_loss=float('inf'), train_losses=None, test_losses=None):
    if test_losses is None:
        test_losses = []
    if train_losses is None:
        train_losses = []

    @jax.jit
    def update(_opt_state, _x, _y, _step):
        params = get_params(_opt_state)
        _loss_val, grads = jax.value_and_grad(loss_fn)(params, _x, _y)

        # Add weight decay
        grads = {'params': jax.tree.map(
            lambda g, p: g + loss_fn.model_data["fno"]["weight_decay"] * p,
            grads['params'], params['params'])}
        _opt_state = update_fun(_step, grads, _opt_state)
        return _opt_state, _loss_val

    # train_losses = train_losses
    # test_losses = test_losses
    # best_loss = best_loss
    best_model_params = model_params

    # Training loop
    step = 0
    for epoch in range(loss_fn.model_data["num_epoch"]):
        tqdm.write(f"\nEpoch {epoch}")
        # Perform one epoch of training
        with tqdm(train_loader, unit="batch", leave=False) as tEpoch:
            avg_loss = 0
            for batch in tEpoch:
                tEpoch.set_description(f"Epoch {epoch}")
                # Update parameters
                x, y = batch
                opt_state, loss_val = update(opt_state, x, y, step)

                # Log
                tEpoch.set_postfix(loss=loss_val)
                avg_loss += loss_val * len(x)
                # print("training error = ", lossVal)
                step += 1
            avg_loss /= sum(len(batch[0]) for batch in train_loader)
            train_losses.append(avg_loss)
        # Get new parameters
        model_params = get_params(opt_state)

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model_params = model_params
            tqdm.write(f"New best loss: {best_loss}")

        # Log a training image
        # y_pred = model.apply(model_params, x)

        # Validation
        with tqdm(test_loader, unit="batch") as tVal:
            avg_loss = 0
            for batch in tVal:
                tVal.set_description(f"Epoch (val) {epoch}")
                x, y = batch
                loss_val = loss_fn(model_params, x, y)
                avg_loss += loss_val * len(x)
                # print("test error = ", lossVal)

                tVal.set_postfix(loss=loss_val)

            avg_loss /= sum(len(batch[0]) for batch in test_loader)
            test_losses.append(avg_loss)

    return best_model_params, train_losses, test_losses


def train_fno_scheduled(model, train_loader, test_loader, loss_fn, opt_state, optimizer, model_params,
                        best_loss=float('inf'), train_losses=None, test_losses=None):
    if test_losses is None:
        test_losses = []
    if train_losses is None:
        train_losses = []

    @jax.jit
    def update(_opt_state, _x, _y, _model_params):
        _loss_val, grads = jax.value_and_grad(loss_fn)(_model_params, _x, _y)

        # Add weight decay
        grads = {'params': jax.tree.map(
            lambda g, p: g + loss_fn.model_data["fno"]["weight_decay"] * p,
            grads['params'], _model_params['params'])}
        updates, _opt_state = optimizer.update(grads, _opt_state)
        params = optax.apply_updates(_model_params, updates)
        return params, _opt_state, _loss_val

    # train_losses = []
    # test_losses = []
    # best_loss = float('inf')
    best_model_params = model_params

    # Training loop
    step = 0
    for epoch in range(loss_fn.model_data["num_epoch"]):
        tqdm.write(f"\nEpoch {epoch}")
        # Perform one epoch of training
        with tqdm(train_loader, unit="batch", leave=False) as tEpoch:
            avg_loss = 0
            for batch in tEpoch:
                tEpoch.set_description(f"Epoch {epoch}")
                # Update parameters
                x, y = batch
                model_params, opt_state, loss_val = update(opt_state, x, y, model_params)

                # Log
                tEpoch.set_postfix(loss=loss_val)
                avg_loss += loss_val * len(x)

                step += 1
            avg_loss /= sum(len(batch[0]) for batch in train_loader)
            train_losses.append(avg_loss)

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model_params = model_params
            tqdm.write(f"New best loss: {best_loss}")

        # Log a training image
        # y_pred = model.apply(model_params, x)

        # Validation
        with tqdm(test_loader, unit="batch") as tVal:
            avg_loss = 0
            for batch in tVal:
                tVal.set_description(f"Epoch (val) {epoch}")
                x, y = batch
                loss_val = loss_fn(model_params, x, y)
                avg_loss += loss_val * len(x)

                tVal.set_postfix(loss=loss_val)

            avg_loss /= sum(len(batch[0]) for batch in test_loader)
            test_losses.append(avg_loss)

    return best_model_params, train_losses, test_losses


class LpLoss(object):
    def __init__(self, model, model_data, d=2, p=None, size_average=True, reduction=True):
        super(LpLoss, self).__init__()
        # assert d > 0 and p > 0
        # Dimension and Lp-norm type are positive
        self.model = model
        self.model_data = model_data
        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size
        # Assume uniform mesh
        h = 1.0 / (x.shape[1] - 1.0)

        all_norms = (h ** (self.d / self.p)) * jnp.linalg.norm(x.reshape(num_examples, -1) -
                                                               y.reshape(num_examples, -1), ord=self.p)

        if self.reduction:
            if self.size_average:
                return jnp.mean(all_norms)
            else:
                return jnp.sum(all_norms)
        return all_norms

    def rel(self, x, y):
        num_examples = x.size
        diff_norms = jnp.linalg.norm(x.reshape(num_examples, -1) - y.reshape(num_examples, -1), ord=self.p)
        y_norms = jnp.linalg.norm(y.reshape(num_examples, -1), ord=self.p)
        if self.reduction:
            if self.size_average:
                return jnp.mean(diff_norms / y_norms)
            else:
                return jnp.sum(diff_norms / y_norms)
        return diff_norms / y_norms

    def loss_fn(self, params, x, y):
        y_pred = self.model.apply(params, x)
        # return jnp.mean(jnp.square(y - y_pred))
        return self.rel(y_pred, y)

    def __call__(self, *args):
        if len(args) == 2:
            x, y = args
            return self.rel(x, y)
        elif len(args) == 3:
            params, x, y = args
            return self.loss_fn(params, x, y)
        else:
            raise ValueError("Invalid number of arguments. Expected 2 or 3 arguments.")


class PinoDarcyLoss(LpLoss):
    def __init__(self, model, model_data, normalizers, d=2, p=None, size_average=True, reduction=True):
        super().__init__(model, model_data, d, p, size_average, reduction)
        if self.model_data["normalized"] is True:
            self.normalizer_x = normalizers[0]
            self.normalizer_y = normalizers[1]

    def loss_fn(self, params, x, y):
        # batch_size = x.shape[0]
        if self.model_data["normalized"] is True:
            y_out = self.model.apply(params, x)
            y_out = self.normalizer_y.decode(y_out)
            a = self.normalizer_x.decode(x)
        else:
            y_out = self.model.apply(params, x)
            a = x

        dudx = diff_x(y_out, 1)
        dudy = diff_y(y_out, 1)

        f_pred_1 = diff_x((a * dudx), 1)
        f_pred_2 = diff_y((a * dudy), 1)

        f_pred = -(f_pred_1 + f_pred_2)
        f_ground = jnp.ones_like(a)

        # return jnp.sqrt(jnp.mean(jnp.square(f_pred - f_ground)))
        # return jnp.mean(jnp.square(f_pred - f_ground))
        return self.rel(f_pred, f_ground)
        # return jnp.mean(jnp.abs(f_pred - f_ground))


class VinoDarcyLoss(LpLoss):
    def __init__(self, model, model_data, normalizers, d=2, p=None, size_average=True, reduction=True):
        super().__init__(model, model_data, d, p, size_average, reduction)
        if self.model_data["normalized"] is True:
            self.normalizer_x = normalizers[0]
            self.normalizer_y = normalizers[1]

    def loss_fn(self, params, x, y):
        # batch_size = x.shape[0]
        if self.model_data["normalized"] is True:
            y_out = self.model.apply(params, x)
            y_out = self.normalizer_y.decode(y_out)
            a = self.normalizer_x.decode(x)
        else:
            y_out = self.model.apply(params, x)
            a = x

        # num_x = self.model_data.grid_point_num
        # num_y = self.model_data.grid_point_num
        # dUdx = diff_x(y_out, 1)
        # dUdy = diff_y(y_out, 1)
        # integrand = 0.5 * a * (dUdx ** 2 + dUdy ** 2) - y_out
        # return jnp.sum(integrand) / ((num_x - 1) * (num_y - 1))

        dudx = difference_x(y_out)
        dudy = difference_y(y_out)

        a_00 = a[:, 0:-1, 0:-1, :]
        a_10 = a[:, 1:, 0:-1, :]
        a_01 = a[:, 0:-1, 1:, :]
        a_11 = a[:, 1:, 1:, :]

        dudx_0 = dudx[:, 0:-1, 0:-1, :]
        dudx_1 = dudx[:, 1:, 0:-1, :]
        dudy_0 = dudy[:, 0:-1, 0:-1, :]

        loss_grad = 1 / 2 * jnp.sum(
            (a_00 * dudx_0 ** 2) / 6 + (a_00 * dudx_1 ** 2) / 12 + (a_00 * dudy_0 ** 2) / 4 + (a_01 * dudx_0 ** 2) / 4
            + (a_01 * dudx_1 ** 2) / 6 + (a_01 * dudy_0 ** 2) / 4 + (a_10 * dudx_0 ** 2) / 12 + (a_10 * dudx_1 ** 2) / 6
            + (a_10 * dudy_0 ** 2) / 4 + (a_11 * dudx_0 ** 2) / 6 + (a_11 * dudx_1 ** 2) / 4 + (a_11 * dudy_0 ** 2) / 4
            - (a_00 * dudx_0 * dudy_0) / 6 + (a_00 * dudx_1 * dudy_0) / 6 - (a_01 * dudx_0 * dudx_1) / 6
            - (a_01 * dudx_0 * dudy_0) / 3 + (a_01 * dudx_1 * dudy_0) / 3 - (a_10 * dudx_0 * dudy_0) / 6
            + (a_10 * dudx_1 * dudy_0) / 6 - (a_11 * dudx_0 * dudx_1) / 6 - (a_11 * dudx_0 * dudy_0) / 3
            + (a_11 * dudx_1 * dudy_0) / 3
        )

        num_x = x.shape[2]
        num_y = x.shape[1]
        # mask = 0.5 * jnp.ones_like(x)
        # mask = mask.at[:, 1:-1, 1:-1, :].set(jnp.ones_like(x[:, 1:-1, 1:-1, :]))
        # mask = mask.at[:, 0, 0, 0].set(0.25)
        # mask = mask.at[:, 0, -1, 0].set(0.25)
        # mask = mask.at[:, -1, 0, 0].set(0.25)
        # mask = mask.at[:, -1, -1, 0].set(0.25)
        # loss_int = jnp.sum(mask * y_out) / ((num_x - 1) * (num_y - 1))
        loss_int = jnp.sum(y_out) / ((num_x - 1) * (num_y - 1))

        return loss_grad - loss_int


class PinoFGMLoss(LpLoss):
    def __init__(self, model, model_data, normalizers, d=2, p=2, size_average=True, reduction=True):
        super().__init__(model, model_data, d=d, p=p, size_average=size_average, reduction=reduction)
        self.L = model_data['beam']['length']
        self.W = model_data['beam']['width']
        self.num_x = model_data['beam']['numPtsU']
        self.num_y = model_data['beam']['numPtsV']
        self.nu = model_data['beam']['nu']
        self.state = model_data['beam']['state']
        self.data_type = model_data['data_type']
        self.states = {"plane strain": 1 / ((1 + self.nu) * (1 - 2 * self.nu)) * jnp.array(
            [[1 - self.nu, self.nu, 0], [self.nu, 1 - self.nu, 0], [0, 0, (1 - 2 * self.nu) / 2]]),
                       "plane stress": 1 / (1 - self.nu ** 2) * jnp.array(
                           [[1, self.nu, 0], [self.nu, 1, 0], [0, 0, (1 - self.nu) / 2]])}
        if self.model_data["normalized"]:
            self.normalizer_x = normalizers[0]
            self.normalizer_y = normalizers[1]

    def grad_disp(self, y_out):
        u = y_out[..., 0:1]
        v = y_out[..., 1:2]
        dudx = diff_x(u, self.L)
        dudy = diff_y(u, self.W)
        dvdx = diff_x(v, self.L)
        dvdy = diff_y(v, self.W)
        return dudx, dudy, dvdx, dvdy

    def kinematicEq(self, y_out):
        dudx, dudy, dvdx, dvdy = self.grad_disp(y_out)
        eps_xx_val = dudx
        eps_yy_val = dvdy
        eps_xy_val = (dudy + dvdx)
        return eps_xx_val, eps_yy_val, eps_xy_val

    def constitutiveEq(self, x, y_out):
        e_mat = self.states[self.state]
        elasticity_modulus = x[..., 1:2]
        eps_xx_val, eps_yy_val, eps_xy_val = self.kinematicEq(y_out)
        eps_val = jnp.concatenate([eps_xx_val, eps_yy_val, eps_xy_val], axis=-1)
        elasticity_modulus = jnp.expand_dims(elasticity_modulus, axis=-1)
        e_mat = jnp.expand_dims(jnp.expand_dims(e_mat, axis=0), axis=0)
        e_porous = jnp.multiply(elasticity_modulus, e_mat)

        eps_val = jnp.expand_dims(eps_val, axis=-1)
        stress_val = jnp.matmul(e_porous, eps_val).squeeze(-1)

        stress_xx_val = stress_val[..., 0:1]
        stress_yy_val = stress_val[..., 1:2]
        stress_xy_val = stress_val[..., 2:3]
        return stress_xx_val, stress_yy_val, stress_xy_val

    def balanceEq(self, x, y_out):
        stress_xx_val, stress_yy_val, stress_xy_val = self.constitutiveEq(x, y_out)
        dx_stress_xx_val = diff_x(stress_xx_val, self.L)
        dy_stress_yy_val = diff_y(stress_yy_val, self.W)
        dx_stress_xy_val = diff_x(stress_xy_val, self.L)
        dy_stress_xy_val = diff_y(stress_xy_val, self.W)
        f_x = dx_stress_xx_val + dy_stress_xy_val
        f_y = dx_stress_xy_val + dy_stress_yy_val
        return f_x, f_y

    def loss_fn(self, params, x, y):
        if self.model_data["normalized"]:
            y_out = self.model.apply(params, x)
            y_out = self.normalizer_y.decode(y_out)
            x_real = self.normalizer_x.decode(x)
            # y_real = self.normalizer_y.decode(y)
        else:
            y_out = self.model.apply(params, x)
            x_real = x
            # y_real = y

        # loss_data = self.rel(y_out, y_real)
        # calculate the interior loss
        f_x_int, f_y_int = self.balanceEq(x_real, y_out)
        loss_int = (jnp.mean(jnp.abs(f_x_int[:, 1:-1, 1:-1, :])) / len(f_x_int) +
                    jnp.mean(jnp.abs(f_y_int[:, 1:-1, 1:-1, :])) / len(f_y_int))

        # calculate the boundary loss
        stress_xx_val, stress_yy_val, stress_xy_val = self.constitutiveEq(x_real, y_out)
        stress_xy_down = stress_xy_val[:, 0, 1:-1, :]
        stress_yy_down = stress_yy_val[:, 0, 1:-1, :]
        trac_x_down = -1 * stress_xy_down
        trac_y_down = -1 * stress_yy_down

        stress_xy_up = stress_xy_val[:, -1, 1:-1, :]
        stress_yy_up = stress_yy_val[:, -1, 1:-1, :]
        trac_x_up = stress_xy_up
        trac_y_up = stress_yy_up
        ty = -x_real[:, 0, 1:-1, 0:1]

        loss_bnd_down = jnp.mean(jnp.abs(trac_x_down)) / len(trac_x_down) + jnp.mean(jnp.abs(trac_y_down)) / len(
            trac_y_down)
        loss_bnd_up = jnp.mean(jnp.abs(trac_x_up)) / len(trac_x_up) + jnp.mean(jnp.abs(trac_y_up - ty)) / len(trac_y_up)

        # jax.debug.print("\n loss_int: {}, loss_bnd_down: {}, loss_bnd_up: {}, loss_data: {}",
        #                 loss_int, loss_bnd_down, loss_bnd_up, loss_data)
        # return 1e-2 * loss_int + loss_bnd_down + loss_bnd_up + loss_data
        # return 1e-2 * loss_int + loss_bnd_down + loss_bnd_up
        return loss_int + loss_bnd_down + loss_bnd_up
        # return loss_data


class VinoFGMLoss(PinoFGMLoss):
    def __init__(self, model, model_data, normalizers, d=2, p=2, size_average=True, reduction=True):
        super().__init__(model, model_data, normalizers, d, p, size_average, reduction)

    @staticmethod
    def difference_disp(y_out):
        u = y_out[..., 0:1]
        v = y_out[..., 1:2]
        dudx = difference_x(u)
        dudy = difference_y(u)
        dvdx = difference_x(v)
        dvdy = difference_y(v)
        return dudx, dudy, dvdx, dvdy

    def loss_fn(self, params, x, y):

        if self.model_data["normalized"]:
            y_out = self.model.apply(params, x)
            y_out = self.normalizer_y.decode(y_out)
            x_real = self.normalizer_x.decode(x)
            # y_real = self.normalizer_y.decode(y)
        else:
            y_out = self.model.apply(params, x)
            x_real = x
            # y_real = y

        elasticity_modulus = x_real[..., 1:2]
        e_00 = elasticity_modulus[:, 0:-1, 0:-1, :]
        e_10 = elasticity_modulus[:, 1:, 0:-1, :]
        e_01 = elasticity_modulus[:, 0:-1, 1:, :]
        e_11 = elasticity_modulus[:, 1:, 1:, :]

        # loss_data = self.rel(y_out, y_real)
        # calculate the interior loss
        dudx, dudy, dvdx, dvdy = self.difference_disp(y_out)

        dudx_0 = dudx[:, 0:-1, 0:-1, :]
        dudx_1 = dudx[:, 1:, 0:-1, :]
        dudy_0 = dudy[:, 0:-1, 0:-1, :]
        dvdx_0 = dvdx[:, 0:-1, 0:-1, :]
        dvdx_1 = dvdx[:, 1:, 0:-1, :]
        dvdy_0 = dvdy[:, 0:-1, 0:-1, :]

        dx = self.L / self.num_x
        dy = self.W / self.num_y
        n = dy / dx
        nu = self.nu

        loss_int = jnp.sum(
            -(3 * e_00 * dudx_0 ** 2 + 3 * e_00 * dudx_1 ** 2 + 18 * e_00 * dudy_0 ** 2 + 9 * e_01 * dudx_0 ** 2 +
              9 * e_01 * dudx_1 ** 2 + 18 * e_01 * dudy_0 ** 2 + 3 * e_10 * dudx_0 ** 2 + 3 * e_10 * dudx_1 ** 2 +
              18 * e_10 * dudy_0 ** 2 + 9 * e_11 * dudx_0 ** 2 + 9 * e_11 * dudx_1 ** 2 + 18 * e_11 * dudy_0 ** 2 +
              6 * e_00 * dvdx_0 ** 2 + 6 * e_00 * dvdx_1 ** 2 + 36 * e_00 * dvdy_0 ** 2 + 18 * e_01 * dvdx_0 ** 2 +
              18 * e_01 * dvdx_1 ** 2 + 36 * e_01 * dvdy_0 ** 2 + 6 * e_10 * dvdx_0 ** 2 + 6 * e_10 * dvdx_1 ** 2 +
              36 * e_10 * dvdy_0 ** 2 + 18 * e_11 * dvdx_0 ** 2 + 18 * e_11 * dvdx_1 ** 2 + 36 * e_11 * dvdy_0 ** 2 +
              18 * e_00 * dudx_0 ** 2 * n ** 2 + 6 * e_00 * dudx_1 ** 2 * n ** 2 + 18 * e_01 * dudx_0 ** 2 * n ** 2 +
              6 * e_01 * dudx_1 ** 2 * n ** 2 + 6 * e_10 * dudx_0 ** 2 * n ** 2 + 18 * e_10 * dudx_1 ** 2 * n ** 2 +
              6 * e_11 * dudx_0 ** 2 * n ** 2 + 18 * e_11 * dudx_1 ** 2 * n ** 2 + 9 * e_00 * dvdx_0 ** 2 * n ** 2 +
              3 * e_00 * dvdx_1 ** 2 * n ** 2 + 9 * e_01 * dvdx_0 ** 2 * n ** 2 + 3 * e_01 * dvdx_1 ** 2 * n ** 2 +
              3 * e_10 * dvdx_0 ** 2 * n ** 2 + 9 * e_10 * dvdx_1 ** 2 * n ** 2 + 3 * e_11 * dvdx_0 ** 2 * n ** 2 +
              9 * e_11 * dvdx_1 ** 2 * n ** 2 - 6 * e_00 * dudx_0 * dudx_1 - 12 * e_00 * dudx_0 * dudy_0 +
              12 * e_00 * dudx_1 * dudy_0 - 18 * e_01 * dudx_0 * dudx_1 - 24 * e_01 * dudx_0 * dudy_0 +
              24 * e_01 * dudx_1 * dudy_0 - 6 * e_10 * dudx_0 * dudx_1 - 12 * e_10 * dudx_0 * dudy_0 +
              12 * e_10 * dudx_1 * dudy_0 - 18 * e_11 * dudx_0 * dudx_1 - 24 * e_11 * dudx_0 * dudy_0 +
              24 * e_11 * dudx_1 * dudy_0 - 12 * e_00 * dvdx_0 * dvdx_1 - 24 * e_00 * dvdx_0 * dvdy_0 +
              24 * e_00 * dvdx_1 * dvdy_0 - 36 * e_01 * dvdx_0 * dvdx_1 - 48 * e_01 * dvdx_0 * dvdy_0 +
              48 * e_01 * dvdx_1 * dvdy_0 - 12 * e_10 * dvdx_0 * dvdx_1 - 24 * e_10 * dvdx_0 * dvdy_0 +
              24 * e_10 * dvdx_1 * dvdy_0 - 36 * e_11 * dvdx_0 * dvdx_1 - 48 * e_11 * dvdx_0 * dvdy_0 +
              48 * e_11 * dvdx_1 * dvdy_0 - 3 * e_00 * dudx_0 ** 2 * nu - 3 * e_00 * dudx_1 ** 2 * nu -
              18 * e_00 * dudy_0 ** 2 * nu - 9 * e_01 * dudx_0 ** 2 * nu - 9 * e_01 * dudx_1 ** 2 * nu -
              18 * e_01 * dudy_0 ** 2 * nu - 3 * e_10 * dudx_0 ** 2 * nu - 3 * e_10 * dudx_1 ** 2 * nu -
              18 * e_10 * dudy_0 ** 2 * nu - 9 * e_11 * dudx_0 ** 2 * nu - 9 * e_11 * dudx_1 ** 2 * nu -
              18 * e_11 * dudy_0 ** 2 * nu + 12 * e_00 * dudx_0 * dudx_1 * n ** 2 +
              12 * e_01 * dudx_0 * dudx_1 * n ** 2 + 12 * e_10 * dudx_0 * dudx_1 * n ** 2 +
              12 * e_11 * dudx_0 * dudx_1 * n ** 2 + 6 * e_00 * dvdx_0 * dvdx_1 * n ** 2 +
              6 * e_01 * dvdx_0 * dvdx_1 * n ** 2 + 6 * e_10 * dvdx_0 * dvdx_1 * n ** 2 +
              6 * e_11 * dvdx_0 * dvdx_1 * n ** 2 - 9 * e_00 * dvdx_0 ** 2 * n ** 2 * nu -
              3 * e_00 * dvdx_1 ** 2 * n ** 2 * nu - 9 * e_01 * dvdx_0 ** 2 * n ** 2 * nu -
              3 * e_01 * dvdx_1 ** 2 * n ** 2 * nu - 3 * e_10 * dvdx_0 ** 2 * n ** 2 * nu -
              9 * e_10 * dvdx_1 ** 2 * n ** 2 * nu - 3 * e_11 * dvdx_0 ** 2 * n ** 2 * nu -
              9 * e_11 * dvdx_1 ** 2 * n ** 2 * nu - 8 * e_00 * dudx_0 * dvdx_0 * n - 4 * e_00 * dudx_0 * dvdx_1 * n +
              8 * e_00 * dudx_1 * dvdx_0 * n + 24 * e_00 * dudy_0 * dvdx_0 * n - 16 * e_01 * dudx_0 * dvdx_0 * n +
              4 * e_00 * dudx_1 * dvdx_1 * n + 12 * e_00 * dudy_0 * dvdx_1 * n - 8 * e_01 * dudx_0 * dvdx_1 * n +
              16 * e_01 * dudx_1 * dvdx_0 * n + 24 * e_01 * dudy_0 * dvdx_0 * n + 8 * e_01 * dudx_1 * dvdx_1 * n +
              12 * e_01 * dudy_0 * dvdx_1 * n - 4 * e_10 * dudx_0 * dvdx_0 * n - 8 * e_10 * dudx_0 * dvdx_1 * n +
              4 * e_10 * dudx_1 * dvdx_0 * n + 12 * e_10 * dudy_0 * dvdx_0 * n - 8 * e_11 * dudx_0 * dvdx_0 * n +
              8 * e_10 * dudx_1 * dvdx_1 * n + 24 * e_10 * dudy_0 * dvdx_1 * n - 16 * e_11 * dudx_0 * dvdx_1 * n +
              8 * e_11 * dudx_1 * dvdx_0 * n + 12 * e_11 * dudy_0 * dvdx_0 * n + 16 * e_11 * dudx_1 * dvdx_1 * n +
              24 * e_11 * dudy_0 * dvdx_1 * n + 6 * e_00 * dudx_0 * dudx_1 * nu + 12 * e_00 * dudx_0 * dudy_0 * nu -
              12 * e_00 * dudx_1 * dudy_0 * nu + 18 * e_01 * dudx_0 * dudx_1 * nu + 24 * e_01 * dudx_0 * dudy_0 * nu -
              24 * e_01 * dudx_1 * dudy_0 * nu + 6 * e_10 * dudx_0 * dudx_1 * nu + 12 * e_10 * dudx_0 * dudy_0 * nu -
              12 * e_10 * dudx_1 * dudy_0 * nu + 18 * e_11 * dudx_0 * dudx_1 * nu + 24 * e_11 * dudx_0 * dudy_0 * nu -
              24 * e_11 * dudx_1 * dudy_0 * nu - 8 * e_00 * dudx_0 * dvdx_0 * n * nu +
              20 * e_00 * dudx_0 * dvdx_1 * n * nu + 48 * e_00 * dudx_0 * dvdy_0 * n * nu -
              16 * e_00 * dudx_1 * dvdx_0 * n * nu - 24 * e_00 * dudy_0 * dvdx_0 * n * nu -
              16 * e_01 * dudx_0 * dvdx_0 * n * nu + 4 * e_00 * dudx_1 * dvdx_1 * n * nu +
              24 * e_00 * dudx_1 * dvdy_0 * n * nu - 12 * e_00 * dudy_0 * dvdx_1 * n * nu +
              40 * e_01 * dudx_0 * dvdx_1 * n * nu + 48 * e_01 * dudx_0 * dvdy_0 * n * nu -
              32 * e_01 * dudx_1 * dvdx_0 * n * nu - 24 * e_01 * dudy_0 * dvdx_0 * n * nu +
              8 * e_01 * dudx_1 * dvdx_1 * n * nu + 24 * e_01 * dudx_1 * dvdy_0 * n * nu -
              12 * e_01 * dudy_0 * dvdx_1 * n * nu - 4 * e_10 * dudx_0 * dvdx_0 * n * nu +
              16 * e_10 * dudx_0 * dvdx_1 * n * nu + 24 * e_10 * dudx_0 * dvdy_0 * n * nu -
              20 * e_10 * dudx_1 * dvdx_0 * n * nu - 12 * e_10 * dudy_0 * dvdx_0 * n * nu -
              8 * e_11 * dudx_0 * dvdx_0 * n * nu + 8 * e_10 * dudx_1 * dvdx_1 * n * nu +
              48 * e_10 * dudx_1 * dvdy_0 * n * nu - 24 * e_10 * dudy_0 * dvdx_1 * n * nu +
              32 * e_11 * dudx_0 * dvdx_1 * n * nu + 24 * e_11 * dudx_0 * dvdy_0 * n * nu -
              40 * e_11 * dudx_1 * dvdx_0 * n * nu - 12 * e_11 * dudy_0 * dvdx_0 * n * nu +
              16 * e_11 * dudx_1 * dvdx_1 * n * nu + 48 * e_11 * dudx_1 * dvdy_0 * n * nu -
              24 * e_11 * dudy_0 * dvdx_1 * n * nu - 6 * e_00 * dvdx_0 * dvdx_1 * n ** 2 * nu -
              6 * e_01 * dvdx_0 * dvdx_1 * n ** 2 * nu - 6 * e_10 * dvdx_0 * dvdx_1 * n ** 2 * nu -
              6 * e_11 * dvdx_0 * dvdx_1 * n ** 2 * nu) / (288 * n * (nu ** 2 - 1))
        )

        # calculate the boundary loss
        disp_x, disp_y = y_out[..., 0:1], y_out[..., 1:2]
        disp_y_top = disp_y[:, -1, :, :]
        ty = x_real[:, 0, :, 0:1]

        loss_bnd = jnp.sum(disp_y_top * ty) * dx
        return loss_int - loss_bnd


class DemHyperelasticityLoss(LpLoss):
    def __init__(self, model, model_data, normalizers, d=2, p=2, size_average=True, reduction=True):
        super().__init__(model, model_data, d=d, p=p, size_average=size_average, reduction=reduction)
        self.L = model_data["beam"]['length']
        self.W = model_data["beam"]['width']
        self.num_x = model_data["beam"]["numPtsU"]
        self.num_y = model_data["beam"]["numPtsV"]
        self.nu = model_data["beam"]["nu"]
        self.data_type = model_data["data_type"]

        self.type = model_data["FEM_data"]["energy_type"]
        self.dim = model_data["FEM_data"]["dimension"]
        if self.type == 'neohookean':
            self.E = model_data["beam"]["E"]
            self.nu = model_data["beam"]["nu"]
            self.mu = self.E / (2 * (1 + self.nu))
            self.lam = (self.E * self.nu) / ((1 + self.nu) * (1 - 2 * self.nu))
        if self.type == 'mooneyrivlin':
            self.param_c1 = model_data["beam"]["param_c1"]
            self.param_c2 = model_data["beam"]["param_c2"]
            self.param_c = model_data["beam"]["param_c"]
            self.param_d = 2 * (self.param_c1 + 2 * self.param_c2)
        self.Wint = None
        self.get_int_weights()
        if self.model_data["normalized"]:
            self.normalizer_x = normalizers[0]
            self.normalizer_y = normalizers[1]

    def get_int_weights(self):
        self.Wint = jnp.ones((1, self.num_y - 2, self.num_x - 2, 1), dtype=self.data_type)
        top_bottom_edge = 1 / 2 * jnp.ones((1, 1, self.num_x - 2, 1), dtype=self.data_type)
        self.Wint = jnp.concatenate((top_bottom_edge, self.Wint, top_bottom_edge), axis=1)
        left_right_edge = jnp.concatenate((jnp.array([[[[1 / 4]]]], dtype=self.data_type),
                                           1 / 2 * jnp.ones((1, self.num_y - 2, 1, 1), dtype=self.data_type),
                                           jnp.array([[[[1 / 4]]]], dtype=self.data_type)), axis=1)
        self.Wint = jnp.concatenate((left_right_edge, self.Wint, left_right_edge), axis=2)

    def grad_disp(self, y_out):
        u = y_out[..., 0:1]
        v = y_out[..., 1:2]
        dudx = diff_x(u, self.L)
        dudy = diff_y(u, self.W)
        dvdx = diff_x(v, self.L)
        dvdy = diff_y(v, self.W)
        return dudx, dudy, dvdx, dvdy

    def MooneyRivlin2D(self, y_out):
        dudx, dudy, dvdx, dvdy = self.grad_disp(y_out)
        fxx = dudx + 1
        fxy = dudy + 0
        fyx = dvdx + 0
        fyy = dvdy + 1
        det_f = fxx * fyy - fxy * fyx
        c11 = fxx * fxx + fyx * fyx
        c12 = fxx * fxy + fyx * fyy
        c21 = fxy * fxx + fyy * fyx
        c22 = fxy * fxy + fyy * fyy
        j = det_f
        trace_c = c11 + c22
        i1 = trace_c
        trace_c2 = c11 * c11 + c12 * c21 + c21 * c12 + c22 * c22
        i2 = 0.5 * (trace_c ** 2 - trace_c2)
        return (self.param_c * (j - 1) ** 2 - self.param_d * jnp.log(j) +
                self.param_c1 * (i1 - 2) + self.param_c2 * (i2 - 1))

    def NeoHookean2D(self, y_out):
        dudx, dudy, dvdx, dvdy = self.grad_disp(y_out)
        fxx = dudx + 1
        fxy = dudy + 0
        fyx = dvdx + 0
        fyy = dvdy + 1
        det_f = fxx * fyy - fxy * fyx
        tr_c = fxx ** 2 + fxy ** 2 + fyx ** 2 + fyy ** 2
        strain_energy = (0.5 * self.lam * (jnp.log(det_f) * jnp.log(det_f)) -
                        self.mu * jnp.log(det_f) + 0.5 * self.mu * (tr_c - 2))
        return jnp.where(jnp.any(jnp.isnan(strain_energy)), 1e9, strain_energy)

    def getStoredEnergy(self, y_out):
        if self.type == 'neohookean':
            if self.dim == 2:
                return self.NeoHookean2D(y_out)
            if self.dim == 3:
                raise ValueError("3D setup is not implemented yet.")
        if self.type == 'mooneyrivlin':
            if self.dim == 2:
                return self.MooneyRivlin2D(y_out)
            if self.dim == 3:
                raise ValueError("3D setup is not implemented yet.")

    def loss_fn(self, params, x, y):

        if self.model_data["normalized"]:
            y_out = self.model.apply(params, x)
            y_out = self.normalizer_y.decode(y_out)
            x_real = self.normalizer_x.decode(x)
            # y_real = self.normalizer_y.decode(y)
        else:
            y_out = self.model.apply(params, x)
            x_real = x
            # y_real = y

        # loss_data = self.rel(y_out, y_real)
        # calculate the interior loss

        # calculate the interior loss
        stored_energy = self.getStoredEnergy(y_out)
        loss_int = (jnp.sum(stored_energy * self.Wint) * (self.L / self.num_x * self.W / self.num_y))

        # calculate the boundary loss
        disp_x, disp_y = y_out[..., 0:1], y_out[..., 1:2]
        disp_y_right = disp_y[:, :, -1, :]
        ty = x_real[:, :, 0, 0:1]
        loss_bnd = jnp.sum(disp_y_right * ty) * (self.W / self.num_y)
        #jax.debug.print("losses: data = {}, DEM: {},  int: {}, bnd: {}",
        #                loss_data, loss_int-loss_bnd, loss_int, loss_bnd)
        return jnp.where(jnp.isnan(jnp.min(stored_energy)), 1e6, loss_int - loss_bnd)
        #return loss_int - loss_bnd


class VinoHyperelasticityLoss(DemHyperelasticityLoss):
    def __init__(self, model, model_data, normalizers, d=2, p=2, size_average=True, reduction=True):
        super().__init__(model, model_data, normalizers, d, p, size_average, reduction)

    @staticmethod
    def difference_disp(y_out):
        u = y_out[..., 0:1]
        v = y_out[..., 1:2]
        dudx = difference_x(u)
        dudy = difference_y(u)
        dvdx = difference_x(v)
        dvdy = difference_y(v)
        return dudx, dudy, dvdx, dvdy

    def loss_fn(self, params, x, y):

        if self.model_data["normalized"]:
            y_out = self.model.apply(params, x)
            y_out = self.normalizer_y.decode(y_out)
            x_real = self.normalizer_x.decode(x)
            # y_real = self.normalizer_y.decode(y)
        else:
            y_out = self.model.apply(params, x)
            x_real = x
            # y_real = y

        dudx, dudy, dvdx, dvdy = self.difference_disp(y_out)

        dudx_0 = dudx[:, 0:-1, 0:-1, :]
        dudx_1 = dudx[:, 1:, 0:-1, :]
        dudy_0 = dudy[:, 0:-1, 0:-1, :]
        dvdx_0 = dvdx[:, 0:-1, 0:-1, :]
        dvdx_1 = dvdx[:, 1:, 0:-1, :]
        dvdy_0 = dvdy[:, 0:-1, 0:-1, :]

        dx = self.L / self.num_x
        dy = self.W / self.num_y
        # n = dy / dx

        c = self.param_c
        c1 = self.param_c1
        c2 = self.param_c2
        d = self.param_d

        # Forth-order Taylor polynomial
        loss_int = jnp.sum(6 * d * dudx_0 ** 4 * dvdx_1 ** 4 + 6 * d * dudx_0 ** 4 * dvdy_0 ** 4 + 6 * d * dudx_1 ** 4 * dvdx_0 ** 4 + 6 * d * dudy_0 ** 4 * dvdx_0 ** 4 + 6 * d * dudx_1 ** 4 * dvdy_0 ** 4 + 6 * d * dudy_0 ** 4 * dvdx_1 ** 4 + 6 * d * dudx_0 ** 4 * dy ** 4 + 6 * d * dvdx_0 ** 4 * dx ** 4 + 6 * d * dudx_1 ** 4 * dy ** 4 + 6 * d * dvdx_1 ** 4 * dx ** 4 + 30 * d * dvdy_0 ** 4 * dx ** 4 + 40 * c * dudx_0 ** 2 * dx ** 2 * dy ** 4 + 40 * c * dudx_1 ** 2 * dx ** 2 * dy ** 4 + 40 * c1 * dudx_0 ** 2 * dx ** 2 * dy ** 4 + 40 * c1 * dudx_0 ** 2 * dx ** 4 * dy ** 2 + 40 * c1 * dudx_1 ** 2 * dx ** 2 * dy ** 4 + 40 * c1 * dudx_1 ** 2 * dx ** 4 * dy ** 2 + 120 * c1 * dudy_0 ** 2 * dx ** 4 * dy ** 2 + 40 * c2 * dudx_0 ** 2 * dx ** 2 * dy ** 4 + 40 * c2 * dudx_1 ** 2 * dx ** 2 * dy ** 4 + 40 * c * dvdx_0 ** 2 * dx ** 4 * dy ** 2 + 40 * c * dvdx_1 ** 2 * dx ** 4 * dy ** 2 + 120 * c * dvdy_0 ** 2 * dx ** 4 * dy ** 2 + 40 * c1 * dvdx_0 ** 2 * dx ** 2 * dy ** 4 + 40 * c1 * dvdx_0 ** 2 * dx ** 4 * dy ** 2 + 40 * c1 * dvdx_1 ** 2 * dx ** 2 * dy ** 4 + 40 * c1 * dvdx_1 ** 2 * dx ** 4 * dy ** 2 + 120 * c1 * dvdy_0 ** 2 * dx ** 4 * dy ** 2 + 40 * c2 * dvdx_0 ** 2 * dx ** 4 * dy ** 2 + 40 * c2 * dvdx_1 ** 2 * dx ** 4 * dy ** 2 + 120 * c2 * dvdy_0 ** 2 * dx ** 4 * dy ** 2 + 6 * d * dudx_0 ** 2 * dudx_1 ** 2 * dvdy_0 ** 4 + 20 * d * dudx_0 ** 2 * dudy_0 ** 2 * dvdx_1 ** 4 + 20 * d * dudx_1 ** 2 * dudy_0 ** 2 * dvdx_0 ** 4 + 20 * d * dudx_0 ** 4 * dvdx_1 ** 2 * dvdy_0 ** 2 + 20 * d * dudx_1 ** 4 * dvdx_0 ** 2 * dvdy_0 ** 2 + 6 * d * dudy_0 ** 4 * dvdx_0 ** 2 * dvdx_1 ** 2 + 6 * d * dudx_0 ** 2 * dudx_1 ** 2 * dy ** 4 + 36 * d * dudx_0 ** 2 * dvdx_1 ** 4 * dx ** 2 + 60 * d * dudx_0 ** 2 * dvdy_0 ** 4 * dx ** 2 + 36 * d * dudx_1 ** 2 * dvdx_0 ** 4 * dx ** 2 + 20 * d * dudy_0 ** 2 * dvdx_0 ** 4 * dx ** 2 + 60 * d * dudx_1 ** 2 * dvdy_0 ** 4 * dx ** 2 + 20 * d * dudy_0 ** 2 * dvdx_1 ** 4 * dx ** 2 + 20 * d * dudx_0 ** 4 * dvdx_1 ** 2 * dy ** 2 + 36 * d * dudx_0 ** 4 * dvdy_0 ** 2 * dy ** 2 + 20 * d * dudx_1 ** 4 * dvdx_0 ** 2 * dy ** 2 + 36 * d * dvdx_0 ** 2 * dvdx_1 ** 2 * dx ** 4 + 60 * d * dvdx_0 ** 2 * dvdy_0 ** 2 * dx ** 4 + 36 * d * dudx_1 ** 4 * dvdy_0 ** 2 * dy ** 2 + 60 * d * dvdx_1 ** 2 * dvdy_0 ** 2 * dx ** 4 + 20 * d * dudx_0 ** 2 * dx ** 2 * dy ** 4 + 20 * d * dudx_1 ** 2 * dx ** 2 * dy ** 4 + 20 * d * dvdx_0 ** 2 * dx ** 4 * dy ** 2 + 20 * d * dvdx_1 ** 2 * dx ** 4 * dy ** 2 + 60 * d * dvdy_0 ** 2 * dx ** 4 * dy ** 2 + 120 * c1 * dudx_0 * dx ** 3 * dy ** 4 + 120 * c1 * dudx_1 * dx ** 3 * dy ** 4 + 120 * c2 * dudx_0 * dx ** 3 * dy ** 4 + 120 * c2 * dudx_1 * dx ** 3 * dy ** 4 - 120 * c1 * dvdx_0 * dx ** 4 * dy ** 3 + 120 * c1 * dvdx_1 * dx ** 4 * dy ** 3 + 240 * c1 * dvdy_0 * dx ** 4 * dy ** 3 - 120 * c2 * dvdx_0 * dx ** 4 * dy ** 3 + 120 * c2 * dvdx_1 * dx ** 4 * dy ** 3 + 240 * c2 * dvdy_0 * dx ** 4 * dy ** 3 + 6 * d * dudx_0 * dudx_1 ** 3 * dvdy_0 ** 4 - 15 * d * dudx_0 * dudy_0 ** 3 * dvdx_1 ** 4 + 15 * d * dudx_1 * dudy_0 ** 3 * dvdx_0 ** 4 + 6 * d * dudx_0 ** 3 * dudx_1 * dvdy_0 ** 4 - 15 * d * dudx_0 ** 3 * dudy_0 * dvdx_1 ** 4 + 15 * d * dudx_1 ** 3 * dudy_0 * dvdx_0 ** 4 + 15 * d * dudx_0 ** 4 * dvdx_1 * dvdy_0 ** 3 + 15 * d * dudx_0 ** 4 * dvdx_1 ** 3 * dvdy_0 - 15 * d * dudx_1 ** 4 * dvdx_0 * dvdy_0 ** 3 - 15 * d * dudx_1 ** 4 * dvdx_0 ** 3 * dvdy_0 + 6 * d * dudy_0 ** 4 * dvdx_0 * dvdx_1 ** 3 + 6 * d * dudy_0 ** 4 * dvdx_0 ** 3 * dvdx_1 + 6 * d * dudx_0 * dudx_1 ** 3 * dy ** 4 + 24 * d * dudx_0 * dvdx_1 ** 4 * dx ** 3 + 60 * d * dudx_0 * dvdy_0 ** 4 * dx ** 3 + 24 * d * dudx_1 * dvdx_0 ** 4 * dx ** 3 + 15 * d * dudy_0 * dvdx_0 ** 4 * dx ** 3 + 6 * d * dudx_0 ** 3 * dudx_1 * dy ** 4 + 24 * d * dudx_0 ** 3 * dvdx_1 ** 4 * dx + 30 * d * dudx_0 ** 3 * dvdy_0 ** 4 * dx + 24 * d * dudx_1 ** 3 * dvdx_0 ** 4 * dx + 15 * d * dudy_0 ** 3 * dvdx_0 ** 4 * dx + 60 * d * dudx_1 * dvdy_0 ** 4 * dx ** 3 - 15 * d * dudy_0 * dvdx_1 ** 4 * dx ** 3 + 30 * d * dudx_1 ** 3 * dvdy_0 ** 4 * dx - 15 * d * dudy_0 ** 3 * dvdx_1 ** 4 * dx - 24 * d * dvdx_0 * dvdx_1 ** 3 * dx ** 4 - 60 * d * dvdx_0 * dvdy_0 ** 3 * dx ** 4 + 15 * d * dudx_0 ** 4 * dvdx_1 * dy ** 3 + 24 * d * dudx_0 ** 4 * dvdy_0 * dy ** 3 + 15 * d * dudx_0 ** 4 * dvdx_1 ** 3 * dy + 24 * d * dudx_0 ** 4 * dvdy_0 ** 3 * dy - 15 * d * dudx_1 ** 4 * dvdx_0 * dy ** 3 - 15 * d * dudx_1 ** 4 * dvdx_0 ** 3 * dy - 24 * d * dvdx_0 ** 3 * dvdx_1 * dx ** 4 - 30 * d * dvdx_0 ** 3 * dvdy_0 * dx ** 4 + 60 * d * dvdx_1 * dvdy_0 ** 3 * dx ** 4 + 24 * d * dudx_1 ** 4 * dvdy_0 * dy ** 3 + 24 * d * dudx_1 ** 4 * dvdy_0 ** 3 * dy + 30 * d * dvdx_1 ** 3 * dvdy_0 * dx ** 4 - 60 * d * dudx_0 * dx ** 3 * dy ** 4 - 10 * d * dudx_0 ** 3 * dx * dy ** 4 - 60 * d * dudx_1 * dx ** 3 * dy ** 4 - 10 * d * dudx_1 ** 3 * dx * dy ** 4 + 60 * d * dvdx_0 * dx ** 4 * dy ** 3 + 10 * d * dvdx_0 ** 3 * dx ** 4 * dy - 60 * d * dvdx_1 * dx ** 4 * dy ** 3 - 120 * d * dvdy_0 * dx ** 4 * dy ** 3 - 10 * d * dvdx_1 ** 3 * dx ** 4 * dy - 40 * d * dvdy_0 ** 3 * dx ** 4 * dy + 40 * c * dudx_0 * dudx_1 * dx ** 2 * dy ** 4 + 40 * c1 * dudx_0 * dudx_1 * dx ** 2 * dy ** 4 - 80 * c1 * dudx_0 * dudx_1 * dx ** 4 * dy ** 2 - 120 * c1 * dudx_0 * dudy_0 * dx ** 4 * dy ** 2 + 120 * c1 * dudx_1 * dudy_0 * dx ** 4 * dy ** 2 + 40 * c2 * dudx_0 * dudx_1 * dx ** 2 * dy ** 4 - 60 * c * dudx_0 * dvdx_0 * dx ** 3 * dy ** 3 + 60 * c * dudx_0 * dvdx_1 * dx ** 3 * dy ** 3 + 120 * c * dudx_0 * dvdy_0 * dx ** 3 * dy ** 3 - 60 * c * dudx_1 * dvdx_0 * dx ** 3 * dy ** 3 + 60 * c * dudx_1 * dvdx_1 * dx ** 3 * dy ** 3 + 120 * c * dudx_1 * dvdy_0 * dx ** 3 * dy ** 3 - 60 * c2 * dudx_0 * dvdx_0 * dx ** 3 * dy ** 3 + 180 * c2 * dudx_0 * dvdx_1 * dx ** 3 * dy ** 3 + 240 * c2 * dudx_0 * dvdy_0 * dx ** 3 * dy ** 3 - 180 * c2 * dudx_1 * dvdx_0 * dx ** 3 * dy ** 3 - 120 * c2 * dudy_0 * dvdx_0 * dx ** 3 * dy ** 3 + 60 * c2 * dudx_1 * dvdx_1 * dx ** 3 * dy ** 3 + 240 * c2 * dudx_1 * dvdy_0 * dx ** 3 * dy ** 3 - 120 * c2 * dudy_0 * dvdx_1 * dx ** 3 * dy ** 3 - 80 * c * dvdx_0 * dvdx_1 * dx ** 4 * dy ** 2 - 120 * c * dvdx_0 * dvdy_0 * dx ** 4 * dy ** 2 + 120 * c * dvdx_1 * dvdy_0 * dx ** 4 * dy ** 2 + 40 * c1 * dvdx_0 * dvdx_1 * dx ** 2 * dy ** 4 - 80 * c1 * dvdx_0 * dvdx_1 * dx ** 4 * dy ** 2 - 120 * c1 * dvdx_0 * dvdy_0 * dx ** 4 * dy ** 2 + 120 * c1 * dvdx_1 * dvdy_0 * dx ** 4 * dy ** 2 - 80 * c2 * dvdx_0 * dvdx_1 * dx ** 4 * dy ** 2 - 120 * c2 * dvdx_0 * dvdy_0 * dx ** 4 * dy ** 2 + 120 * c2 * dvdx_1 * dvdy_0 * dx ** 4 * dy ** 2 - 15 * d * dudx_0 * dudx_1 ** 3 * dvdx_0 * dvdy_0 ** 3 - 24 * d * dudx_0 * dudx_1 ** 3 * dvdx_0 ** 3 * dvdx_1 - 15 * d * dudx_0 * dudx_1 ** 3 * dvdx_0 ** 3 * dvdy_0 - 15 * d * dudx_0 * dudy_0 ** 3 * dvdx_0 * dvdx_1 ** 3 - 15 * d * dudx_0 * dudy_0 ** 3 * dvdx_0 ** 3 * dvdx_1 - 24 * d * dudx_0 * dudy_0 ** 3 * dvdx_0 ** 3 * dvdy_0 - 24 * d * dudx_0 ** 3 * dudx_1 * dvdx_0 * dvdx_1 ** 3 - 15 * d * dudx_0 ** 3 * dudx_1 * dvdx_0 * dvdy_0 ** 3 - 15 * d * dudx_0 ** 3 * dudy_0 * dvdx_0 * dvdx_1 ** 3 - 24 * d * dudx_0 ** 3 * dudy_0 * dvdx_0 * dvdy_0 ** 3 + 15 * d * dudx_0 * dudx_1 ** 3 * dvdx_1 * dvdy_0 ** 3 - 6 * d * dudx_0 * dudy_0 ** 3 * dvdx_1 ** 3 * dvdy_0 + 15 * d * dudx_1 * dudy_0 ** 3 * dvdx_0 * dvdx_1 ** 3 + 15 * d * dudx_1 * dudy_0 ** 3 * dvdx_0 ** 3 * dvdx_1 - 6 * d * dudx_1 * dudy_0 ** 3 * dvdx_0 ** 3 * dvdy_0 + 15 * d * dudx_0 ** 3 * dudx_1 * dvdx_1 * dvdy_0 ** 3 + 15 * d * dudx_0 ** 3 * dudx_1 * dvdx_1 ** 3 * dvdy_0 - 6 * d * dudx_0 ** 3 * dudy_0 * dvdx_1 * dvdy_0 ** 3 - 20 * d * dudx_0 ** 3 * dudy_0 * dvdx_1 ** 3 * dvdy_0 - 6 * d * dudx_1 ** 3 * dudy_0 * dvdx_0 * dvdy_0 ** 3 + 15 * d * dudx_1 ** 3 * dudy_0 * dvdx_0 ** 3 * dvdx_1 - 20 * d * dudx_1 ** 3 * dudy_0 * dvdx_0 ** 3 * dvdy_0 - 24 * d * dudx_1 * dudy_0 ** 3 * dvdx_1 ** 3 * dvdy_0 - 24 * d * dudx_1 ** 3 * dudy_0 * dvdx_1 * dvdy_0 ** 3 + 60 * d * dudx_0 * dudx_1 * dvdy_0 ** 4 * dx ** 2 - 45 * d * dudx_0 * dudy_0 * dvdx_1 ** 4 * dx ** 2 + 30 * d * dudx_0 * dudx_1 ** 2 * dvdy_0 ** 4 * dx)
        loss_int += jnp.sum(40 * d * dudx_0 * dudy_0 ** 2 * dvdx_1 ** 4 * dx + 45 * d * dudx_1 * dudy_0 * dvdx_0 ** 4 * dx ** 2 + 40 * d * dudx_1 * dudy_0 ** 2 * dvdx_0 ** 4 * dx + 30 * d * dudx_0 ** 2 * dudx_1 * dvdy_0 ** 4 * dx - 45 * d * dudx_0 ** 2 * dudy_0 * dvdx_1 ** 4 * dx + 45 * d * dudx_1 ** 2 * dudy_0 * dvdx_0 ** 4 * dx - 72 * d * dudx_0 * dvdx_0 * dvdx_1 ** 3 * dx ** 3 - 90 * d * dudx_0 * dvdx_0 * dvdy_0 ** 3 * dx ** 3 - 15 * d * dudx_0 * dudx_1 ** 3 * dvdx_0 * dy ** 3 - 15 * d * dudx_0 * dudx_1 ** 3 * dvdx_0 ** 3 * dy - 24 * d * dudx_0 * dudy_0 ** 3 * dvdx_0 ** 3 * dy - 24 * d * dudx_0 * dvdx_0 ** 3 * dvdx_1 * dx ** 3 - 15 * d * dudx_0 * dvdx_0 ** 3 * dvdy_0 * dx ** 3 - 15 * d * dudx_0 ** 3 * dudx_1 * dvdx_0 * dy ** 3 - 24 * d * dudx_0 ** 3 * dudy_0 * dvdx_0 * dy ** 3 - 24 * d * dudx_0 ** 3 * dvdx_0 * dvdx_1 ** 3 * dx - 15 * d * dudx_0 ** 3 * dvdx_0 * dvdy_0 ** 3 * dx + 150 * d * dudx_0 * dvdx_1 * dvdy_0 ** 3 * dx ** 3 + 15 * d * dudx_0 * dudx_1 ** 3 * dvdx_1 * dy ** 3 + 24 * d * dudx_0 * dudx_1 ** 3 * dvdy_0 * dy ** 3 + 24 * d * dudx_0 * dudx_1 ** 3 * dvdy_0 ** 3 * dy - 6 * d * dudx_0 * dudy_0 ** 3 * dvdx_1 ** 3 * dy + 105 * d * dudx_0 * dvdx_1 ** 3 * dvdy_0 * dx ** 3 - 24 * d * dudx_1 * dvdx_0 * dvdx_1 ** 3 * dx ** 3 - 150 * d * dudx_1 * dvdx_0 * dvdy_0 ** 3 * dx ** 3 - 6 * d * dudx_1 * dudy_0 ** 3 * dvdx_0 ** 3 * dy - 72 * d * dudx_1 * dvdx_0 ** 3 * dvdx_1 * dx ** 3 - 105 * d * dudx_1 * dvdx_0 ** 3 * dvdy_0 * dx ** 3 + 30 * d * dudy_0 * dvdx_0 * dvdx_1 ** 3 * dx ** 3 - 60 * d * dudy_0 * dvdx_0 * dvdy_0 ** 3 * dx ** 3 - 30 * d * dudy_0 * dvdx_0 ** 3 * dvdx_1 * dx ** 3 - 60 * d * dudy_0 * dvdx_0 ** 3 * dvdy_0 * dx ** 3 + 15 * d * dudx_0 ** 3 * dudx_1 * dvdx_1 * dy ** 3 + 24 * d * dudx_0 ** 3 * dudx_1 * dvdy_0 * dy ** 3 + 15 * d * dudx_0 ** 3 * dudx_1 * dvdx_1 ** 3 * dy + 24 * d * dudx_0 ** 3 * dudx_1 * dvdy_0 ** 3 * dy - 6 * d * dudx_0 ** 3 * dudy_0 * dvdx_1 * dy ** 3 - 20 * d * dudx_0 ** 3 * dudy_0 * dvdx_1 ** 3 * dy + 75 * d * dudx_0 ** 3 * dvdx_1 * dvdy_0 ** 3 * dx + 75 * d * dudx_0 ** 3 * dvdx_1 ** 3 * dvdy_0 * dx - 6 * d * dudx_1 ** 3 * dudy_0 * dvdx_0 * dy ** 3 - 20 * d * dudx_1 ** 3 * dudy_0 * dvdx_0 ** 3 * dy - 75 * d * dudx_1 ** 3 * dvdx_0 * dvdy_0 ** 3 * dx - 24 * d * dudx_1 ** 3 * dvdx_0 ** 3 * dvdx_1 * dx - 75 * d * dudx_1 ** 3 * dvdx_0 ** 3 * dvdy_0 * dx - 30 * d * dudy_0 ** 3 * dvdx_0 ** 3 * dvdy_0 * dx + 90 * d * dudx_1 * dvdx_1 * dvdy_0 ** 3 * dx ** 3 - 24 * d * dudx_1 * dudy_0 ** 3 * dvdx_1 ** 3 * dy + 15 * d * dudx_1 * dvdx_1 ** 3 * dvdy_0 * dx ** 3 - 60 * d * dudy_0 * dvdx_1 * dvdy_0 ** 3 * dx ** 3 - 60 * d * dudy_0 * dvdx_1 ** 3 * dvdy_0 * dx ** 3 - 24 * d * dudx_1 ** 3 * dudy_0 * dvdx_1 * dy ** 3 + 15 * d * dudx_1 ** 3 * dvdx_1 * dvdy_0 ** 3 * dx - 30 * d * dudy_0 ** 3 * dvdx_1 ** 3 * dvdy_0 * dx - 120 * d * dvdx_0 * dvdx_1 * dvdy_0 ** 2 * dx ** 4 - 90 * d * dvdx_0 * dvdx_1 ** 2 * dvdy_0 * dx ** 4 + 45 * d * dudx_0 ** 4 * dvdx_1 * dvdy_0 * dy ** 2 + 45 * d * dudx_0 ** 4 * dvdx_1 * dvdy_0 ** 2 * dy + 40 * d * dudx_0 ** 4 * dvdx_1 ** 2 * dvdy_0 * dy - 45 * d * dudx_1 ** 4 * dvdx_0 * dvdy_0 * dy ** 2 - 45 * d * dudx_1 ** 4 * dvdx_0 * dvdy_0 ** 2 * dy + 40 * d * dudx_1 ** 4 * dvdx_0 ** 2 * dvdy_0 * dy + 90 * d * dvdx_0 ** 2 * dvdx_1 * dvdy_0 * dx ** 4 + 20 * d * dudx_0 * dudx_1 * dx ** 2 * dy ** 4 - 10 * d * dudx_0 * dudx_1 ** 2 * dx * dy ** 4 - 10 * d * dudx_0 ** 2 * dudx_1 * dx * dy ** 4 - 30 * d * dudx_0 * dvdx_0 * dx ** 3 * dy ** 3 - 15 * d * dudx_0 * dvdx_0 ** 3 * dx ** 3 * dy - 15 * d * dudx_0 ** 3 * dvdx_0 * dx * dy ** 3 - 30 * d * dudx_0 * dvdx_1 * dx ** 3 * dy ** 3 - 15 * d * dudx_0 * dvdx_1 ** 3 * dx ** 3 * dy + 30 * d * dudx_1 * dvdx_0 * dx ** 3 * dy ** 3 + 15 * d * dudx_1 * dvdx_0 ** 3 * dx ** 3 * dy + 60 * d * dudy_0 * dvdx_0 * dx ** 3 * dy ** 3 + 20 * d * dudy_0 * dvdx_0 ** 3 * dx ** 3 * dy - 5 * d * dudx_0 ** 3 * dvdx_1 * dx * dy ** 3 + 35 * d * dudx_0 ** 3 * dvdx_1 ** 3 * dx * dy + 80 * d * dudx_0 ** 3 * dvdy_0 ** 3 * dx * dy + 5 * d * dudx_1 ** 3 * dvdx_0 * dx * dy ** 3 - 35 * d * dudx_1 ** 3 * dvdx_0 ** 3 * dx * dy + 10 * d * dudy_0 ** 3 * dvdx_0 ** 3 * dx * dy + 30 * d * dudx_1 * dvdx_1 * dx ** 3 * dy ** 3 + 15 * d * dudx_1 * dvdx_1 ** 3 * dx ** 3 * dy + 60 * d * dudy_0 * dvdx_1 * dx ** 3 * dy ** 3 + 20 * d * dudy_0 * dvdx_1 ** 3 * dx ** 3 * dy + 15 * d * dudx_1 ** 3 * dvdx_1 * dx * dy ** 3 + 80 * d * dudx_1 ** 3 * dvdy_0 ** 3 * dx * dy + 10 * d * dudy_0 ** 3 * dvdx_1 ** 3 * dx * dy - 40 * d * dvdx_0 * dvdx_1 * dx ** 4 * dy ** 2 - 60 * d * dvdx_0 * dvdy_0 * dx ** 4 * dy ** 2 + 30 * d * dvdx_0 * dvdx_1 ** 2 * dx ** 4 * dy)
        loss_int += jnp.sum(60 * d * dvdx_0 * dvdy_0 ** 2 * dx ** 4 * dy - 30 * d * dvdx_0 ** 2 * dvdx_1 * dx ** 4 * dy - 40 * d * dvdx_0 ** 2 * dvdy_0 * dx ** 4 * dy + 60 * d * dvdx_1 * dvdy_0 * dx ** 4 * dy ** 2 - 60 * d * dvdx_1 * dvdy_0 ** 2 * dx ** 4 * dy - 40 * d * dvdx_1 ** 2 * dvdy_0 * dx ** 4 * dy + 80 * c * dudx_0 * dvdx_1 ** 2 * dx ** 3 * dy ** 2 + 120 * c * dudx_0 * dvdy_0 ** 2 * dx ** 3 * dy ** 2 + 80 * c * dudx_1 * dvdx_0 ** 2 * dx ** 3 * dy ** 2 + 60 * c * dudy_0 * dvdx_0 ** 2 * dx ** 3 * dy ** 2 + 60 * c * dudx_0 ** 2 * dvdx_1 * dx ** 2 * dy ** 3 + 80 * c * dudx_0 ** 2 * dvdy_0 * dx ** 2 * dy ** 3 - 60 * c * dudx_1 ** 2 * dvdx_0 * dx ** 2 * dy ** 3 + 120 * c * dudx_1 * dvdy_0 ** 2 * dx ** 3 * dy ** 2 - 60 * c * dudy_0 * dvdx_1 ** 2 * dx ** 3 * dy ** 2 + 80 * c * dudx_1 ** 2 * dvdy_0 * dx ** 2 * dy ** 3 + 80 * c2 * dudx_0 * dvdx_1 ** 2 * dx ** 3 * dy ** 2 + 120 * c2 * dudx_0 * dvdy_0 ** 2 * dx ** 3 * dy ** 2 + 80 * c2 * dudx_1 * dvdx_0 ** 2 * dx ** 3 * dy ** 2 + 60 * c2 * dudy_0 * dvdx_0 ** 2 * dx ** 3 * dy ** 2 + 60 * c2 * dudx_0 ** 2 * dvdx_1 * dx ** 2 * dy ** 3 + 80 * c2 * dudx_0 ** 2 * dvdy_0 * dx ** 2 * dy ** 3 - 60 * c2 * dudx_1 ** 2 * dvdx_0 * dx ** 2 * dy ** 3 + 120 * c2 * dudx_1 * dvdy_0 ** 2 * dx ** 3 * dy ** 2 - 60 * c2 * dudy_0 * dvdx_1 ** 2 * dx ** 3 * dy ** 2 + 80 * c2 * dudx_1 ** 2 * dvdy_0 * dx ** 2 * dy ** 3 + 20 * d * dudx_0 * dudx_1 ** 3 * dvdx_0 ** 2 * dvdy_0 ** 2 - 15 * d * dudx_0 * dudy_0 ** 3 * dvdx_0 ** 2 * dvdx_1 ** 2 - 15 * d * dudx_0 ** 2 * dudx_1 ** 2 * dvdx_0 * dvdy_0 ** 3 + 20 * d * dudx_0 ** 2 * dudy_0 ** 2 * dvdx_0 * dvdx_1 ** 3 + 15 * d * dudx_1 * dudy_0 ** 3 * dvdx_0 ** 2 * dvdx_1 ** 2 + 15 * d * dudx_0 ** 2 * dudx_1 ** 2 * dvdx_1 * dvdy_0 ** 3 + 15 * d * dudx_0 ** 2 * dudy_0 ** 2 * dvdx_1 ** 3 * dvdy_0 + 20 * d * dudx_0 ** 3 * dudx_1 * dvdx_1 ** 2 * dvdy_0 ** 2 - 15 * d * dudx_0 ** 3 * dudy_0 * dvdx_1 ** 2 * dvdy_0 ** 2 + 20 * d * dudx_1 ** 2 * dudy_0 ** 2 * dvdx_0 ** 3 * dvdx_1 - 15 * d * dudx_1 ** 2 * dudy_0 ** 2 * dvdx_0 ** 3 * dvdy_0 + 15 * d * dudx_1 ** 3 * dudy_0 * dvdx_0 ** 2 * dvdy_0 ** 2 + 20 * d * dudx_0 * dudx_1 ** 3 * dvdx_0 ** 2 * dy ** 2 + 72 * d * dudx_0 * dvdx_0 ** 2 * dvdx_1 ** 2 * dx ** 3 + 60 * d * dudx_0 * dvdx_0 ** 2 * dvdy_0 ** 2 * dx ** 3 - 72 * d * dudx_0 ** 2 * dvdx_0 * dvdx_1 ** 3 * dx ** 2 - 60 * d * dudx_0 ** 2 * dvdx_0 * dvdy_0 ** 3 * dx ** 2 - 15 * d * dudx_0 ** 2 * dudx_1 ** 2 * dvdx_0 * dy ** 3 + 36 * d * dudx_0 * dudx_1 ** 3 * dvdy_0 ** 2 * dy ** 2 + 180 * d * dudx_0 * dvdx_1 ** 2 * dvdy_0 ** 2 * dx ** 3 + 72 * d * dudx_1 * dvdx_0 ** 2 * dvdx_1 ** 2 * dx ** 3 + 180 * d * dudx_1 * dvdx_0 ** 2 * dvdy_0 ** 2 * dx ** 3 + 90 * d * dudy_0 * dvdx_0 ** 2 * dvdy_0 ** 2 * dx ** 3 + 150 * d * dudx_0 ** 2 * dvdx_1 * dvdy_0 ** 3 * dx ** 2 + 15 * d * dudx_0 ** 2 * dudx_1 ** 2 * dvdx_1 * dy ** 3 + 24 * d * dudx_0 ** 2 * dudx_1 ** 2 * dvdy_0 * dy ** 3 + 24 * d * dudx_0 ** 2 * dudx_1 ** 2 * dvdy_0 ** 3 * dy + 15 * d * dudx_0 ** 2 * dudy_0 ** 2 * dvdx_1 ** 3 * dy)
        loss_int += jnp.sum(135 * d * dudx_0 ** 2 * dvdx_1 ** 3 * dvdy_0 * dx ** 2 + 20 * d * dudx_0 ** 3 * dudx_1 * dvdx_1 ** 2 * dy ** 2 + 36 * d * dudx_0 ** 3 * dudx_1 * dvdy_0 ** 2 * dy ** 2 - 15 * d * dudx_0 ** 3 * dudy_0 * dvdx_1 ** 2 * dy ** 2 + 100 * d * dudx_0 ** 3 * dvdx_1 ** 2 * dvdy_0 ** 2 * dx - 150 * d * dudx_1 ** 2 * dvdx_0 * dvdy_0 ** 3 * dx ** 2 - 15 * d * dudx_1 ** 2 * dudy_0 ** 2 * dvdx_0 ** 3 * dy - 72 * d * dudx_1 ** 2 * dvdx_0 ** 3 * dvdx_1 * dx ** 2 - 135 * d * dudx_1 ** 2 * dvdx_0 ** 3 * dvdy_0 * dx ** 2 - 20 * d * dudy_0 ** 2 * dvdx_0 * dvdx_1 ** 3 * dx ** 2 - 20 * d * dudy_0 ** 2 * dvdx_0 ** 3 * dvdx_1 * dx ** 2 - 60 * d * dudy_0 ** 2 * dvdx_0 ** 3 * dvdy_0 * dx ** 2 + 15 * d * dudx_1 ** 3 * dudy_0 * dvdx_0 ** 2 * dy ** 2 + 100 * d * dudx_1 ** 3 * dvdx_0 ** 2 * dvdy_0 ** 2 * dx + 60 * d * dudx_1 * dvdx_1 ** 2 * dvdy_0 ** 2 * dx ** 3 - 90 * d * dudy_0 * dvdx_1 ** 2 * dvdy_0 ** 2 * dx ** 3 + 60 * d * dudx_1 ** 2 * dvdx_1 * dvdy_0 ** 3 * dx ** 2 + 60 * d * dudy_0 ** 2 * dvdx_1 ** 3 * dvdy_0 * dx ** 2 - 20 * d * dudx_0 * dvdx_0 ** 2 * dx ** 3 * dy ** 2 + 20 * d * dudx_0 ** 2 * dvdx_0 * dx ** 2 * dy ** 3 + 20 * d * dudx_0 * dvdx_1 ** 2 * dx ** 3 * dy ** 2 + 20 * d * dudx_1 * dvdx_0 ** 2 * dx ** 3 * dy ** 2 + 30 * d * dudy_0 * dvdx_0 ** 2 * dx ** 3 * dy ** 2 + 10 * d * dudx_0 ** 2 * dvdx_1 * dx ** 2 * dy ** 3 + 15 * d * dudx_0 ** 2 * dvdx_1 ** 3 * dx ** 2 * dy + 80 * d * dudx_0 ** 2 * dvdy_0 ** 3 * dx ** 2 * dy + 20 * d * dudx_0 ** 3 * dvdx_1 ** 2 * dx * dy ** 2 + 60 * d * dudx_0 ** 3 * dvdy_0 ** 2 * dx * dy ** 2 - 10 * d * dudx_1 ** 2 * dvdx_0 * dx ** 2 * dy ** 3 - 15 * d * dudx_1 ** 2 * dvdx_0 ** 3 * dx ** 2 * dy + 20 * d * dudy_0 ** 2 * dvdx_0 ** 3 * dx ** 2 * dy + 20 * d * dudx_1 ** 3 * dvdx_0 ** 2 * dx * dy ** 2 - 20 * d * dudx_1 * dvdx_1 ** 2 * dx ** 3 * dy ** 2 - 30 * d * dudy_0 * dvdx_1 ** 2 * dx ** 3 * dy ** 2 - 20 * d * dudx_1 ** 2 * dvdx_1 * dx ** 2 * dy ** 3 + 80 * d * dudx_1 ** 2 * dvdy_0 ** 3 * dx ** 2 * dy - 20 * d * dudy_0 ** 2 * dvdx_1 ** 3 * dx ** 2 * dy + 60 * d * dudx_1 ** 3 * dvdy_0 ** 2 * dx * dy ** 2 + 40 * c * dudx_0 ** 2 * dvdx_1 ** 2 * dx ** 2 * dy ** 2 + 40 * c * dudx_0 ** 2 * dvdy_0 ** 2 * dx ** 2 * dy ** 2 + 40 * c * dudx_1 ** 2 * dvdx_0 ** 2 * dx ** 2 * dy ** 2 + 40 * c * dudy_0 ** 2 * dvdx_0 ** 2 * dx ** 2 * dy ** 2 + 40 * c * dudx_1 ** 2 * dvdy_0 ** 2 * dx ** 2 * dy ** 2 + 40 * c * dudy_0 ** 2 * dvdx_1 ** 2 * dx ** 2 * dy ** 2 + 40 * c2 * dudx_0 ** 2 * dvdx_1 ** 2 * dx ** 2 * dy ** 2 + 40 * c2 * dudx_0 ** 2 * dvdy_0 ** 2 * dx ** 2 * dy ** 2 + 40 * c2 * dudx_1 ** 2 * dvdx_0 ** 2 * dx ** 2 * dy ** 2 + 40 * c2 * dudy_0 ** 2 * dvdx_0 ** 2 * dx ** 2 * dy ** 2 + 40 * c2 * dudx_1 ** 2 * dvdy_0 ** 2 * dx ** 2 * dy ** 2 + 40 * c2 * dudy_0 ** 2 * dvdx_1 ** 2 * dx ** 2 * dy ** 2 + 36 * d * dudx_0 ** 2 * dudx_1 ** 2 * dvdx_0 ** 2 * dvdx_1 ** 2 + 20 * d * dudx_0 ** 2 * dudx_1 ** 2 * dvdx_0 ** 2 * dvdy_0 ** 2 + 20 * d * dudx_0 ** 2 * dudy_0 ** 2 * dvdx_0 ** 2 * dvdx_1 ** 2 + 36 * d * dudx_0 ** 2 * dudy_0 ** 2 * dvdx_0 ** 2 * dvdy_0 ** 2 + 20 * d * dudx_0 ** 2 * dudx_1 ** 2 * dvdx_1 ** 2 * dvdy_0 ** 2 + 6 * d * dudx_0 ** 2 * dudy_0 ** 2 * dvdx_1 ** 2 * dvdy_0 ** 2 + 20 * d * dudx_1 ** 2 * dudy_0 ** 2 * dvdx_0 ** 2 * dvdx_1 ** 2 + 6 * d * dudx_1 ** 2 * dudy_0 ** 2 * dvdx_0 ** 2 * dvdy_0 ** 2 + 36 * d * dudx_1 ** 2 * dudy_0 ** 2 * dvdx_1 ** 2 * dvdy_0 ** 2 + 20 * d * dudx_0 ** 2 * dudx_1 ** 2 * dvdx_0 ** 2 * dy ** 2 + 36 * d * dudx_0 ** 2 * dudy_0 ** 2 * dvdx_0 ** 2 * dy ** 2 + 36 * d * dudx_0 ** 2 * dvdx_0 ** 2 * dvdx_1 ** 2 * dx ** 2 + 20 * d * dudx_0 ** 2 * dvdx_0 ** 2 * dvdy_0 ** 2 * dx ** 2 + 20 * d * dudx_0 ** 2 * dudx_1 ** 2 * dvdx_1 ** 2 * dy ** 2 + 36 * d * dudx_0 ** 2 * dudx_1 ** 2 * dvdy_0 ** 2 * dy ** 2 + 6 * d * dudx_0 ** 2 * dudy_0 ** 2 * dvdx_1 ** 2 * dy ** 2 + 200 * d * dudx_0 ** 2 * dvdx_1 ** 2 * dvdy_0 ** 2 * dx ** 2 + 6 * d * dudx_1 ** 2 * dudy_0 ** 2 * dvdx_0 ** 2 * dy ** 2 + 36 * d * dudx_1 ** 2 * dvdx_0 ** 2 * dvdx_1 ** 2 * dx ** 2 + 200 * d * dudx_1 ** 2 * dvdx_0 ** 2 * dvdy_0 ** 2 * dx ** 2 + 60 * d * dudy_0 ** 2 * dvdx_0 ** 2 * dvdy_0 ** 2 * dx ** 2 + 36 * d * dudx_1 ** 2 * dudy_0 ** 2 * dvdx_1 ** 2 * dy ** 2 + 20 * d * dudx_1 ** 2 * dvdx_1 ** 2 * dvdy_0 ** 2 * dx ** 2 + 60 * d * dudy_0 ** 2 * dvdx_1 ** 2 * dvdy_0 ** 2 * dx ** 2 + 20 * d * dudx_0 ** 2 * dvdx_0 ** 2 * dx ** 2 * dy ** 2 + 20 * d * dudy_0 ** 2 * dvdx_0 ** 2 * dx ** 2 * dy ** 2 + 20 * d * dudx_1 ** 2 * dvdx_1 ** 2 * dx ** 2 * dy ** 2 + 20 * d * dudy_0 ** 2 * dvdx_1 ** 2 * dx ** 2 * dy ** 2 + 80 * d * dvdx_0 * dvdx_1 * dvdy_0 * dx ** 4 * dy - 60 * c * dudx_0 * dudx_1 * dvdx_0 * dx ** 2 * dy ** 3 - 80 * c * dudx_0 * dudy_0 * dvdx_0 * dx ** 2 * dy ** 3)
        loss_int += jnp.sum(60 * c * dudx_0 * dudx_1 * dvdx_1 * dx ** 2 * dy ** 3 + 80 * c * dudx_0 * dudx_1 * dvdy_0 * dx ** 2 * dy ** 3 - 40 * c * dudx_0 * dudy_0 * dvdx_1 * dx ** 2 * dy ** 3 - 40 * c * dudx_1 * dudy_0 * dvdx_0 * dx ** 2 * dy ** 3 - 80 * c * dudx_1 * dudy_0 * dvdx_1 * dx ** 2 * dy ** 3 - 60 * c2 * dudx_0 * dudx_1 * dvdx_0 * dx ** 2 * dy ** 3 - 80 * c2 * dudx_0 * dudy_0 * dvdx_0 * dx ** 2 * dy ** 3 + 60 * c2 * dudx_0 * dudx_1 * dvdx_1 * dx ** 2 * dy ** 3 + 80 * c2 * dudx_0 * dudx_1 * dvdy_0 * dx ** 2 * dy ** 3 - 40 * c2 * dudx_0 * dudy_0 * dvdx_1 * dx ** 2 * dy ** 3 - 40 * c2 * dudx_1 * dudy_0 * dvdx_0 * dx ** 2 * dy ** 3 - 80 * c2 * dudx_1 * dudy_0 * dvdx_1 * dx ** 2 * dy ** 3 - 80 * c * dudx_0 * dvdx_0 * dvdx_1 * dx ** 3 * dy ** 2 - 60 * c * dudx_0 * dvdx_0 * dvdy_0 * dx ** 3 * dy ** 2 + 180 * c * dudx_0 * dvdx_1 * dvdy_0 * dx ** 3 * dy ** 2 - 80 * c * dudx_1 * dvdx_0 * dvdx_1 * dx ** 3 * dy ** 2 - 180 * c * dudx_1 * dvdx_0 * dvdy_0 * dx ** 3 * dy ** 2 - 120 * c * dudy_0 * dvdx_0 * dvdy_0 * dx ** 3 * dy ** 2 + 60 * c * dudx_1 * dvdx_1 * dvdy_0 * dx ** 3 * dy ** 2 - 120 * c * dudy_0 * dvdx_1 * dvdy_0 * dx ** 3 * dy ** 2 - 80 * c2 * dudx_0 * dvdx_0 * dvdx_1 * dx ** 3 * dy ** 2 - 60 * c2 * dudx_0 * dvdx_0 * dvdy_0 * dx ** 3 * dy ** 2 + 180 * c2 * dudx_0 * dvdx_1 * dvdy_0 * dx ** 3 * dy ** 2 - 80 * c2 * dudx_1 * dvdx_0 * dvdx_1 * dx ** 3 * dy ** 2 - 180 * c2 * dudx_1 * dvdx_0 * dvdy_0 * dx ** 3 * dy ** 2 - 120 * c2 * dudy_0 * dvdx_0 * dvdy_0 * dx ** 3 * dy ** 2 + 60 * c2 * dudx_1 * dvdx_1 * dvdy_0 * dx ** 3 * dy ** 2 - 120 * c2 * dudy_0 * dvdx_1 * dvdy_0 * dx ** 3 * dy ** 2 - 40 * d * dudx_0 * dudx_1 * dudy_0 ** 2 * dvdx_0 * dvdx_1 ** 3 - 40 * d * dudx_0 * dudx_1 * dudy_0 ** 2 * dvdx_0 ** 3 * dvdx_1 - 45 * d * dudx_0 * dudx_1 * dudy_0 ** 2 * dvdx_0 ** 3 * dvdy_0 - 12 * d * dudx_0 * dudx_1 ** 2 * dudy_0 * dvdx_0 * dvdy_0 ** 3 - 45 * d * dudx_0 * dudx_1 ** 2 * dudy_0 * dvdx_0 ** 3 * dvdx_1 - 40 * d * dudx_0 * dudx_1 ** 2 * dudy_0 * dvdx_0 ** 3 * dvdy_0 + 45 * d * dudx_0 ** 2 * dudx_1 * dudy_0 * dvdx_0 * dvdx_1 ** 3 - 18 * d * dudx_0 ** 2 * dudx_1 * dudy_0 * dvdx_0 * dvdy_0 ** 3 + 45 * d * dudx_0 * dudx_1 * dudy_0 ** 2 * dvdx_1 ** 3 * dvdy_0 - 18 * d * dudx_0 * dudx_1 ** 2 * dudy_0 * dvdx_1 * dvdy_0 ** 3 - 12 * d * dudx_0 ** 2 * dudx_1 * dudy_0 * dvdx_1 * dvdy_0 ** 3 - 40 * d * dudx_0 ** 2 * dudx_1 * dudy_0 * dvdx_1 ** 3 * dvdy_0 - 40 * d * dudx_0 * dudx_1 ** 3 * dvdx_0 * dvdx_1 * dvdy_0 ** 2 + 45 * d * dudx_0 * dudx_1 ** 3 * dvdx_0 ** 2 * dvdx_1 * dvdy_0 - 12 * d * dudx_0 * dudy_0 ** 3 * dvdx_0 * dvdx_1 ** 2 * dvdy_0 - 18 * d * dudx_0 * dudy_0 ** 3 * dvdx_0 ** 2 * dvdx_1 * dvdy_0 - 40 * d * dudx_0 ** 3 * dudx_1 * dvdx_0 * dvdx_1 * dvdy_0 ** 2 - 45 * d * dudx_0 ** 3 * dudx_1 * dvdx_0 * dvdx_1 ** 2 * dvdy_0 - 45 * d * dudx_0 ** 3 * dudy_0 * dvdx_0 * dvdx_1 * dvdy_0 ** 2 - 40 * d * dudx_0 ** 3 * dudy_0 * dvdx_0 * dvdx_1 ** 2 * dvdy_0 - 18 * d * dudx_1 * dudy_0 ** 3 * dvdx_0 * dvdx_1 ** 2 * dvdy_0 - 12 * d * dudx_1 * dudy_0 ** 3 * dvdx_0 ** 2 * dvdx_1 * dvdy_0 + 45 * d * dudx_1 ** 3 * dudy_0 * dvdx_0 * dvdx_1 * dvdy_0 ** 2 - 40 * d * dudx_1 ** 3 * dudy_0 * dvdx_0 ** 2 * dvdx_1 * dvdy_0 - 72 * d * dudx_0 * dudx_1 * dvdx_0 * dvdx_1 ** 3 * dx ** 2 - 150 * d * dudx_0 * dudx_1 * dvdx_0 * dvdy_0 ** 3 * dx ** 2 - 45 * d * dudx_0 * dudx_1 * dudy_0 ** 2 * dvdx_0 ** 3 * dy - 72 * d * dudx_0 * dudx_1 * dvdx_0 ** 3 * dvdx_1 * dx ** 2 - 45 * d * dudx_0 * dudx_1 * dvdx_0 ** 3 * dvdy_0 * dx ** 2 + 45 * d * dudx_0 * dudy_0 * dvdx_0 * dvdx_1 ** 3 * dx ** 2 - 120 * d * dudx_0 * dudy_0 * dvdx_0 * dvdy_0 ** 3 * dx ** 2 - 45 * d * dudx_0 * dudy_0 * dvdx_0 ** 3 * dvdx_1 * dx ** 2 - 40 * d * dudx_0 * dudy_0 * dvdx_0 ** 3 * dvdy_0 * dx ** 2 - 12 * d * dudx_0 * dudx_1 ** 2 * dudy_0 * dvdx_0 * dy ** 3 - 40 * d * dudx_0 * dudx_1 ** 2 * dudy_0 * dvdx_0 ** 3 * dy - 75 * d * dudx_0 * dudx_1 ** 2 * dvdx_0 * dvdy_0 ** 3 * dx - 72 * d * dudx_0 * dudx_1 ** 2 * dvdx_0 ** 3 * dvdx_1 * dx - 45 * d * dudx_0 * dudx_1 ** 2 * dvdx_0 ** 3 * dvdy_0 * dx - 40 * d * dudx_0 * dudy_0 ** 2 * dvdx_0 ** 3 * dvdx_1 * dx - 45 * d * dudx_0 * dudy_0 ** 2 * dvdx_0 ** 3 * dvdy_0 * dx - 18 * d * dudx_0 ** 2 * dudx_1 * dudy_0 * dvdx_0 * dy ** 3 - 72 * d * dudx_0 ** 2 * dudx_1 * dvdx_0 * dvdx_1 ** 3 * dx - 75 * d * dudx_0 ** 2 * dudx_1 * dvdx_0 * dvdy_0 ** 3 * dx - 90 * d * dudx_0 ** 2 * dudy_0 * dvdx_0 * dvdy_0 ** 3 * dx + 150 * d * dudx_0 * dudx_1 * dvdx_1 * dvdy_0 ** 3 * dx ** 2 + 45 * d * dudx_0 * dudx_1 * dudy_0 ** 2 * dvdx_1 ** 3 * dy + 45 * d * dudx_0 * dudx_1 * dvdx_1 ** 3 * dvdy_0 * dx ** 2 - 60 * d * dudx_0 * dudy_0 * dvdx_1 * dvdy_0 ** 3 * dx ** 2 - 140 * d * dudx_0 * dudy_0 * dvdx_1 ** 3 * dvdy_0 * dx ** 2 - 18 * d * dudx_0 * dudx_1 ** 2 * dudy_0 * dvdx_1 * dy ** 3 + 75 * d * dudx_0 * dudx_1 ** 2 * dvdx_1 * dvdy_0 ** 3 * dx + 75 * d * dudx_0 * dudy_0 ** 2 * dvdx_1 ** 3 * dvdy_0 * dx + 45 * d * dudx_1 * dudy_0 * dvdx_0 * dvdx_1 ** 3 * dx ** 2 - 60 * d * dudx_1 * dudy_0 * dvdx_0 * dvdy_0 ** 3 * dx ** 2 - 45 * d * dudx_1 * dudy_0 * dvdx_0 ** 3 * dvdx_1 * dx ** 2 - 140 * d * dudx_1 * dudy_0 * dvdx_0 ** 3 * dvdy_0 * dx ** 2 - 40 * d * dudx_1 * dudy_0 ** 2 * dvdx_0 * dvdx_1 ** 3 * dx - 75 * d * dudx_1 * dudy_0 ** 2 * dvdx_0 ** 3 * dvdy_0 * dx - 12 * d * dudx_0 ** 2 * dudx_1 * dudy_0 * dvdx_1 * dy ** 3 - 40 * d * dudx_0 ** 2 * dudx_1 * dudy_0 * dvdx_1 ** 3 * dy + 75 * d * dudx_0 ** 2 * dudx_1 * dvdx_1 * dvdy_0 ** 3 * dx + 45 * d * dudx_0 ** 2 * dudx_1 * dvdx_1 ** 3 * dvdy_0 * dx - 30 * d * dudx_0 ** 2 * dudy_0 * dvdx_1 * dvdy_0 ** 3 * dx - 100 * d * dudx_0 ** 2 * dudy_0 * dvdx_1 ** 3 * dvdy_0 * dx - 30 * d * dudx_1 ** 2 * dudy_0 * dvdx_0 * dvdy_0 ** 3 * dx - 100 * d * dudx_1 ** 2 * dudy_0 * dvdx_0 ** 3 * dvdy_0 * dx - 120 * d * dudx_1 * dudy_0 * dvdx_1 * dvdy_0 ** 3 * dx ** 2 - 40 * d * dudx_1 * dudy_0 * dvdx_1 ** 3 * dvdy_0 * dx ** 2)
        loss_int += jnp.sum(45 * d * dudx_1 * dudy_0 ** 2 * dvdx_1 ** 3 * dvdy_0 * dx - 90 * d * dudx_1 ** 2 * dudy_0 * dvdx_1 * dvdy_0 ** 3 * dx - 240 * d * dudx_0 * dvdx_0 * dvdx_1 * dvdy_0 ** 2 * dx ** 3 - 225 * d * dudx_0 * dvdx_0 * dvdx_1 ** 2 * dvdy_0 * dx ** 3 - 40 * d * dudx_0 * dudx_1 ** 3 * dvdx_0 * dvdx_1 * dy ** 2 - 45 * d * dudx_0 * dudx_1 ** 3 * dvdx_0 * dvdy_0 * dy ** 2 - 45 * d * dudx_0 * dudx_1 ** 3 * dvdx_0 * dvdy_0 ** 2 * dy + 45 * d * dudx_0 * dudx_1 ** 3 * dvdx_0 ** 2 * dvdx_1 * dy + 40 * d * dudx_0 * dudx_1 ** 3 * dvdx_0 ** 2 * dvdy_0 * dy - 12 * d * dudx_0 * dudy_0 ** 3 * dvdx_0 * dvdx_1 ** 2 * dy - 18 * d * dudx_0 * dudy_0 ** 3 * dvdx_0 ** 2 * dvdx_1 * dy + 135 * d * dudx_0 * dvdx_0 ** 2 * dvdx_1 * dvdy_0 * dx ** 3 - 40 * d * dudx_0 ** 3 * dudx_1 * dvdx_0 * dvdx_1 * dy ** 2 - 45 * d * dudx_0 ** 3 * dudx_1 * dvdx_0 * dvdy_0 * dy ** 2 - 45 * d * dudx_0 ** 3 * dudx_1 * dvdx_0 * dvdx_1 ** 2 * dy - 45 * d * dudx_0 ** 3 * dudx_1 * dvdx_0 * dvdy_0 ** 2 * dy - 45 * d * dudx_0 ** 3 * dudy_0 * dvdx_0 * dvdx_1 * dy ** 2 - 72 * d * dudx_0 ** 3 * dudy_0 * dvdx_0 * dvdy_0 * dy ** 2 - 40 * d * dudx_0 ** 3 * dudy_0 * dvdx_0 * dvdx_1 ** 2 * dy - 72 * d * dudx_0 ** 3 * dudy_0 * dvdx_0 * dvdy_0 ** 2 * dy - 40 * d * dudx_0 ** 3 * dvdx_0 * dvdx_1 * dvdy_0 ** 2 * dx - 45 * d * dudx_0 ** 3 * dvdx_0 * dvdx_1 ** 2 * dvdy_0 * dx + 45 * d * dudx_0 * dudx_1 ** 3 * dvdx_1 * dvdy_0 * dy ** 2 + 45 * d * dudx_0 * dudx_1 ** 3 * dvdx_1 * dvdy_0 ** 2 * dy - 240 * d * dudx_1 * dvdx_0 * dvdx_1 * dvdy_0 ** 2 * dx ** 3 - 135 * d * dudx_1 * dvdx_0 * dvdx_1 ** 2 * dvdy_0 * dx ** 3 - 18 * d * dudx_1 * dudy_0 ** 3 * dvdx_0 * dvdx_1 ** 2 * dy - 12 * d * dudx_1 * dudy_0 ** 3 * dvdx_0 ** 2 * dvdx_1 * dy + 225 * d * dudx_1 * dvdx_0 ** 2 * dvdx_1 * dvdy_0 * dx ** 3 + 60 * d * dudy_0 * dvdx_0 * dvdx_1 ** 2 * dvdy_0 * dx ** 3 + 60 * d * dudy_0 * dvdx_0 ** 2 * dvdx_1 * dvdy_0 * dx ** 3 + 45 * d * dudx_0 ** 3 * dudx_1 * dvdx_1 * dvdy_0 * dy ** 2 + 45 * d * dudx_0 ** 3 * dudx_1 * dvdx_1 * dvdy_0 ** 2 * dy + 40 * d * dudx_0 ** 3 * dudx_1 * dvdx_1 ** 2 * dvdy_0 * dy - 18 * d * dudx_0 ** 3 * dudy_0 * dvdx_1 * dvdy_0 * dy ** 2 - 18 * d * dudx_0 ** 3 * dudy_0 * dvdx_1 * dvdy_0 ** 2 * dy - 30 * d * dudx_0 ** 3 * dudy_0 * dvdx_1 ** 2 * dvdy_0 * dy + 45 * d * dudx_1 ** 3 * dudy_0 * dvdx_0 * dvdx_1 * dy ** 2 - 18 * d * dudx_1 ** 3 * dudy_0 * dvdx_0 * dvdy_0 * dy ** 2 - 18 * d * dudx_1 ** 3 * dudy_0 * dvdx_0 * dvdy_0 ** 2 * dy - 40 * d * dudx_1 ** 3 * dudy_0 * dvdx_0 ** 2 * dvdx_1 * dy + 30 * d * dudx_1 ** 3 * dudy_0 * dvdx_0 ** 2 * dvdy_0 * dy - 40 * d * dudx_1 ** 3 * dvdx_0 * dvdx_1 * dvdy_0 ** 2 * dx + 45 * d * dudx_1 ** 3 * dvdx_0 ** 2 * dvdx_1 * dvdy_0 * dx - 30 * d * dudy_0 ** 3 * dvdx_0 * dvdx_1 ** 2 * dvdy_0 * dx - 30 * d * dudy_0 ** 3 * dvdx_0 ** 2 * dvdx_1 * dvdy_0 * dx - 72 * d * dudx_1 ** 3 * dudy_0 * dvdx_1 * dvdy_0 * dy ** 2 - 72 * d * dudx_1 ** 3 * dudy_0 * dvdx_1 * dvdy_0 ** 2 * dy - 10 * d * dudx_0 * dudx_1 * dvdx_0 * dx ** 2 * dy ** 3 - 45 * d * dudx_0 * dudx_1 * dvdx_0 ** 3 * dx ** 2 * dy - 40 * d * dudx_0 * dudy_0 * dvdx_0 * dx ** 2 * dy ** 3 - 40 * d * dudx_0 * dudy_0 * dvdx_0 ** 3 * dx ** 2 * dy + 5 * d * dudx_0 * dudx_1 ** 2 * dvdx_0 * dx * dy ** 3 - 45 * d * dudx_0 * dudx_1 ** 2 * dvdx_0 ** 3 * dx * dy - 45 * d * dudx_0 * dudy_0 ** 2 * dvdx_0 ** 3 * dx * dy + 5 * d * dudx_0 ** 2 * dudx_1 * dvdx_0 * dx * dy ** 3 + 30 * d * dudx_0 ** 2 * dudy_0 * dvdx_0 * dx * dy ** 3 + 10 * d * dudx_0 * dudx_1 * dvdx_1 * dx ** 2 * dy ** 3 + 45 * d * dudx_0 * dudx_1 * dvdx_1 ** 3 * dx ** 2 * dy + 80 * d * dudx_0 * dudx_1 * dvdy_0 ** 3 * dx ** 2 * dy - 20 * d * dudx_0 * dudy_0 * dvdx_1 * dx ** 2 * dy ** 3 + 20 * d * dudx_0 * dudy_0 * dvdx_1 ** 3 * dx ** 2 * dy - 5 * d * dudx_0 * dudx_1 ** 2 * dvdx_1 * dx * dy ** 3 + 80 * d * dudx_0 * dudx_1 ** 2 * dvdy_0 ** 3 * dx * dy - 5 * d * dudx_0 * dudy_0 ** 2 * dvdx_1 ** 3 * dx * dy - 20 * d * dudx_1 * dudy_0 * dvdx_0 * dx ** 2 * dy ** 3 + 20 * d * dudx_1 * dudy_0 * dvdx_0 ** 3 * dx ** 2 * dy + 5 * d * dudx_1 * dudy_0 ** 2 * dvdx_0 ** 3 * dx * dy - 5 * d * dudx_0 ** 2 * dudx_1 * dvdx_1 * dx * dy ** 3 + 45 * d * dudx_0 ** 2 * dudx_1 * dvdx_1 ** 3 * dx * dy + 80 * d * dudx_0 ** 2 * dudx_1 * dvdy_0 ** 3 * dx * dy + 10 * d * dudx_0 ** 2 * dudy_0 * dvdx_1 * dx * dy ** 3 - 20 * d * dudx_0 ** 2 * dudy_0 * dvdx_1 ** 3 * dx * dy + 10 * d * dudx_1 ** 2 * dudy_0 * dvdx_0 * dx * dy ** 3 - 20 * d * dudx_1 ** 2 * dudy_0 * dvdx_0 ** 3 * dx * dy - 40 * d * dudx_1 * dudy_0 * dvdx_1 * dx ** 2 * dy ** 3 - 40 * d * dudx_1 * dudy_0 * dvdx_1 ** 3 * dx ** 2 * dy + 45 * d * dudx_1 * dudy_0 ** 2 * dvdx_1 ** 3 * dx * dy + 30 * d * dudx_1 ** 2 * dudy_0 * dvdx_1 * dx * dy ** 3 + 30 * d * dudx_0 * dvdx_0 * dvdy_0 * dx ** 3 * dy ** 2 + 15 * d * dudx_0 * dvdx_0 * dvdx_1 ** 2 * dx ** 3 * dy - 30 * d * dudx_0 * dvdx_0 * dvdy_0 ** 2 * dx ** 3 * dy + 15 * d * dudx_0 * dvdx_0 ** 2 * dvdx_1 * dx ** 3 * dy + 40 * d * dudx_0 * dvdx_0 ** 2 * dvdy_0 * dx ** 3 * dy - 40 * d * dudx_0 ** 3 * dvdx_0 * dvdx_1 * dx * dy ** 2 - 45 * d * dudx_0 ** 3 * dvdx_0 * dvdy_0 * dx * dy ** 2 - 45 * d * dudx_0 ** 3 * dvdx_0 * dvdx_1 ** 2 * dx * dy - 45 * d * dudx_0 ** 3 * dvdx_0 * dvdy_0 ** 2 * dx * dy + 30 * d * dudx_0 * dvdx_1 * dvdy_0 * dx ** 3 * dy ** 2 - 30 * d * dudx_0 * dvdx_1 * dvdy_0 ** 2 * dx ** 3 * dy - 40 * d * dudx_0 * dvdx_1 ** 2 * dvdy_0 * dx ** 3 * dy - 30 * d * dudx_1 * dvdx_0 * dvdy_0 * dx ** 3 * dy ** 2 - 15 * d * dudx_1 * dvdx_0 * dvdx_1 ** 2 * dx ** 3 * dy + 30 * d * dudx_1 * dvdx_0 * dvdy_0 ** 2 * dx ** 3 * dy - 15 * d * dudx_1 * dvdx_0 ** 2 * dvdx_1 * dx ** 3 * dy - 40 * d * dudx_1 * dvdx_0 ** 2 * dvdy_0 * dx ** 3 * dy - 60 * d * dudy_0 * dvdx_0 * dvdy_0 * dx ** 3 * dy ** 2 - 20 * d * dudy_0 * dvdx_0 * dvdx_1 ** 2 * dx ** 3 * dy + 60 * d * dudy_0 * dvdx_0 * dvdy_0 ** 2 * dx ** 3 * dy - 20 * d * dudy_0 * dvdx_0 ** 2 * dvdx_1 * dx ** 3 * dy - 60 * d * dudy_0 * dvdx_0 ** 2 * dvdy_0 * dx ** 3 * dy + 65 * d * dudx_0 ** 3 * dvdx_1 * dvdy_0 * dx * dy ** 2 + 145 * d * dudx_0 ** 3 * dvdx_1 * dvdy_0 ** 2 * dx * dy + 120 * d * dudx_0 ** 3 * dvdx_1 ** 2 * dvdy_0 * dx * dy - 40 * d * dudx_1 ** 3 * dvdx_0 * dvdx_1 * dx * dy ** 2 - 65 * d * dudx_1 ** 3 * dvdx_0 * dvdy_0 * dx * dy ** 2 - 145 * d * dudx_1 ** 3 * dvdx_0 * dvdy_0 ** 2 * dx * dy + 45 * d * dudx_1 ** 3 * dvdx_0 ** 2 * dvdx_1 * dx * dy + 120 * d * dudx_1 ** 3 * dvdx_0 ** 2 * dvdy_0 * dx * dy + 10 * d * dudy_0 ** 3 * dvdx_0 * dvdx_1 ** 2 * dx * dy + 10 * d * dudy_0 ** 3 * dvdx_0 ** 2 * dvdx_1 * dx * dy - 30 * d * dudx_1 * dvdx_1 * dvdy_0 * dx ** 3 * dy ** 2 + 30 * d * dudx_1 * dvdx_1 * dvdy_0 ** 2 * dx ** 3 * dy + 40 * d * dudx_1 * dvdx_1 ** 2 * dvdy_0 * dx ** 3 * dy - 60 * d * dudy_0 * dvdx_1 * dvdy_0 * dx ** 3 * dy ** 2 + 60 * d * dudy_0 * dvdx_1 * dvdy_0 ** 2 * dx ** 3 * dy + 60 * d * dudy_0 * dvdx_1 ** 2 * dvdy_0 * dx ** 3 * dy + 45 * d * dudx_1 ** 3 * dvdx_1 * dvdy_0 * dx * dy ** 2 + 45 * d * dudx_1 ** 3 * dvdx_1 * dvdy_0 ** 2 * dx * dy + 40 * c * dudx_0 * dudx_1 * dvdy_0 ** 2 * dx ** 2 * dy ** 2 - 60 * c * dudx_0 * dudy_0 * dvdx_1 ** 2 * dx ** 2 * dy ** 2 + 60 * c * dudx_1 * dudy_0 * dvdx_0 ** 2 * dx ** 2 * dy ** 2 + 40 * c2 * dudx_0 * dudx_1 * dvdy_0 ** 2 * dx ** 2 * dy ** 2 - 60 * c2 * dudx_0 * dudy_0 * dvdx_1 ** 2 * dx ** 2 * dy ** 2 + 60 * c2 * dudx_1 * dudy_0 * dvdx_0 ** 2 * dx ** 2 * dy ** 2 + 60 * c * dudx_0 ** 2 * dvdx_1 * dvdy_0 * dx ** 2 * dy ** 2 - 60 * c * dudx_1 ** 2 * dvdx_0 * dvdy_0 * dx ** 2 * dy ** 2 + 40 * c * dudy_0 ** 2 * dvdx_0 * dvdx_1 * dx ** 2 * dy ** 2 + 60 * c2 * dudx_0 ** 2 * dvdx_1 * dvdy_0 * dx ** 2 * dy ** 2 - 60 * c2 * dudx_1 ** 2 * dvdx_0 * dvdy_0 * dx ** 2 * dy ** 2 + 40 * c2 * dudy_0 ** 2 * dvdx_0 * dvdx_1 * dx ** 2 * dy ** 2 - 40 * d * dudx_0 * dudx_1 * dudy_0 ** 2 * dvdx_0 ** 2 * dvdx_1 ** 2 + 18 * d * dudx_0 * dudx_1 * dudy_0 ** 2 * dvdx_0 ** 2 * dvdy_0 ** 2 - 45 * d * dudx_0 * dudx_1 ** 2 * dudy_0 * dvdx_0 ** 2 * dvdx_1 ** 2)
        loss_int += jnp.sum(30 * d * dudx_0 * dudx_1 ** 2 * dudy_0 * dvdx_0 ** 2 * dvdy_0 ** 2 + 45 * d * dudx_0 ** 2 * dudx_1 * dudy_0 * dvdx_0 ** 2 * dvdx_1 ** 2 + 45 * d * dudx_0 ** 2 * dudx_1 * dudy_0 * dvdx_0 ** 2 * dvdy_0 ** 2 + 18 * d * dudx_0 * dudx_1 * dudy_0 ** 2 * dvdx_1 ** 2 * dvdy_0 ** 2 - 45 * d * dudx_0 * dudx_1 ** 2 * dudy_0 * dvdx_1 ** 2 * dvdy_0 ** 2 - 30 * d * dudx_0 ** 2 * dudx_1 * dudy_0 * dvdx_1 ** 2 * dvdy_0 ** 2 - 40 * d * dudx_0 ** 2 * dudx_1 ** 2 * dvdx_0 * dvdx_1 * dvdy_0 ** 2 - 45 * d * dudx_0 ** 2 * dudx_1 ** 2 * dvdx_0 * dvdx_1 ** 2 * dvdy_0 + 45 * d * dudx_0 ** 2 * dudx_1 ** 2 * dvdx_0 ** 2 * dvdx_1 * dvdy_0 + 18 * d * dudx_0 ** 2 * dudy_0 ** 2 * dvdx_0 * dvdx_1 * dvdy_0 ** 2 + 30 * d * dudx_0 ** 2 * dudy_0 ** 2 * dvdx_0 * dvdx_1 ** 2 * dvdy_0 + 45 * d * dudx_0 ** 2 * dudy_0 ** 2 * dvdx_0 ** 2 * dvdx_1 * dvdy_0 + 18 * d * dudx_1 ** 2 * dudy_0 ** 2 * dvdx_0 * dvdx_1 * dvdy_0 ** 2 - 45 * d * dudx_1 ** 2 * dudy_0 ** 2 * dvdx_0 * dvdx_1 ** 2 * dvdy_0 - 30 * d * dudx_1 ** 2 * dudy_0 ** 2 * dvdx_0 ** 2 * dvdx_1 * dvdy_0 + 18 * d * dudx_0 * dudx_1 * dudy_0 ** 2 * dvdx_0 ** 2 * dy ** 2 + 144 * d * dudx_0 * dudx_1 * dvdx_0 ** 2 * dvdx_1 ** 2 * dx ** 2 + 140 * d * dudx_0 * dudx_1 * dvdx_0 ** 2 * dvdy_0 ** 2 * dx ** 2 + 45 * d * dudx_0 * dudy_0 * dvdx_0 ** 2 * dvdx_1 ** 2 * dx ** 2 + 120 * d * dudx_0 * dudy_0 * dvdx_0 ** 2 * dvdy_0 ** 2 * dx ** 2 + 30 * d * dudx_0 * dudx_1 ** 2 * dudy_0 * dvdx_0 ** 2 * dy ** 2 + 72 * d * dudx_0 * dudx_1 ** 2 * dvdx_0 ** 2 * dvdx_1 ** 2 * dx + 100 * d * dudx_0 * dudx_1 ** 2 * dvdx_0 ** 2 * dvdy_0 ** 2 * dx + 90 * d * dudx_0 * dudy_0 ** 2 * dvdx_0 ** 2 * dvdy_0 ** 2 * dx + 45 * d * dudx_0 ** 2 * dudx_1 * dudy_0 * dvdx_0 ** 2 * dy ** 2 + 72 * d * dudx_0 ** 2 * dudx_1 * dvdx_0 ** 2 * dvdx_1 ** 2 * dx + 40 * d * dudx_0 ** 2 * dudx_1 * dvdx_0 ** 2 * dvdy_0 ** 2 * dx + 45 * d * dudx_0 ** 2 * dudy_0 * dvdx_0 ** 2 * dvdx_1 ** 2 * dx + 45 * d * dudx_0 ** 2 * dudy_0 * dvdx_0 ** 2 * dvdy_0 ** 2 * dx + 18 * d * dudx_0 * dudx_1 * dudy_0 ** 2 * dvdx_1 ** 2 * dy ** 2 + 140 * d * dudx_0 * dudx_1 * dvdx_1 ** 2 * dvdy_0 ** 2 * dx ** 2 - 150 * d * dudx_0 * dudy_0 * dvdx_1 ** 2 * dvdy_0 ** 2 * dx ** 2 - 45 * d * dudx_0 * dudx_1 ** 2 * dudy_0 * dvdx_1 ** 2 * dy ** 2 + 40 * d * dudx_0 * dudx_1 ** 2 * dvdx_1 ** 2 * dvdy_0 ** 2 * dx + 30 * d * dudx_0 * dudy_0 ** 2 * dvdx_1 ** 2 * dvdy_0 ** 2 * dx - 45 * d * dudx_1 * dudy_0 * dvdx_0 ** 2 * dvdx_1 ** 2 * dx ** 2 + 150 * d * dudx_1 * dudy_0 * dvdx_0 ** 2 * dvdy_0 ** 2 * dx ** 2 + 30 * d * dudx_1 * dudy_0 ** 2 * dvdx_0 ** 2 * dvdy_0 ** 2 * dx - 30 * d * dudx_0 ** 2 * dudx_1 * dudy_0 * dvdx_1 ** 2 * dy ** 2 + 100 * d * dudx_0 ** 2 * dudx_1 * dvdx_1 ** 2 * dvdy_0 ** 2 * dx - 75 * d * dudx_0 ** 2 * dudy_0 * dvdx_1 ** 2 * dvdy_0 ** 2 * dx - 45 * d * dudx_1 ** 2 * dudy_0 * dvdx_0 ** 2 * dvdx_1 ** 2 * dx + 75 * d * dudx_1 ** 2 * dudy_0 * dvdx_0 ** 2 * dvdy_0 ** 2 * dx - 120 * d * dudx_1 * dudy_0 * dvdx_1 ** 2 * dvdy_0 ** 2 * dx ** 2 + 90 * d * dudx_1 * dudy_0 ** 2 * dvdx_1 ** 2 * dvdy_0 ** 2 * dx - 45 * d * dudx_1 ** 2 * dudy_0 * dvdx_1 ** 2 * dvdy_0 ** 2 * dx - 160 * d * dudx_0 ** 2 * dvdx_0 * dvdx_1 * dvdy_0 ** 2 * dx ** 2 - 180 * d * dudx_0 ** 2 * dvdx_0 * dvdx_1 ** 2 * dvdy_0 * dx ** 2 - 40 * d * dudx_0 ** 2 * dudx_1 ** 2 * dvdx_0 * dvdx_1 * dy ** 2 - 45 * d * dudx_0 ** 2 * dudx_1 ** 2 * dvdx_0 * dvdy_0 * dy ** 2 - 45 * d * dudx_0 ** 2 * dudx_1 ** 2 * dvdx_0 * dvdx_1 ** 2 * dy - 45 * d * dudx_0 ** 2 * dudx_1 ** 2 * dvdx_0 * dvdy_0 ** 2 * dy + 45 * d * dudx_0 ** 2 * dudx_1 ** 2 * dvdx_0 ** 2 * dvdx_1 * dy + 40 * d * dudx_0 ** 2 * dudx_1 ** 2 * dvdx_0 ** 2 * dvdy_0 * dy + 18 * d * dudx_0 ** 2 * dudy_0 ** 2 * dvdx_0 * dvdx_1 * dy ** 2 + 30 * d * dudx_0 ** 2 * dudy_0 ** 2 * dvdx_0 * dvdx_1 ** 2 * dy + 45 * d * dudx_0 ** 2 * dudy_0 ** 2 * dvdx_0 ** 2 * dvdx_1 * dy + 72 * d * dudx_0 ** 2 * dudy_0 ** 2 * dvdx_0 ** 2 * dvdy_0 * dy + 45 * d * dudx_0 ** 2 * dvdx_0 ** 2 * dvdx_1 * dvdy_0 * dx ** 2 + 45 * d * dudx_0 ** 2 * dudx_1 ** 2 * dvdx_1 * dvdy_0 * dy ** 2 + 45 * d * dudx_0 ** 2 * dudx_1 ** 2 * dvdx_1 * dvdy_0 ** 2 * dy + 40 * d * dudx_0 ** 2 * dudx_1 ** 2 * dvdx_1 ** 2 * dvdy_0 * dy + 12 * d * dudx_0 ** 2 * dudy_0 ** 2 * dvdx_1 ** 2 * dvdy_0 * dy - 160 * d * dudx_1 ** 2 * dvdx_0 * dvdx_1 * dvdy_0 ** 2 * dx ** 2 - 45 * d * dudx_1 ** 2 * dvdx_0 * dvdx_1 ** 2 * dvdy_0 * dx ** 2 + 18 * d * dudx_1 ** 2 * dudy_0 ** 2 * dvdx_0 * dvdx_1 * dy ** 2 - 45 * d * dudx_1 ** 2 * dudy_0 ** 2 * dvdx_0 * dvdx_1 ** 2 * dy - 30 * d * dudx_1 ** 2 * dudy_0 ** 2 * dvdx_0 ** 2 * dvdx_1 * dy + 12 * d * dudx_1 ** 2 * dudy_0 ** 2 * dvdx_0 ** 2 * dvdy_0 * dy + 180 * d * dudx_1 ** 2 * dvdx_0 ** 2 * dvdx_1 * dvdy_0 * dx ** 2 + 60 * d * dudy_0 ** 2 * dvdx_0 * dvdx_1 * dvdy_0 ** 2 * dx ** 2 + 72 * d * dudx_1 ** 2 * dudy_0 ** 2 * dvdx_1 ** 2 * dvdy_0 * dy - 20 * d * dudx_0 * dudx_1 * dvdx_0 ** 2 * dx ** 2 * dy ** 2 - 40 * d * dudx_0 * dudy_0 * dvdx_0 ** 2 * dx ** 2 * dy ** 2 + 20 * d * dudx_0 * dudx_1 ** 2 * dvdx_0 ** 2 * dx * dy ** 2 - 30 * d * dudx_0 * dudy_0 ** 2 * dvdx_0 ** 2 * dx * dy ** 2 + 40 * d * dudx_0 ** 2 * dudx_1 * dvdx_0 ** 2 * dx * dy ** 2 + 45 * d * dudx_0 ** 2 * dudy_0 * dvdx_0 ** 2 * dx * dy ** 2 - 20 * d * dudx_0 * dudx_1 * dvdx_1 ** 2 * dx ** 2 * dy ** 2 - 10 * d * dudx_0 * dudy_0 * dvdx_1 ** 2 * dx ** 2 * dy ** 2 + 40 * d * dudx_0 * dudx_1 ** 2 * dvdx_1 ** 2 * dx * dy ** 2 + 60 * d * dudx_0 * dudx_1 ** 2 * dvdy_0 ** 2 * dx * dy ** 2 - 10 * d * dudx_0 * dudy_0 ** 2 * dvdx_1 ** 2 * dx * dy ** 2 + 10 * d * dudx_1 * dudy_0 * dvdx_0 ** 2 * dx ** 2 * dy ** 2 - 10 * d * dudx_1 * dudy_0 ** 2 * dvdx_0 ** 2 * dx * dy ** 2)
        loss_int += jnp.sum(20 * d * dudx_0 ** 2 * dudx_1 * dvdx_1 ** 2 * dx * dy ** 2 + 60 * d * dudx_0 ** 2 * dudx_1 * dvdy_0 ** 2 * dx * dy ** 2 + 5 * d * dudx_0 ** 2 * dudy_0 * dvdx_1 ** 2 * dx * dy ** 2 - 5 * d * dudx_1 ** 2 * dudy_0 * dvdx_0 ** 2 * dx * dy ** 2 + 40 * d * dudx_1 * dudy_0 * dvdx_1 ** 2 * dx ** 2 * dy ** 2 - 30 * d * dudx_1 * dudy_0 ** 2 * dvdx_1 ** 2 * dx * dy ** 2 - 45 * d * dudx_1 ** 2 * dudy_0 * dvdx_1 ** 2 * dx * dy ** 2 - 20 * d * dudx_0 ** 2 * dvdx_0 * dvdy_0 * dx ** 2 * dy ** 2 - 60 * d * dudx_0 ** 2 * dvdx_0 * dvdx_1 ** 2 * dx ** 2 * dy - 100 * d * dudx_0 ** 2 * dvdx_0 * dvdy_0 ** 2 * dx ** 2 * dy + 45 * d * dudx_0 ** 2 * dvdx_0 ** 2 * dvdx_1 * dx ** 2 * dy + 40 * d * dudx_0 ** 2 * dvdx_0 ** 2 * dvdy_0 * dx ** 2 * dy - 10 * d * dudx_0 ** 2 * dvdx_1 * dvdy_0 * dx ** 2 * dy ** 2 + 130 * d * dudx_0 ** 2 * dvdx_1 * dvdy_0 ** 2 * dx ** 2 * dy + 80 * d * dudx_0 ** 2 * dvdx_1 ** 2 * dvdy_0 * dx ** 2 * dy + 10 * d * dudx_1 ** 2 * dvdx_0 * dvdy_0 * dx ** 2 * dy ** 2 - 45 * d * dudx_1 ** 2 * dvdx_0 * dvdx_1 ** 2 * dx ** 2 * dy - 130 * d * dudx_1 ** 2 * dvdx_0 * dvdy_0 ** 2 * dx ** 2 * dy + 60 * d * dudx_1 ** 2 * dvdx_0 ** 2 * dvdx_1 * dx ** 2 * dy + 80 * d * dudx_1 ** 2 * dvdx_0 ** 2 * dvdy_0 * dx ** 2 * dy + 20 * d * dudy_0 ** 2 * dvdx_0 * dvdx_1 * dx ** 2 * dy ** 2 - 40 * d * dudy_0 ** 2 * dvdx_0 ** 2 * dvdy_0 * dx ** 2 * dy + 20 * d * dudx_1 ** 2 * dvdx_1 * dvdy_0 * dx ** 2 * dy ** 2 + 100 * d * dudx_1 ** 2 * dvdx_1 * dvdy_0 ** 2 * dx ** 2 * dy + 40 * d * dudx_1 ** 2 * dvdx_1 ** 2 * dvdy_0 * dx ** 2 * dy - 40 * d * dudy_0 ** 2 * dvdx_1 ** 2 * dvdy_0 * dx ** 2 * dy + 90 * d * dudx_0 * dudx_1 * dudy_0 * dvdx_0 * dvdx_1 ** 3 * dx - 60 * d * dudx_0 * dudx_1 * dudy_0 * dvdx_0 * dvdy_0 ** 3 * dx - 90 * d * dudx_0 * dudx_1 * dudy_0 * dvdx_0 ** 3 * dvdx_1 * dx - 80 * d * dudx_0 * dudx_1 * dudy_0 * dvdx_0 ** 3 * dvdy_0 * dx - 60 * d * dudx_0 * dudx_1 * dudy_0 * dvdx_1 * dvdy_0 ** 3 * dx - 80 * d * dudx_0 * dudx_1 * dudy_0 * dvdx_1 ** 3 * dvdy_0 * dx - 80 * d * dudx_0 * dudx_1 ** 3 * dvdx_0 * dvdx_1 * dvdy_0 * dy - 80 * d * dudx_0 ** 3 * dudx_1 * dvdx_0 * dvdx_1 * dvdy_0 * dy - 90 * d * dudx_0 ** 3 * dudy_0 * dvdx_0 * dvdx_1 * dvdy_0 * dy + 90 * d * dudx_1 ** 3 * dudy_0 * dvdx_0 * dvdx_1 * dvdy_0 * dy + 20 * d * dudx_0 * dudx_1 * dudy_0 * dvdx_0 * dx * dy ** 3 - 80 * d * dudx_0 * dudx_1 * dudy_0 * dvdx_0 ** 3 * dx * dy + 20 * d * dudx_0 * dudx_1 * dudy_0 * dvdx_1 * dx * dy ** 3 - 80 * d * dudx_0 * dudx_1 * dudy_0 * dvdx_1 ** 3 * dx * dy - 80 * d * dudx_0 ** 3 * dvdx_0 * dvdx_1 * dvdy_0 * dx * dy - 80 * d * dudx_1 ** 3 * dvdx_0 * dvdx_1 * dvdy_0 * dx * dy - 80 * c * dudx_0 * dudx_1 * dvdx_0 * dvdx_1 * dx ** 2 * dy ** 2 - 60 * c * dudx_0 * dudx_1 * dvdx_0 * dvdy_0 * dx ** 2 * dy ** 2 - 60 * c * dudx_0 * dudy_0 * dvdx_0 * dvdx_1 * dx ** 2 * dy ** 2 - 80 * c * dudx_0 * dudy_0 * dvdx_0 * dvdy_0 * dx ** 2 * dy ** 2 + 60 * c * dudx_0 * dudx_1 * dvdx_1 * dvdy_0 * dx ** 2 * dy ** 2 - 40 * c * dudx_0 * dudy_0 * dvdx_1 * dvdy_0 * dx ** 2 * dy ** 2 + 60 * c * dudx_1 * dudy_0 * dvdx_0 * dvdx_1 * dx ** 2 * dy ** 2 - 40 * c * dudx_1 * dudy_0 * dvdx_0 * dvdy_0 * dx ** 2 * dy ** 2 - 80 * c * dudx_1 * dudy_0 * dvdx_1 * dvdy_0 * dx ** 2 * dy ** 2 - 80 * c2 * dudx_0 * dudx_1 * dvdx_0 * dvdx_1 * dx ** 2 * dy ** 2 - 60 * c2 * dudx_0 * dudx_1 * dvdx_0 * dvdy_0 * dx ** 2 * dy ** 2 - 60 * c2 * dudx_0 * dudy_0 * dvdx_0 * dvdx_1 * dx ** 2 * dy ** 2 - 80 * c2 * dudx_0 * dudy_0 * dvdx_0 * dvdy_0 * dx ** 2 * dy ** 2 + 60 * c2 * dudx_0 * dudx_1 * dvdx_1 * dvdy_0 * dx ** 2 * dy ** 2 - 40 * c2 * dudx_0 * dudy_0 * dvdx_1 * dvdy_0 * dx ** 2 * dy ** 2 + 60 * c2 * dudx_1 * dudy_0 * dvdx_0 * dvdx_1 * dx ** 2 * dy ** 2 - 40 * c2 * dudx_1 * dudy_0 * dvdx_0 * dvdy_0 * dx ** 2 * dy ** 2 - 80 * c2 * dudx_1 * dudy_0 * dvdx_1 * dvdy_0 * dx ** 2 * dy ** 2 + 24 * d * dudx_0 * dudx_1 * dudy_0 ** 2 * dvdx_0 * dvdx_1 * dvdy_0 ** 2 + 15 * d * dudx_0 * dudx_1 * dudy_0 ** 2 * dvdx_0 * dvdx_1 ** 2 * dvdy_0 - 15 * d * dudx_0 * dudx_1 * dudy_0 ** 2 * dvdx_0 ** 2 * dvdx_1 * dvdy_0 + 15 * d * dudx_0 * dudx_1 ** 2 * dudy_0 * dvdx_0 * dvdx_1 * dvdy_0 ** 2 + 80 * d * dudx_0 * dudx_1 ** 2 * dudy_0 * dvdx_0 * dvdx_1 ** 2 * dvdy_0 + 20 * d * dudx_0 * dudx_1 ** 2 * dudy_0 * dvdx_0 ** 2 * dvdx_1 * dvdy_0 - 15 * d * dudx_0 ** 2 * dudx_1 * dudy_0 * dvdx_0 * dvdx_1 * dvdy_0 ** 2 + 20 * d * dudx_0 ** 2 * dudx_1 * dudy_0 * dvdx_0 * dvdx_1 ** 2 * dvdy_0 + 80 * d * dudx_0 ** 2 * dudx_1 * dudy_0 * dvdx_0 ** 2 * dvdx_1 * dvdy_0 + 150 * d * dudx_0 * dudx_1 * dudy_0 * dvdx_0 ** 2 * dvdy_0 ** 2 * dx - 150 * d * dudx_0 * dudx_1 * dudy_0 * dvdx_1 ** 2 * dvdy_0 ** 2 * dx - 400 * d * dudx_0 * dudx_1 * dvdx_0 * dvdx_1 * dvdy_0 ** 2 * dx ** 2 - 315 * d * dudx_0 * dudx_1 * dvdx_0 * dvdx_1 ** 2 * dvdy_0 * dx ** 2 + 24 * d * dudx_0 * dudx_1 * dudy_0 ** 2 * dvdx_0 * dvdx_1 * dy ** 2 + 15 * d * dudx_0 * dudx_1 * dudy_0 ** 2 * dvdx_0 * dvdx_1 ** 2 * dy - 15 * d * dudx_0 * dudx_1 * dudy_0 ** 2 * dvdx_0 ** 2 * dvdx_1 * dy + 36 * d * dudx_0 * dudx_1 * dudy_0 ** 2 * dvdx_0 ** 2 * dvdy_0 * dy + 315 * d * dudx_0 * dudx_1 * dvdx_0 ** 2 * dvdx_1 * dvdy_0 * dx ** 2 - 150 * d * dudx_0 * dudy_0 * dvdx_0 * dvdx_1 * dvdy_0 ** 2 * dx ** 2 + 180 * d * dudx_0 * dudy_0 * dvdx_0 ** 2 * dvdx_1 * dvdy_0 * dx ** 2 + 15 * d * dudx_0 * dudx_1 ** 2 * dudy_0 * dvdx_0 * dvdx_1 * dy ** 2 - 36 * d * dudx_0 * dudx_1 ** 2 * dudy_0 * dvdx_0 * dvdy_0 * dy ** 2 + 80 * d * dudx_0 * dudx_1 ** 2 * dudy_0 * dvdx_0 * dvdx_1 ** 2 * dy - 36 * d * dudx_0 * dudx_1 ** 2 * dudy_0 * dvdx_0 * dvdy_0 ** 2 * dy + 20 * d * dudx_0 * dudx_1 ** 2 * dudy_0 * dvdx_0 ** 2 * dvdx_1 * dy)
        loss_int += jnp.sum(60 * d * dudx_0 * dudx_1 ** 2 * dudy_0 * dvdx_0 ** 2 * dvdy_0 * dy - 200 * d * dudx_0 * dudx_1 ** 2 * dvdx_0 * dvdx_1 * dvdy_0 ** 2 * dx - 90 * d * dudx_0 * dudx_1 ** 2 * dvdx_0 * dvdx_1 ** 2 * dvdy_0 * dx + 225 * d * dudx_0 * dudx_1 ** 2 * dvdx_0 ** 2 * dvdx_1 * dvdy_0 * dx + 60 * d * dudx_0 * dudy_0 ** 2 * dvdx_0 * dvdx_1 * dvdy_0 ** 2 * dx + 75 * d * dudx_0 * dudy_0 ** 2 * dvdx_0 * dvdx_1 ** 2 * dvdy_0 * dx + 75 * d * dudx_0 * dudy_0 ** 2 * dvdx_0 ** 2 * dvdx_1 * dvdy_0 * dx - 15 * d * dudx_0 ** 2 * dudx_1 * dudy_0 * dvdx_0 * dvdx_1 * dy ** 2 - 54 * d * dudx_0 ** 2 * dudx_1 * dudy_0 * dvdx_0 * dvdy_0 * dy ** 2 + 20 * d * dudx_0 ** 2 * dudx_1 * dudy_0 * dvdx_0 * dvdx_1 ** 2 * dy - 54 * d * dudx_0 ** 2 * dudx_1 * dudy_0 * dvdx_0 * dvdy_0 ** 2 * dy + 80 * d * dudx_0 ** 2 * dudx_1 * dudy_0 * dvdx_0 ** 2 * dvdx_1 * dy + 90 * d * dudx_0 ** 2 * dudx_1 * dudy_0 * dvdx_0 ** 2 * dvdy_0 * dy - 200 * d * dudx_0 ** 2 * dudx_1 * dvdx_0 * dvdx_1 * dvdy_0 ** 2 * dx - 225 * d * dudx_0 ** 2 * dudx_1 * dvdx_0 * dvdx_1 ** 2 * dvdy_0 * dx + 90 * d * dudx_0 ** 2 * dudx_1 * dvdx_0 ** 2 * dvdx_1 * dvdy_0 * dx - 150 * d * dudx_0 ** 2 * dudy_0 * dvdx_0 * dvdx_1 * dvdy_0 ** 2 * dx - 100 * d * dudx_0 ** 2 * dudy_0 * dvdx_0 * dvdx_1 ** 2 * dvdy_0 * dx + 80 * d * dudx_0 ** 2 * dudy_0 * dvdx_0 ** 2 * dvdx_1 * dvdy_0 * dx + 36 * d * dudx_0 * dudx_1 * dudy_0 ** 2 * dvdx_1 ** 2 * dvdy_0 * dy - 54 * d * dudx_0 * dudx_1 ** 2 * dudy_0 * dvdx_1 * dvdy_0 * dy ** 2 - 54 * d * dudx_0 * dudx_1 ** 2 * dudy_0 * dvdx_1 * dvdy_0 ** 2 * dy - 90 * d * dudx_0 * dudx_1 ** 2 * dudy_0 * dvdx_1 ** 2 * dvdy_0 * dy + 150 * d * dudx_1 * dudy_0 * dvdx_0 * dvdx_1 * dvdy_0 ** 2 * dx ** 2 + 180 * d * dudx_1 * dudy_0 * dvdx_0 * dvdx_1 ** 2 * dvdy_0 * dx ** 2 + 60 * d * dudx_1 * dudy_0 ** 2 * dvdx_0 * dvdx_1 * dvdy_0 ** 2 * dx - 75 * d * dudx_1 * dudy_0 ** 2 * dvdx_0 * dvdx_1 ** 2 * dvdy_0 * dx - 75 * d * dudx_1 * dudy_0 ** 2 * dvdx_0 ** 2 * dvdx_1 * dvdy_0 * dx - 36 * d * dudx_0 ** 2 * dudx_1 * dudy_0 * dvdx_1 * dvdy_0 * dy ** 2 - 36 * d * dudx_0 ** 2 * dudx_1 * dudy_0 * dvdx_1 * dvdy_0 ** 2 * dy - 60 * d * dudx_0 ** 2 * dudx_1 * dudy_0 * dvdx_1 ** 2 * dvdy_0 * dy + 150 * d * dudx_1 ** 2 * dudy_0 * dvdx_0 * dvdx_1 * dvdy_0 ** 2 * dx + 80 * d * dudx_1 ** 2 * dudy_0 * dvdx_0 * dvdx_1 ** 2 * dvdy_0 * dx - 100 * d * dudx_1 ** 2 * dudy_0 * dvdx_0 ** 2 * dvdx_1 * dvdy_0 * dx - 80 * d * dudx_0 ** 2 * dudx_1 ** 2 * dvdx_0 * dvdx_1 * dvdy_0 * dy + 36 * d * dudx_0 ** 2 * dudy_0 ** 2 * dvdx_0 * dvdx_1 * dvdy_0 * dy + 36 * d * dudx_1 ** 2 * dudy_0 ** 2 * dvdx_0 * dvdx_1 * dvdy_0 * dy - 10 * d * dudx_0 * dudx_1 * dudy_0 * dvdx_0 ** 2 * dx * dy ** 2 + 10 * d * dudx_0 * dudx_1 * dudy_0 * dvdx_1 ** 2 * dx * dy ** 2 + 10 * d * dudx_0 * dudx_1 * dvdx_0 * dvdy_0 * dx ** 2 * dy ** 2 - 75 * d * dudx_0 * dudx_1 * dvdx_0 * dvdx_1 ** 2 * dx ** 2 * dy - 130 * d * dudx_0 * dudx_1 * dvdx_0 * dvdy_0 ** 2 * dx ** 2 * dy + 75 * d * dudx_0 * dudx_1 * dvdx_0 ** 2 * dvdx_1 * dx ** 2 * dy + 120 * d * dudx_0 * dudx_1 * dvdx_0 ** 2 * dvdy_0 * dx ** 2 * dy - 10 * d * dudx_0 * dudy_0 * dvdx_0 * dvdx_1 * dx ** 2 * dy ** 2 + 40 * d * dudx_0 * dudy_0 * dvdx_0 * dvdy_0 * dx ** 2 * dy ** 2 - 40 * d * dudx_0 * dudy_0 * dvdx_0 * dvdy_0 ** 2 * dx ** 2 * dy + 20 * d * dudx_0 * dudy_0 * dvdx_0 ** 2 * dvdx_1 * dx ** 2 * dy + 80 * d * dudx_0 * dudy_0 * dvdx_0 ** 2 * dvdy_0 * dx ** 2 * dy - 40 * d * dudx_0 * dudx_1 ** 2 * dvdx_0 * dvdx_1 * dx * dy ** 2 - 65 * d * dudx_0 * dudx_1 ** 2 * dvdx_0 * dvdy_0 * dx * dy ** 2 - 90 * d * dudx_0 * dudx_1 ** 2 * dvdx_0 * dvdx_1 ** 2 * dx * dy - 145 * d * dudx_0 * dudx_1 ** 2 * dvdx_0 * dvdy_0 ** 2 * dx * dy + 105 * d * dudx_0 * dudx_1 ** 2 * dvdx_0 ** 2 * dvdx_1 * dx * dy + 120 * d * dudx_0 * dudx_1 ** 2 * dvdx_0 ** 2 * dvdy_0 * dx * dy - 20 * d * dudx_0 * dudy_0 ** 2 * dvdx_0 * dvdx_1 * dx * dy ** 2 - 5 * d * dudx_0 * dudy_0 ** 2 * dvdx_0 * dvdx_1 ** 2 * dx * dy - 5 * d * dudx_0 * dudy_0 ** 2 * dvdx_0 ** 2 * dvdx_1 * dx * dy + 60 * d * dudx_0 * dudy_0 ** 2 * dvdx_0 ** 2 * dvdy_0 * dx * dy - 40 * d * dudx_0 ** 2 * dudx_1 * dvdx_0 * dvdx_1 * dx * dy ** 2 - 65 * d * dudx_0 ** 2 * dudx_1 * dvdx_0 * dvdy_0 * dx * dy ** 2 - 105 * d * dudx_0 ** 2 * dudx_1 * dvdx_0 * dvdx_1 ** 2 * dx * dy - 145 * d * dudx_0 ** 2 * dudx_1 * dvdx_0 * dvdy_0 ** 2 * dx * dy + 90 * d * dudx_0 ** 2 * dudx_1 * dvdx_0 ** 2 * dvdx_1 * dx * dy + 80 * d * dudx_0 ** 2 * dudx_1 * dvdx_0 ** 2 * dvdy_0 * dx * dy + 10 * d * dudx_0 ** 2 * dudy_0 * dvdx_0 * dvdx_1 * dx * dy ** 2 - 30 * d * dudx_0 ** 2 * dudy_0 * dvdx_0 * dvdy_0 * dx * dy ** 2 - 20 * d * dudx_0 ** 2 * dudy_0 * dvdx_0 * dvdx_1 ** 2 * dx * dy - 150 * d * dudx_0 ** 2 * dudy_0 * dvdx_0 * dvdy_0 ** 2 * dx * dy + 80 * d * dudx_0 ** 2 * dudy_0 * dvdx_0 ** 2 * dvdx_1 * dx * dy + 90 * d * dudx_0 ** 2 * dudy_0 * dvdx_0 ** 2 * dvdy_0 * dx * dy - 10 * d * dudx_0 * dudx_1 * dvdx_1 * dvdy_0 * dx ** 2 * dy ** 2 + 130 * d * dudx_0 * dudx_1 * dvdx_1 * dvdy_0 ** 2 * dx ** 2 * dy + 120 * d * dudx_0 * dudx_1 * dvdx_1 ** 2 * dvdy_0 * dx ** 2 * dy + 20 * d * dudx_0 * dudy_0 * dvdx_1 * dvdy_0 * dx ** 2 * dy ** 2 - 20 * d * dudx_0 * dudy_0 * dvdx_1 * dvdy_0 ** 2 * dx ** 2 * dy + 20 * d * dudx_0 * dudy_0 * dvdx_1 ** 2 * dvdy_0 * dx ** 2 * dy + 65 * d * dudx_0 * dudx_1 ** 2 * dvdx_1 * dvdy_0 * dx * dy ** 2 + 145 * d * dudx_0 * dudx_1 ** 2 * dvdx_1 * dvdy_0 ** 2 * dx * dy + 80 * d * dudx_0 * dudx_1 ** 2 * dvdx_1 ** 2 * dvdy_0 * dx * dy + 20 * d * dudx_0 * dudy_0 ** 2 * dvdx_1 ** 2 * dvdy_0 * dx * dy + 10 * d * dudx_1 * dudy_0 * dvdx_0 * dvdx_1 * dx ** 2 * dy ** 2 + 20 * d * dudx_1 * dudy_0 * dvdx_0 * dvdy_0 * dx ** 2 * dy ** 2 + 20 * d * dudx_1 * dudy_0 * dvdx_0 * dvdx_1 ** 2 * dx ** 2 * dy - 20 * d * dudx_1 * dudy_0 * dvdx_0 * dvdy_0 ** 2 * dx ** 2 * dy - 20 * d * dudx_1 * dudy_0 * dvdx_0 ** 2 * dvdy_0 * dx ** 2 * dy - 20 * d * dudx_1 * dudy_0 ** 2 * dvdx_0 * dvdx_1 * dx * dy ** 2 + 5 * d * dudx_1 * dudy_0 ** 2 * dvdx_0 * dvdx_1 ** 2 * dx * dy + 5 * d * dudx_1 * dudy_0 ** 2 * dvdx_0 ** 2 * dvdx_1 * dx * dy + 20 * d * dudx_1 * dudy_0 ** 2 * dvdx_0 ** 2 * dvdy_0 * dx * dy + 65 * d * dudx_0 ** 2 * dudx_1 * dvdx_1 * dvdy_0 * dx * dy ** 2 + 145 * d * dudx_0 ** 2 * dudx_1 * dvdx_1 * dvdy_0 ** 2 * dx * dy + 120 * d * dudx_0 ** 2 * dudx_1 * dvdx_1 ** 2 * dvdy_0 * dx * dy - 10 * d * dudx_0 ** 2 * dudy_0 * dvdx_1 * dvdy_0 * dx * dy ** 2 - 50 * d * dudx_0 ** 2 * dudy_0 * dvdx_1 * dvdy_0 ** 2 * dx * dy - 70 * d * dudx_0 ** 2 * dudy_0 * dvdx_1 ** 2 * dvdy_0 * dx * dy - 10 * d * dudx_1 ** 2 * dudy_0 * dvdx_0 * dvdx_1 * dx * dy ** 2 - 10 * d * dudx_1 ** 2 * dudy_0 * dvdx_0 * dvdy_0 * dx * dy ** 2 + 80 * d * dudx_1 ** 2 * dudy_0 * dvdx_0 * dvdx_1 ** 2 * dx * dy - 50 * d * dudx_1 ** 2 * dudy_0 * dvdx_0 * dvdy_0 ** 2 * dx * dy - 20 * d * dudx_1 ** 2 * dudy_0 * dvdx_0 ** 2 * dvdx_1 * dx * dy + 70 * d * dudx_1 ** 2 * dudy_0 * dvdx_0 ** 2 * dvdy_0 * dx * dy + 40 * d * dudx_1 * dudy_0 * dvdx_1 * dvdy_0 * dx ** 2 * dy ** 2 - 40 * d * dudx_1 * dudy_0 * dvdx_1 * dvdy_0 ** 2 * dx ** 2 * dy - 80 * d * dudx_1 * dudy_0 * dvdx_1 ** 2 * dvdy_0 * dx ** 2 * dy + 60 * d * dudx_1 * dudy_0 ** 2 * dvdx_1 ** 2 * dvdy_0 * dx * dy - 30 * d * dudx_1 ** 2 * dudy_0 * dvdx_1 * dvdy_0 * dx * dy ** 2 - 150 * d * dudx_1 ** 2 * dudy_0 * dvdx_1 * dvdy_0 ** 2 * dx * dy - 90 * d * dudx_1 ** 2 * dudy_0 * dvdx_1 ** 2 * dvdy_0 * dx * dy - 160 * d * dudx_0 ** 2 * dvdx_0 * dvdx_1 * dvdy_0 * dx ** 2 * dy - 160 * d * dudx_1 ** 2 * dvdx_0 * dvdx_1 * dvdy_0 * dx ** 2 * dy - 40 * d * dudy_0 ** 2 * dvdx_0 * dvdx_1 * dvdy_0 * dx ** 2 * dy + 200 * d * dudx_0 * dudx_1 * dudy_0 * dvdx_0 * dvdx_1 ** 2 * dvdy_0 * dx + 200 * d * dudx_0 * dudx_1 * dudy_0 * dvdx_0 ** 2 * dvdx_1 * dvdy_0 * dx + 48 * d * dudx_0 * dudx_1 * dudy_0 ** 2 * dvdx_0 * dvdx_1 * dvdy_0 * dy + 30 * d * dudx_0 * dudx_1 ** 2 * dudy_0 * dvdx_0 * dvdx_1 * dvdy_0 * dy - 30 * d * dudx_0 ** 2 * dudx_1 * dudy_0 * dvdx_0 * dvdx_1 * dvdy_0 * dy - 20 * d * dudx_0 * dudx_1 * dudy_0 * dvdx_0 * dvdy_0 * dx * dy ** 2 + 40 * d * dudx_0 * dudx_1 * dudy_0 * dvdx_0 * dvdx_1 ** 2 * dx * dy - 100 * d * dudx_0 * dudx_1 * dudy_0 * dvdx_0 * dvdy_0 ** 2 * dx * dy + 40 * d * dudx_0 * dudx_1 * dudy_0 * dvdx_0 ** 2 * dvdx_1 * dx * dy + 140 * d * dudx_0 * dudx_1 * dudy_0 * dvdx_0 ** 2 * dvdy_0 * dx * dy - 20 * d * dudx_0 * dudx_1 * dudy_0 * dvdx_1 * dvdy_0 * dx * dy ** 2 - 100 * d * dudx_0 * dudx_1 * dudy_0 * dvdx_1 * dvdy_0 ** 2 * dx * dy - 140 * d * dudx_0 * dudx_1 * dudy_0 * dvdx_1 ** 2 * dvdy_0 * dx * dy - 160 * d * dudx_0 * dudx_1 * dvdx_0 * dvdx_1 * dvdy_0 * dx ** 2 * dy + 20 * d * dudx_0 * dudy_0 * dvdx_0 * dvdx_1 * dvdy_0 * dx ** 2 * dy - 240 * d * dudx_0 * dudx_1 ** 2 * dvdx_0 * dvdx_1 * dvdy_0 * dx * dy + 40 * d * dudx_0 * dudy_0 ** 2 * dvdx_0 * dvdx_1 * dvdy_0 * dx * dy - 240 * d * dudx_0 ** 2 * dudx_1 * dvdx_0 * dvdx_1 * dvdy_0 * dx * dy - 140 * d * dudx_0 ** 2 * dudy_0 * dvdx_0 * dvdx_1 * dvdy_0 * dx * dy - 20 * d * dudx_1 * dudy_0 * dvdx_0 * dvdx_1 * dvdy_0 * dx ** 2 * dy + 40 * d * dudx_1 * dudy_0 ** 2 * dvdx_0 * dvdx_1 * dvdy_0 * dx * dy + 140 * d * dudx_1 ** 2 * dudy_0 * dvdx_0 * dvdx_1 * dvdy_0 * dx * dy)
        loss_int = loss_int / (120 * dx ** 3 * dy ** 3)

        # Third-order Taylor polynomial
        # loss_int = jnp.sum((d * dUdx_1 ** 3 * dVdx_0 ** 3 - d * dUdx_0 ** 3 * dVdy_0 ** 3 - d * dUdx_0 ** 3 * dVdx_1 ** 3 + d * dUdy_0 ** 3 * dVdx_0 ** 3 - d * dUdx_1 ** 3 * dVdy_0 ** 3 + d * dUdy_0 ** 3 * dVdx_1 ** 3 + d * dVdx_0 ** 3 * dx ** 3 - d * dVdx_1 ** 3 * dx ** 3 - 4 * d * dVdy_0 ** 3 * dx ** 3 + 4 * c * dUdx_0 ** 2 * dx ** 4 * n ** 3 + 4 * c * dUdx_1 ** 2 * dx ** 4 * n ** 3 + 4 * c1 * dUdx_0 ** 2 * dx ** 4 * n ** 3 + 4 * c1 * dUdx_1 ** 2 * dx ** 4 * n ** 3 + 4 * c2 * dUdx_0 ** 2 * dx ** 4 * n ** 3 + 4 * c2 * dUdx_1 ** 2 * dx ** 4 * n ** 3 + 4 * c1 * dVdx_0 ** 2 * dx ** 4 * n ** 3 + 4 * c1 * dVdx_1 ** 2 * dx ** 4 * n ** 3 + 2 * d * dUdx_0 ** 2 * dx ** 4 * n ** 3 - d * dUdx_0 ** 3 * dx ** 3 * n ** 3 + 2 * d * dUdx_1 ** 2 * dx ** 4 * n ** 3 - d * dUdx_1 ** 3 * dx ** 3 * n ** 3 - d * dUdx_0 * dUdx_1 ** 2 * dVdy_0 ** 3 - 2 * d * dUdx_0 * dUdy_0 ** 2 * dVdx_1 ** 3 + 2 * d * dUdx_1 * dUdy_0 ** 2 * dVdx_0 ** 3 - d * dUdx_0 ** 2 * dUdx_1 * dVdy_0 ** 3 + 2 * d * dUdx_0 ** 2 * dUdy_0 * dVdx_1 ** 3 + 2 * d * dUdx_1 ** 2 * dUdy_0 * dVdx_0 ** 3 - 2 * d * dUdx_0 ** 3 * dVdx_1 * dVdy_0 ** 2 - 2 * d * dUdx_0 ** 3 * dVdx_1 ** 2 * dVdy_0 + 2 * d * dUdx_1 ** 3 * dVdx_0 * dVdy_0 ** 2 - 2 * d * dUdx_1 ** 3 * dVdx_0 ** 2 * dVdy_0 + d * dUdy_0 ** 3 * dVdx_0 * dVdx_1 ** 2 + d * dUdy_0 ** 3 * dVdx_0 ** 2 * dVdx_1 - 3 * d * dUdx_0 * dVdx_1 ** 3 * dx ** 2 - 6 * d * dUdx_0 * dVdy_0 ** 3 * dx ** 2 + 3 * d * dUdx_1 * dVdx_0 ** 3 * dx ** 2 + 2 * d * dUdy_0 * dVdx_0 ** 3 * dx ** 2 - 3 * d * dUdx_0 ** 2 * dVdx_1 ** 3 * dx - 4 * d * dUdx_0 ** 2 * dVdy_0 ** 3 * dx + 3 * d * dUdx_1 ** 2 * dVdx_0 ** 3 * dx + 2 * d * dUdy_0 ** 2 * dVdx_0 ** 3 * dx - 6 * d * dUdx_1 * dVdy_0 ** 3 * dx ** 2 + 2 * d * dUdy_0 * dVdx_1 ** 3 * dx ** 2 - 4 * d * dUdx_1 ** 2 * dVdy_0 ** 3 * dx - 2 * d * dUdy_0 ** 2 * dVdx_1 ** 3 * dx + 3 * d * dVdx_0 * dVdx_1 ** 2 * dx ** 3 + 6 * d * dVdx_0 * dVdy_0 ** 2 * dx ** 3 - 3 * d * dVdx_0 ** 2 * dVdx_1 * dx ** 3 - 4 * d * dVdx_0 ** 2 * dVdy_0 * dx ** 3 - 6 * d * dVdx_1 * dVdy_0 ** 2 * dx ** 3 - 4 * d * dVdx_1 ** 2 * dVdy_0 * dx ** 3 + 4 * c1 * dUdx_0 ** 2 * dx ** 4 * n + 4 * c1 * dUdx_1 ** 2 * dx ** 4 * n + 12 * c1 * dUdy_0 ** 2 * dx ** 4 * n + 12 * c1 * dUdx_0 * dx ** 5 * n ** 3 + 12 * c1 * dUdx_1 * dx ** 5 * n ** 3 + 12 * c2 * dUdx_0 * dx ** 5 * n ** 3 + 12 * c2 * dUdx_1 * dx ** 5 * n ** 3 + 4 * c * dVdx_0 ** 2 * dx ** 4 * n + 4 * c * dVdx_1 ** 2 * dx ** 4 * n + 12 * c * dVdy_0 ** 2 * dx ** 4 * n + 4 * c1 * dVdx_0 ** 2 * dx ** 4 * n - 12 * c1 * dVdx_0 * dx ** 5 * n ** 2 + 4 * c1 * dVdx_1 ** 2 * dx ** 4 * n + 12 * c1 * dVdy_0 ** 2 * dx ** 4 * n + 4 * c2 * dVdx_0 ** 2 * dx ** 4 * n + 12 * c1 * dVdx_1 * dx ** 5 * n ** 2 + 24 * c1 * dVdy_0 * dx ** 5 * n ** 2 - 12 * c2 * dVdx_0 * dx ** 5 * n ** 2 + 4 * c2 * dVdx_1 ** 2 * dx ** 4 * n + 12 * c2 * dVdy_0 ** 2 * dx ** 4 * n + 12 * c2 * dVdx_1 * dx ** 5 * n ** 2 + 24 * c2 * dVdy_0 * dx ** 5 * n ** 2 - 6 * d * dUdx_0 * dx ** 5 * n ** 3 - 6 * d * dUdx_1 * dx ** 5 * n ** 3 + 2 * d * dVdx_0 ** 2 * dx ** 4 * n + 6 * d * dVdx_0 * dx ** 5 * n ** 2 + 2 * d * dVdx_1 ** 2 * dx ** 4 * n + 6 * d * dVdy_0 ** 2 * dx ** 4 * n - 6 * d * dVdx_1 * dx ** 5 * n ** 2 - 12 * d * dVdy_0 * dx ** 5 * n ** 2 + 2 * d * dUdx_0 * dUdx_1 ** 2 * dVdx_0 * dVdy_0 ** 2 - 3 * d * dUdx_0 * dUdx_1 ** 2 * dVdx_0 ** 2 * dVdx_1 - 2 * d * dUdx_0 * dUdx_1 ** 2 * dVdx_0 ** 2 * dVdy_0 - 2 * d * dUdx_0 * dUdy_0 ** 2 * dVdx_0 * dVdx_1 ** 2 - 2 * d * dUdx_0 * dUdy_0 ** 2 * dVdx_0 ** 2 * dVdx_1 - 3 * d * dUdx_0 * dUdy_0 ** 2 * dVdx_0 ** 2 * dVdy_0 + 3 * d * dUdx_0 ** 2 * dUdx_1 * dVdx_0 * dVdx_1 ** 2 + 2 * d * dUdx_0 ** 2 * dUdx_1 * dVdx_0 * dVdy_0 ** 2 + 2 * d * dUdx_0 ** 2 * dUdy_0 * dVdx_0 * dVdx_1 ** 2 + 3 * d * dUdx_0 ** 2 * dUdy_0 * dVdx_0 * dVdy_0 ** 2 - 2 * d * dUdx_0 * dUdx_1 ** 2 * dVdx_1 * dVdy_0 ** 2 - d * dUdx_0 * dUdy_0 ** 2 * dVdx_1 ** 2 * dVdy_0 + 2 * d * dUdx_1 * dUdy_0 ** 2 * dVdx_0 * dVdx_1 ** 2 + 2 * d * dUdx_1 * dUdy_0 ** 2 * dVdx_0 ** 2 * dVdx_1 - d * dUdx_1 * dUdy_0 ** 2 * dVdx_0 ** 2 * dVdy_0 - 2 * d * dUdx_0 ** 2 * dUdx_1 * dVdx_1 * dVdy_0 ** 2 - 2 * d * dUdx_0 ** 2 * dUdx_1 * dVdx_1 ** 2 * dVdy_0 + d * dUdx_0 ** 2 * dUdy_0 * dVdx_1 * dVdy_0 ** 2 + 2 * d * dUdx_0 ** 2 * dUdy_0 * dVdx_1 ** 2 * dVdy_0 + d * dUdx_1 ** 2 * dUdy_0 * dVdx_0 * dVdy_0 ** 2 + 2 * d * dUdx_1 ** 2 * dUdy_0 * dVdx_0 ** 2 * dVdx_1 - 2 * d * dUdx_1 ** 2 * dUdy_0 * dVdx_0 ** 2 * dVdy_0 - 3 * d * dUdx_1 * dUdy_0 ** 2 * dVdx_1 ** 2 * dVdy_0 + 3 * d * dUdx_1 ** 2 * dUdy_0 * dVdx_1 * dVdy_0 ** 2 + 6 * d * dUdx_0 * dVdx_0 * dVdx_1 ** 2 * dx ** 2 + 6 * d * dUdx_0 * dVdx_0 * dVdy_0 ** 2 * dx ** 2 - 3 * d * dUdx_0 * dVdx_0 ** 2 * dVdx_1 * dx ** 2 - 2 * d * dUdx_0 * dVdx_0 ** 2 * dVdy_0 * dx ** 2 + 3 * d * dUdx_0 ** 2 * dVdx_0 * dVdx_1 ** 2 * dx + 2 * d * dUdx_0 ** 2 * dVdx_0 * dVdy_0 ** 2 * dx - 12 * d * dUdx_0 * dVdx_1 * dVdy_0 ** 2 * dx ** 2 - 10 * d * dUdx_0 * dVdx_1 ** 2 * dVdy_0 * dx ** 2 + 3 * d * dUdx_1 * dVdx_0 * dVdx_1 ** 2 * dx ** 2 + 12 * d * dUdx_1 * dVdx_0 * dVdy_0 ** 2 * dx ** 2 - 6 * d * dUdx_1 * dVdx_0 ** 2 * dVdx_1 * dx ** 2 - 10 * d * dUdx_1 * dVdx_0 ** 2 * dVdy_0 * dx ** 2 - 2 * d * dUdy_0 * dVdx_0 * dVdx_1 ** 2 * dx ** 2 + 6 * d * dUdy_0 * dVdx_0 * dVdy_0 ** 2 * dx ** 2 - 2 * d * dUdy_0 * dVdx_0 ** 2 * dVdx_1 * dx ** 2 - 6 * d * dUdy_0 * dVdx_0 ** 2 * dVdy_0 * dx ** 2 - 8 * d * dUdx_0 ** 2 * dVdx_1 * dVdy_0 ** 2 * dx - 8 * d * dUdx_0 ** 2 * dVdx_1 ** 2 * dVdy_0 * dx + 8 * d * dUdx_1 ** 2 * dVdx_0 * dVdy_0 ** 2 * dx - 3 * d * dUdx_1 ** 2 * dVdx_0 ** 2 * dVdx_1 * dx - 8 * d * dUdx_1 ** 2 * dVdx_0 ** 2 * dVdy_0 * dx - 4 * d * dUdy_0 ** 2 * dVdx_0 ** 2 * dVdy_0 * dx - 6 * d * dUdx_1 * dVdx_1 * dVdy_0 ** 2 * dx ** 2 - 2 * d * dUdx_1 * dVdx_1 ** 2 * dVdy_0 * dx ** 2 + 6 * d * dUdy_0 * dVdx_1 * dVdy_0 ** 2 * dx ** 2 + 6 * d * dUdy_0 * dVdx_1 ** 2 * dVdy_0 * dx ** 2 - 2 * d * dUdx_1 ** 2 * dVdx_1 * dVdy_0 ** 2 * dx - 4 * d * dUdy_0 ** 2 * dVdx_1 ** 2 * dVdy_0 * dx + 4 * c * dUdx_0 * dUdx_1 * dx ** 4 * n ** 3 + 4 * c1 * dUdx_0 * dUdx_1 * dx ** 4 * n ** 3 + 4 * c2 * dUdx_0 * dUdx_1 * dx ** 4 * n ** 3 - 6 * c * dUdx_0 * dVdx_0 * dx ** 4 * n ** 2 + 8 * c * dUdx_0 * dVdx_1 ** 2 * dx ** 3 * n + 12 * c * dUdx_0 * dVdy_0 ** 2 * dx ** 3 * n + 8 * c * dUdx_1 * dVdx_0 ** 2 * dx ** 3 * n + 6 * c * dUdy_0 * dVdx_0 ** 2 * dx ** 3 * n + 6 * c * dUdx_0 * dVdx_1 * dx ** 4 * n ** 2 + 12 * c * dUdx_0 * dVdy_0 * dx ** 4 * n ** 2 - 6 * c * dUdx_1 * dVdx_0 * dx ** 4 * n ** 2 + 12 * c * dUdx_1 * dVdy_0 ** 2 * dx ** 3 * n - 6 * c * dUdy_0 * dVdx_1 ** 2 * dx ** 3 * n + 6 * c * dUdx_1 * dVdx_1 * dx ** 4 * n ** 2 + 12 * c * dUdx_1 * dVdy_0 * dx ** 4 * n ** 2 - 6 * c2 * dUdx_0 * dVdx_0 * dx ** 4 * n ** 2 + 8 * c2 * dUdx_0 * dVdx_1 ** 2 * dx ** 3 * n + 12 * c2 * dUdx_0 * dVdy_0 ** 2 * dx ** 3 * n + 8 * c2 * dUdx_1 * dVdx_0 ** 2 * dx ** 3 * n + 6 * c2 * dUdy_0 * dVdx_0 ** 2 * dx ** 3 * n + 18 * c2 * dUdx_0 * dVdx_1 * dx ** 4 * n ** 2 + 24 * c2 * dUdx_0 * dVdy_0 * dx ** 4 * n ** 2 - 18 * c2 * dUdx_1 * dVdx_0 * dx ** 4 * n ** 2 + 12 * c2 * dUdx_1 * dVdy_0 ** 2 * dx ** 3 * n - 12 * c2 * dUdy_0 * dVdx_0 * dx ** 4 * n ** 2 - 6 * c2 * dUdy_0 * dVdx_1 ** 2 * dx ** 3 * n + 6 * c2 * dUdx_1 * dVdx_1 * dx ** 4 * n ** 2 + 24 * c2 * dUdx_1 * dVdy_0 * dx ** 4 * n ** 2 - 12 * c2 * dUdy_0 * dVdx_1 * dx ** 4 * n ** 2 + 4 * c1 * dVdx_0 * dVdx_1 * dx ** 4 * n ** 3 + 2 * d * dUdx_0 * dUdx_1 * dx ** 4 * n ** 3 - 2 * d * dUdx_0 * dVdx_0 ** 2 * dx ** 3 * n - 3 * d * dUdx_0 * dVdx_0 * dx ** 4 * n ** 2 + 2 * d * dUdx_0 * dVdx_1 ** 2 * dx ** 3 * n + 2 * d * dUdx_1 * dVdx_0 ** 2 * dx ** 3 * n + 3 * d * dUdy_0 * dVdx_0 ** 2 * dx ** 3 * n - 2 * d * dUdx_0 ** 3 * dVdx_1 ** 2 * dx * n - 3 * d * dUdx_0 ** 3 * dVdy_0 ** 2 * dx * n - 2 * d * dUdx_1 ** 3 * dVdx_0 ** 2 * dx * n - 3 * d * dUdx_0 * dVdx_1 * dx ** 4 * n ** 2 + 3 * d * dUdx_1 * dVdx_0 * dx ** 4 * n ** 2 - 2 * d * dUdx_1 * dVdx_1 ** 2 * dx ** 3 * n + 6 * d * dUdy_0 * dVdx_0 * dx ** 4 * n ** 2 - 3 * d * dUdy_0 * dVdx_1 ** 2 * dx ** 3 * n - 3 * d * dUdx_1 ** 3 * dVdy_0 ** 2 * dx * n + 3 * d * dUdx_1 * dVdx_1 * dx ** 4 * n ** 2 + 6 * d * dUdy_0 * dVdx_1 * dx ** 4 * n ** 2 + 4 * c * dUdx_0 ** 2 * dVdx_1 ** 2 * dx ** 2 * n + 4 * c * dUdx_0 ** 2 * dVdy_0 ** 2 * dx ** 2 * n + 4 * c * dUdx_1 ** 2 * dVdx_0 ** 2 * dx ** 2 * n + 4 * c * dUdy_0 ** 2 * dVdx_0 ** 2 * dx ** 2 * n + 6 * c * dUdx_0 ** 2 * dVdx_1 * dx ** 3 * n ** 2 + 8 * c * dUdx_0 ** 2 * dVdy_0 * dx ** 3 * n ** 2 - 6 * c * dUdx_1 ** 2 * dVdx_0 * dx ** 3 * n ** 2 + 4 * c * dUdx_1 ** 2 * dVdy_0 ** 2 * dx ** 2 * n + 4 * c * dUdy_0 ** 2 * dVdx_1 ** 2 * dx ** 2 * n + 8 * c * dUdx_1 ** 2 * dVdy_0 * dx ** 3 * n ** 2 + 4 * c2 * dUdx_0 ** 2 * dVdx_1 ** 2 * dx ** 2 * n + 4 * c2 * dUdx_0 ** 2 * dVdy_0 ** 2 * dx ** 2 * n + 4 * c2 * dUdx_1 ** 2 * dVdx_0 ** 2 * dx ** 2 * n + 4 * c2 * dUdy_0 ** 2 * dVdx_0 ** 2 * dx ** 2 * n + 6 * c2 * dUdx_0 ** 2 * dVdx_1 * dx ** 3 * n ** 2 + 8 * c2 * dUdx_0 ** 2 * dVdy_0 * dx ** 3 * n ** 2 - 6 * c2 * dUdx_1 ** 2 * dVdx_0 * dx ** 3 * n ** 2 + 4 * c2 * dUdx_1 ** 2 * dVdy_0 ** 2 * dx ** 2 * n + 4 * c2 * dUdy_0 ** 2 * dVdx_1 ** 2 * dx ** 2 * n + 8 * c2 * dUdx_1 ** 2 * dVdy_0 * dx ** 3 * n ** 2 - d * dUdx_0 * dUdx_1 ** 2 * dx ** 3 * n ** 3 - d * dUdx_0 ** 2 * dUdx_1 * dx ** 3 * n ** 3 + 2 * d * dUdx_0 ** 2 * dVdx_0 * dx ** 3 * n ** 2 - 2 * d * dUdx_0 ** 2 * dVdx_1 ** 2 * dx ** 2 * n - 6 * d * dUdx_0 ** 2 * dVdy_0 ** 2 * dx ** 2 * n - 2 * d * dUdx_1 ** 2 * dVdx_0 ** 2 * dx ** 2 * n + 2 * d * dUdy_0 ** 2 * dVdx_0 ** 2 * dx ** 2 * n + d * dUdx_0 ** 2 * dVdx_1 * dx ** 3 * n ** 2 - 2 * d * dUdx_0 ** 3 * dVdx_1 * dx ** 2 * n ** 2 - 3 * d * dUdx_0 ** 3 * dVdy_0 * dx ** 2 * n ** 2 - d * dUdx_1 ** 2 * dVdx_0 * dx ** 3 * n ** 2 - 6 * d * dUdx_1 ** 2 * dVdy_0 ** 2 * dx ** 2 * n + 2 * d * dUdy_0 ** 2 * dVdx_1 ** 2 * dx ** 2 * n + 2 * d * dUdx_1 ** 3 * dVdx_0 * dx ** 2 * n ** 2 - 2 * d * dUdx_1 ** 2 * dVdx_1 * dx ** 3 * n ** 2 - 3 * d * dUdx_1 ** 3 * dVdy_0 * dx ** 2 * n ** 2 - 4 * d * dUdx_0 * dUdx_1 * dVdy_0 ** 3 * dx + 4 * d * dUdx_0 * dUdy_0 * dVdx_1 ** 3 * dx + 4 * d * dUdx_1 * dUdy_0 * dVdx_0 ** 3 * dx + 8 * d * dVdx_0 * dVdx_1 * dVdy_0 * dx ** 3 - 8 * c1 * dUdx_0 * dUdx_1 * dx ** 4 * n - 12 * c1 * dUdx_0 * dUdy_0 * dx ** 4 * n + 12 * c1 * dUdx_1 * dUdy_0 * dx ** 4 * n - 8 * c * dVdx_0 * dVdx_1 * dx ** 4 * n - 12 * c * dVdx_0 * dVdy_0 * dx ** 4 * n + 12 * c * dVdx_1 * dVdy_0 * dx ** 4 * n - 8 * c1 * dVdx_0 * dVdx_1 * dx ** 4 * n - 12 * c1 * dVdx_0 * dVdy_0 * dx ** 4 * n + 12 * c1 * dVdx_1 * dVdy_0 * dx ** 4 * n - 8 * c2 * dVdx_0 * dVdx_1 * dx ** 4 * n - 12 * c2 * dVdx_0 * dVdy_0 * dx ** 4 * n + 12 * c2 * dVdx_1 * dVdy_0 * dx ** 4 * n - 4 * d * dVdx_0 * dVdx_1 * dx ** 4 * n - 6 * d * dVdx_0 * dVdy_0 * dx ** 4 * n + 6 * d * dVdx_1 * dVdy_0 * dx ** 4 * n - 4 * d * dUdx_0 * dUdx_1 * dUdy_0 * dVdx_0 * dVdx_1 ** 2 + 2 * d * dUdx_0 * dUdx_1 * dUdy_0 * dVdx_0 * dVdy_0 ** 2 - 4 * d * dUdx_0 * dUdx_1 * dUdy_0 * dVdx_0 ** 2 * dVdx_1 - 4 * d * dUdx_0 * dUdx_1 * dUdy_0 * dVdx_0 ** 2 * dVdy_0 + 2 * d * dUdx_0 * dUdx_1 * dUdy_0 * dVdx_1 * dVdy_0 ** 2 + 4 * d * dUdx_0 * dUdx_1 * dUdy_0 * dVdx_1 ** 2 * dVdy_0 + 4 * d * dUdx_0 * dUdx_1 ** 2 * dVdx_0 * dVdx_1 * dVdy_0 - 2 * d * dUdx_0 * dUdy_0 ** 2 * dVdx_0 * dVdx_1 * dVdy_0 + 4 * d * dUdx_0 ** 2 * dUdx_1 * dVdx_0 * dVdx_1 * dVdy_0 + 4 * d * dUdx_0 ** 2 * dUdy_0 * dVdx_0 * dVdx_1 * dVdy_0 - 2 * d * dUdx_1 * dUdy_0 ** 2 * dVdx_0 * dVdx_1 * dVdy_0 - 4 * d * dUdx_1 ** 2 * dUdy_0 * dVdx_0 * dVdx_1 * dVdy_0 + 6 * d * dUdx_0 * dUdx_1 * dVdx_0 * dVdx_1 ** 2 * dx + 8 * d * dUdx_0 * dUdx_1 * dVdx_0 * dVdy_0 ** 2 * dx - 6 * d * dUdx_0 * dUdx_1 * dVdx_0 ** 2 * dVdx_1 * dx - 4 * d * dUdx_0 * dUdx_1 * dVdx_0 ** 2 * dVdy_0 * dx + 8 * d * dUdx_0 * dUdy_0 * dVdx_0 * dVdy_0 ** 2 * dx - 4 * d * dUdx_0 * dUdy_0 * dVdx_0 ** 2 * dVdx_1 * dx - 4 * d * dUdx_0 * dUdy_0 * dVdx_0 ** 2 * dVdy_0 * dx - 8 * d * dUdx_0 * dUdx_1 * dVdx_1 * dVdy_0 ** 2 * dx - 4 * d * dUdx_0 * dUdx_1 * dVdx_1 ** 2 * dVdy_0 * dx + 4 * d * dUdx_0 * dUdy_0 * dVdx_1 * dVdy_0 ** 2 * dx + 8 * d * dUdx_0 * dUdy_0 * dVdx_1 ** 2 * dVdy_0 * dx - 4 * d * dUdx_1 * dUdy_0 * dVdx_0 * dVdx_1 ** 2 * dx + 4 * d * dUdx_1 * dUdy_0 * dVdx_0 * dVdy_0 ** 2 * dx - 8 * d * dUdx_1 * dUdy_0 * dVdx_0 ** 2 * dVdy_0 * dx + 8 * d * dUdx_1 * dUdy_0 * dVdx_1 * dVdy_0 ** 2 * dx + 4 * d * dUdx_1 * dUdy_0 * dVdx_1 ** 2 * dVdy_0 * dx + 12 * d * dUdx_0 * dVdx_0 * dVdx_1 * dVdy_0 * dx ** 2 + 4 * d * dUdx_0 ** 2 * dVdx_0 * dVdx_1 * dVdy_0 * dx + 12 * d * dUdx_1 * dVdx_0 * dVdx_1 * dVdy_0 * dx ** 2 + 4 * d * dUdx_1 ** 2 * dVdx_0 * dVdx_1 * dVdy_0 * dx - 4 * d * dUdy_0 ** 2 * dVdx_0 * dVdx_1 * dVdy_0 * dx - 8 * c * dUdx_0 * dVdx_0 * dVdx_1 * dx ** 3 * n - 6 * c * dUdx_0 * dVdx_0 * dVdy_0 * dx ** 3 * n + 18 * c * dUdx_0 * dVdx_1 * dVdy_0 * dx ** 3 * n - 8 * c * dUdx_1 * dVdx_0 * dVdx_1 * dx ** 3 * n - 18 * c * dUdx_1 * dVdx_0 * dVdy_0 * dx ** 3 * n - 12 * c * dUdy_0 * dVdx_0 * dVdy_0 * dx ** 3 * n + 6 * c * dUdx_1 * dVdx_1 * dVdy_0 * dx ** 3 * n - 12 * c * dUdy_0 * dVdx_1 * dVdy_0 * dx ** 3 * n - 8 * c2 * dUdx_0 * dVdx_0 * dVdx_1 * dx ** 3 * n - 6 * c2 * dUdx_0 * dVdx_0 * dVdy_0 * dx ** 3 * n + 18 * c2 * dUdx_0 * dVdx_1 * dVdy_0 * dx ** 3 * n - 8 * c2 * dUdx_1 * dVdx_0 * dVdx_1 * dx ** 3 * n - 18 * c2 * dUdx_1 * dVdx_0 * dVdy_0 * dx ** 3 * n - 12 * c2 * dUdy_0 * dVdx_0 * dVdy_0 * dx ** 3 * n + 6 * c2 * dUdx_1 * dVdx_1 * dVdy_0 * dx ** 3 * n - 12 * c2 * dUdy_0 * dVdx_1 * dVdy_0 * dx ** 3 * n + 3 * d * dUdx_0 * dVdx_0 * dVdy_0 * dx ** 3 * n + 3 * d * dUdx_0 * dVdx_1 * dVdy_0 * dx ** 3 * n - 3 * d * dUdx_1 * dVdx_0 * dVdy_0 * dx ** 3 * n - 6 * d * dUdy_0 * dVdx_0 * dVdy_0 * dx ** 3 * n - 4 * d * dUdx_0 ** 3 * dVdx_1 * dVdy_0 * dx * n + 4 * d * dUdx_1 ** 3 * dVdx_0 * dVdy_0 * dx * n - 3 * d * dUdx_1 * dVdx_1 * dVdy_0 * dx ** 3 * n - 6 * d * dUdy_0 * dVdx_1 * dVdy_0 * dx ** 3 * n - 6 * c * dUdx_0 * dUdx_1 * dVdx_0 * dx ** 3 * n ** 2 + 4 * c * dUdx_0 * dUdx_1 * dVdy_0 ** 2 * dx ** 2 * n - 8 * c * dUdx_0 * dUdy_0 * dVdx_0 * dx ** 3 * n ** 2 - 6 * c * dUdx_0 * dUdy_0 * dVdx_1 ** 2 * dx ** 2 * n + 6 * c * dUdx_1 * dUdy_0 * dVdx_0 ** 2 * dx ** 2 * n + 6 * c * dUdx_0 * dUdx_1 * dVdx_1 * dx ** 3 * n ** 2 + 8 * c * dUdx_0 * dUdx_1 * dVdy_0 * dx ** 3 * n ** 2 - 4 * c * dUdx_0 * dUdy_0 * dVdx_1 * dx ** 3 * n ** 2 - 4 * c * dUdx_1 * dUdy_0 * dVdx_0 * dx ** 3 * n ** 2 - 8 * c * dUdx_1 * dUdy_0 * dVdx_1 * dx ** 3 * n ** 2 - 6 * c2 * dUdx_0 * dUdx_1 * dVdx_0 * dx ** 3 * n ** 2 + 4 * c2 * dUdx_0 * dUdx_1 * dVdy_0 ** 2 * dx ** 2 * n - 8 * c2 * dUdx_0 * dUdy_0 * dVdx_0 * dx ** 3 * n ** 2 - 6 * c2 * dUdx_0 * dUdy_0 * dVdx_1 ** 2 * dx ** 2 * n + 6 * c2 * dUdx_1 * dUdy_0 * dVdx_0 ** 2 * dx ** 2 * n + 6 * c2 * dUdx_0 * dUdx_1 * dVdx_1 * dx ** 3 * n ** 2 + 8 * c2 * dUdx_0 * dUdx_1 * dVdy_0 * dx ** 3 * n ** 2 - 4 * c2 * dUdx_0 * dUdy_0 * dVdx_1 * dx ** 3 * n ** 2 - 4 * c2 * dUdx_1 * dUdy_0 * dVdx_0 * dx ** 3 * n ** 2 - 8 * c2 * dUdx_1 * dUdy_0 * dVdx_1 * dx ** 3 * n ** 2 + 6 * c * dUdx_0 ** 2 * dVdx_1 * dVdy_0 * dx ** 2 * n - 6 * c * dUdx_1 ** 2 * dVdx_0 * dVdy_0 * dx ** 2 * n + 4 * c * dUdy_0 ** 2 * dVdx_0 * dVdx_1 * dx ** 2 * n + 6 * c2 * dUdx_0 ** 2 * dVdx_1 * dVdy_0 * dx ** 2 * n - 6 * c2 * dUdx_1 ** 2 * dVdx_0 * dVdy_0 * dx ** 2 * n + 4 * c2 * dUdy_0 ** 2 * dVdx_0 * dVdx_1 * dx ** 2 * n - 4 * d * dUdx_0 * dUdx_1 * dVdx_0 ** 2 * dx ** 2 * n - 4 * d * dUdx_0 * dUdy_0 * dVdx_0 ** 2 * dx ** 2 * n - 2 * d * dUdx_0 * dUdx_1 ** 2 * dVdx_0 ** 2 * dx * n - 3 * d * dUdx_0 * dUdy_0 ** 2 * dVdx_0 ** 2 * dx * n - d * dUdx_0 * dUdx_1 * dVdx_0 * dx ** 3 * n ** 2 - 4 * d * dUdx_0 * dUdx_1 * dVdx_1 ** 2 * dx ** 2 * n - 6 * d * dUdx_0 * dUdx_1 * dVdy_0 ** 2 * dx ** 2 * n - 4 * d * dUdx_0 * dUdy_0 * dVdx_0 * dx ** 3 * n ** 2 - d * dUdx_0 * dUdy_0 * dVdx_1 ** 2 * dx ** 2 * n - 3 * d * dUdx_0 * dUdx_1 ** 2 * dVdy_0 ** 2 * dx * n - d * dUdx_0 * dUdy_0 ** 2 * dVdx_1 ** 2 * dx * n + d * dUdx_1 * dUdy_0 * dVdx_0 ** 2 * dx ** 2 * n - d * dUdx_1 * dUdy_0 ** 2 * dVdx_0 ** 2 * dx * n - 2 * d * dUdx_0 ** 2 * dUdx_1 * dVdx_1 ** 2 * dx * n - 3 * d * dUdx_0 ** 2 * dUdx_1 * dVdy_0 ** 2 * dx * n + 2 * d * dUdx_0 ** 2 * dUdy_0 * dVdx_1 ** 2 * dx * n - 2 * d * dUdx_1 ** 2 * dUdy_0 * dVdx_0 ** 2 * dx * n + d * dUdx_0 * dUdx_1 * dVdx_1 * dx ** 3 * n ** 2 - 2 * d * dUdx_0 * dUdy_0 * dVdx_1 * dx ** 3 * n ** 2 - 2 * d * dUdx_1 * dUdy_0 * dVdx_0 * dx ** 3 * n ** 2 + 4 * d * dUdx_1 * dUdy_0 * dVdx_1 ** 2 * dx ** 2 * n - 3 * d * dUdx_1 * dUdy_0 ** 2 * dVdx_1 ** 2 * dx * n - 4 * d * dUdx_1 * dUdy_0 * dVdx_1 * dx ** 3 * n ** 2 + 4 * d * dUdx_0 ** 2 * dVdx_0 * dVdx_1 * dx ** 2 * n + 4 * d * dUdx_0 ** 2 * dVdx_0 * dVdy_0 * dx ** 2 * n - 7 * d * dUdx_0 ** 2 * dVdx_1 * dVdy_0 * dx ** 2 * n + 4 * d * dUdx_1 ** 2 * dVdx_0 * dVdx_1 * dx ** 2 * n + 7 * d * dUdx_1 ** 2 * dVdx_0 * dVdy_0 * dx ** 2 * n + 2 * d * dUdy_0 ** 2 * dVdx_0 * dVdx_1 * dx ** 2 * n - 4 * d * dUdx_1 ** 2 * dVdx_1 * dVdy_0 * dx ** 2 * n + 2 * d * dUdx_0 * dUdx_1 ** 2 * dVdx_0 * dx ** 2 * n ** 2 + 2 * d * dUdx_0 ** 2 * dUdx_1 * dVdx_0 * dx ** 2 * n ** 2 + 3 * d * dUdx_0 ** 2 * dUdy_0 * dVdx_0 * dx ** 2 * n ** 2 - 2 * d * dUdx_0 * dUdx_1 ** 2 * dVdx_1 * dx ** 2 * n ** 2 - 3 * d * dUdx_0 * dUdx_1 ** 2 * dVdy_0 * dx ** 2 * n ** 2 - 2 * d * dUdx_0 ** 2 * dUdx_1 * dVdx_1 * dx ** 2 * n ** 2 - 3 * d * dUdx_0 ** 2 * dUdx_1 * dVdy_0 * dx ** 2 * n ** 2 + d * dUdx_0 ** 2 * dUdy_0 * dVdx_1 * dx ** 2 * n ** 2 + d * dUdx_1 ** 2 * dUdy_0 * dVdx_0 * dx ** 2 * n ** 2 + 3 * d * dUdx_1 ** 2 * dUdy_0 * dVdx_1 * dx ** 2 * n ** 2 + 16 * d * dUdx_0 * dUdx_1 * dVdx_0 * dVdx_1 * dVdy_0 * dx + 8 * d * dUdx_0 * dUdy_0 * dVdx_0 * dVdx_1 * dVdy_0 * dx - 8 * d * dUdx_1 * dUdy_0 * dVdx_0 * dVdx_1 * dVdy_0 * dx - 8 * c * dUdx_0 * dUdx_1 * dVdx_0 * dVdx_1 * dx ** 2 * n - 6 * c * dUdx_0 * dUdx_1 * dVdx_0 * dVdy_0 * dx ** 2 * n - 6 * c * dUdx_0 * dUdy_0 * dVdx_0 * dVdx_1 * dx ** 2 * n - 8 * c * dUdx_0 * dUdy_0 * dVdx_0 * dVdy_0 * dx ** 2 * n + 6 * c * dUdx_0 * dUdx_1 * dVdx_1 * dVdy_0 * dx ** 2 * n - 4 * c * dUdx_0 * dUdy_0 * dVdx_1 * dVdy_0 * dx ** 2 * n + 6 * c * dUdx_1 * dUdy_0 * dVdx_0 * dVdx_1 * dx ** 2 * n - 4 * c * dUdx_1 * dUdy_0 * dVdx_0 * dVdy_0 * dx ** 2 * n - 8 * c * dUdx_1 * dUdy_0 * dVdx_1 * dVdy_0 * dx ** 2 * n - 8 * c2 * dUdx_0 * dUdx_1 * dVdx_0 * dVdx_1 * dx ** 2 * n - 6 * c2 * dUdx_0 * dUdx_1 * dVdx_0 * dVdy_0 * dx ** 2 * n - 6 * c2 * dUdx_0 * dUdy_0 * dVdx_0 * dVdx_1 * dx ** 2 * n - 8 * c2 * dUdx_0 * dUdy_0 * dVdx_0 * dVdy_0 * dx ** 2 * n + 6 * c2 * dUdx_0 * dUdx_1 * dVdx_1 * dVdy_0 * dx ** 2 * n - 4 * c2 * dUdx_0 * dUdy_0 * dVdx_1 * dVdy_0 * dx ** 2 * n + 6 * c2 * dUdx_1 * dUdy_0 * dVdx_0 * dVdx_1 * dx ** 2 * n - 4 * c2 * dUdx_1 * dUdy_0 * dVdx_0 * dVdy_0 * dx ** 2 * n - 8 * c2 * dUdx_1 * dUdy_0 * dVdx_1 * dVdy_0 * dx ** 2 * n - 4 * d * dUdx_0 * dUdx_1 * dUdy_0 * dVdx_0 ** 2 * dx * n + 4 * d * dUdx_0 * dUdx_1 * dUdy_0 * dVdx_1 ** 2 * dx * n + 4 * d * dUdx_0 * dUdx_1 * dVdx_0 * dVdx_1 * dx ** 2 * n + 7 * d * dUdx_0 * dUdx_1 * dVdx_0 * dVdy_0 * dx ** 2 * n - d * dUdx_0 * dUdy_0 * dVdx_0 * dVdx_1 * dx ** 2 * n + 4 * d * dUdx_0 * dUdy_0 * dVdx_0 * dVdy_0 * dx ** 2 * n + 4 * d * dUdx_0 * dUdx_1 ** 2 * dVdx_0 * dVdx_1 * dx * n + 4 * d * dUdx_0 * dUdx_1 ** 2 * dVdx_0 * dVdy_0 * dx * n - 2 * d * dUdx_0 * dUdy_0 ** 2 * dVdx_0 * dVdx_1 * dx * n + 4 * d * dUdx_0 ** 2 * dUdx_1 * dVdx_0 * dVdx_1 * dx * n + 4 * d * dUdx_0 ** 2 * dUdx_1 * dVdx_0 * dVdy_0 * dx * n + 4 * d * dUdx_0 ** 2 * dUdy_0 * dVdx_0 * dVdx_1 * dx * n + 6 * d * dUdx_0 ** 2 * dUdy_0 * dVdx_0 * dVdy_0 * dx * n - 7 * d * dUdx_0 * dUdx_1 * dVdx_1 * dVdy_0 * dx ** 2 * n + 2 * d * dUdx_0 * dUdy_0 * dVdx_1 * dVdy_0 * dx ** 2 * n - 4 * d * dUdx_0 * dUdx_1 ** 2 * dVdx_1 * dVdy_0 * dx * n + d * dUdx_1 * dUdy_0 * dVdx_0 * dVdx_1 * dx ** 2 * n + 2 * d * dUdx_1 * dUdy_0 * dVdx_0 * dVdy_0 * dx ** 2 * n - 2 * d * dUdx_1 * dUdy_0 ** 2 * dVdx_0 * dVdx_1 * dx * n - 4 * d * dUdx_0 ** 2 * dUdx_1 * dVdx_1 * dVdy_0 * dx * n + 2 * d * dUdx_0 ** 2 * dUdy_0 * dVdx_1 * dVdy_0 * dx * n - 4 * d * dUdx_1 ** 2 * dUdy_0 * dVdx_0 * dVdx_1 * dx * n + 2 * d * dUdx_1 ** 2 * dUdy_0 * dVdx_0 * dVdy_0 * dx * n + 4 * d * dUdx_1 * dUdy_0 * dVdx_1 * dVdy_0 * dx ** 2 * n + 6 * d * dUdx_1 ** 2 * dUdy_0 * dVdx_1 * dVdy_0 * dx * n + 2 * d * dUdx_0 * dUdx_1 * dUdy_0 * dVdx_0 * dx ** 2 * n ** 2 + 2 * d * dUdx_0 * dUdx_1 * dUdy_0 * dVdx_1 * dx ** 2 * n ** 2 + 4 * d * dUdx_0 * dUdx_1 * dUdy_0 * dVdx_0 * dVdy_0 * dx * n + 4 * d * dUdx_0 * dUdx_1 * dUdy_0 * dVdx_1 * dVdy_0 * dx * n) / (
        #                     12 * dx ** 4 * n ** 2))
        # Second-order Taylor polynomial
        #loss_int = jnp.sum((4 * c * dUdx_0 ** 2 * dVdx_1 ** 2 + 4 * c * dUdx_0 ** 2 * dVdy_0 ** 2 + 4 * c * dUdx_1 ** 2 * dVdx_0 ** 2 + 4 * c * dUdy_0 ** 2 * dVdx_0 ** 2 + 4 * c * dUdx_1 ** 2 * dVdy_0 ** 2 + 4 * c * dUdy_0 ** 2 * dVdx_1 ** 2 + 4 * c2 * dUdx_0 ** 2 * dVdx_1 ** 2 + 4 * c2 * dUdx_0 ** 2 * dVdy_0 ** 2 + 4 * c2 * dUdx_1 ** 2 * dVdx_0 ** 2 + 4 * c2 * dUdy_0 ** 2 * dVdx_0 ** 2 + 4 * c2 * dUdx_1 ** 2 * dVdy_0 ** 2 + 4 * c2 * dUdy_0 ** 2 * dVdx_1 ** 2 + 4 * c1 * dUdx_0 ** 2 * dx ** 2 + 4 * c1 * dUdx_1 ** 2 * dx ** 2 + 12 * c1 * dUdy_0 ** 2 * dx ** 2 + 4 * c * dUdx_0 ** 2 * dy ** 2 + 4 * c * dVdx_0 ** 2 * dx ** 2 + 4 * c * dUdx_1 ** 2 * dy ** 2 + 4 * c * dVdx_1 ** 2 * dx ** 2 + 12 * c * dVdy_0 ** 2 * dx ** 2 + 4 * c1 * dUdx_0 ** 2 * dy ** 2 + 4 * c1 * dVdx_0 ** 2 * dx ** 2 + 4 * c1 * dUdx_1 ** 2 * dy ** 2 + 4 * c1 * dVdx_1 ** 2 * dx ** 2 + 12 * c1 * dVdy_0 ** 2 * dx ** 2 + 4 * c2 * dUdx_0 ** 2 * dy ** 2 + 4 * c2 * dVdx_0 ** 2 * dx ** 2 + 4 * c2 * dUdx_1 ** 2 * dy ** 2 + 4 * c2 * dVdx_1 ** 2 * dx ** 2 + 12 * c2 * dVdy_0 ** 2 * dx ** 2 + 4 * c1 * dVdx_0 ** 2 * dy ** 2 + 4 * c1 * dVdx_1 ** 2 * dy ** 2 + 2 * d * dUdx_0 ** 2 * dVdx_1 ** 2 + 2 * d * dUdx_0 ** 2 * dVdy_0 ** 2 + 2 * d * dUdx_1 ** 2 * dVdx_0 ** 2 + 2 * d * dUdy_0 ** 2 * dVdx_0 ** 2 + 2 * d * dUdx_1 ** 2 * dVdy_0 ** 2 + 2 * d * dUdy_0 ** 2 * dVdx_1 ** 2 + 2 * d * dUdx_0 ** 2 * dy ** 2 + 2 * d * dVdx_0 ** 2 * dx ** 2 + 2 * d * dUdx_1 ** 2 * dy ** 2 + 2 * d * dVdx_1 ** 2 * dx ** 2 + 6 * d * dVdy_0 ** 2 * dx ** 2 + 4 * c * dUdx_0 * dUdx_1 * dVdy_0 ** 2 - 6 * c * dUdx_0 * dUdy_0 * dVdx_1 ** 2 + 6 * c * dUdx_1 * dUdy_0 * dVdx_0 ** 2 + 4 * c2 * dUdx_0 * dUdx_1 * dVdy_0 ** 2 - 6 * c2 * dUdx_0 * dUdy_0 * dVdx_1 ** 2 + 6 * c2 * dUdx_1 * dUdy_0 * dVdx_0 ** 2 + 6 * c * dUdx_0 ** 2 * dVdx_1 * dVdy_0 - 6 * c * dUdx_1 ** 2 * dVdx_0 * dVdy_0 + 4 * c * dUdy_0 ** 2 * dVdx_0 * dVdx_1 + 6 * c2 * dUdx_0 ** 2 * dVdx_1 * dVdy_0 - 6 * c2 * dUdx_1 ** 2 * dVdx_0 * dVdy_0 + 4 * c2 * dUdy_0 ** 2 * dVdx_0 * dVdx_1 - 8 * c1 * dUdx_0 * dUdx_1 * dx ** 2 - 12 * c1 * dUdx_0 * dUdy_0 * dx ** 2 + 12 * c1 * dUdx_1 * dUdy_0 * dx ** 2 + 4 * c * dUdx_0 * dUdx_1 * dy ** 2 + 8 * c * dUdx_0 * dVdx_1 ** 2 * dx + 12 * c * dUdx_0 * dVdy_0 ** 2 * dx + 8 * c * dUdx_1 * dVdx_0 ** 2 * dx + 6 * c * dUdy_0 * dVdx_0 ** 2 * dx + 12 * c * dUdx_1 * dVdy_0 ** 2 * dx - 6 * c * dUdy_0 * dVdx_1 ** 2 * dx + 4 * c1 * dUdx_0 * dUdx_1 * dy ** 2 + 4 * c2 * dUdx_0 * dUdx_1 * dy ** 2 + 8 * c2 * dUdx_0 * dVdx_1 ** 2 * dx + 12 * c2 * dUdx_0 * dVdy_0 ** 2 * dx + 8 * c2 * dUdx_1 * dVdx_0 ** 2 * dx + 6 * c2 * dUdy_0 * dVdx_0 ** 2 * dx + 12 * c2 * dUdx_1 * dVdy_0 ** 2 * dx - 6 * c2 * dUdy_0 * dVdx_1 ** 2 * dx - 8 * c * dVdx_0 * dVdx_1 * dx ** 2 - 12 * c * dVdx_0 * dVdy_0 * dx ** 2 + 6 * c * dUdx_0 ** 2 * dVdx_1 * dy + 8 * c * dUdx_0 ** 2 * dVdy_0 * dy - 6 * c * dUdx_1 ** 2 * dVdx_0 * dy + 12 * c * dVdx_1 * dVdy_0 * dx ** 2 + 8 * c * dUdx_1 ** 2 * dVdy_0 * dy - 8 * c1 * dVdx_0 * dVdx_1 * dx ** 2 - 12 * c1 * dVdx_0 * dVdy_0 * dx ** 2 + 12 * c1 * dVdx_1 * dVdy_0 * dx ** 2 - 8 * c2 * dVdx_0 * dVdx_1 * dx ** 2 - 12 * c2 * dVdx_0 * dVdy_0 * dx ** 2 + 6 * c2 * dUdx_0 ** 2 * dVdx_1 * dy + 8 * c2 * dUdx_0 ** 2 * dVdy_0 * dy - 6 * c2 * dUdx_1 ** 2 * dVdx_0 * dy + 12 * c2 * dVdx_1 * dVdy_0 * dx ** 2 + 8 * c2 * dUdx_1 ** 2 * dVdy_0 * dy + 4 * c1 * dVdx_0 * dVdx_1 * dy ** 2 + 12 * c1 * dUdx_0 * dx * dy ** 2 + 12 * c1 * dUdx_1 * dx * dy ** 2 + 12 * c2 * dUdx_0 * dx * dy ** 2 + 12 * c2 * dUdx_1 * dx * dy ** 2 - 12 * c1 * dVdx_0 * dx ** 2 * dy + 12 * c1 * dVdx_1 * dx ** 2 * dy + 24 * c1 * dVdy_0 * dx ** 2 * dy - 12 * c2 * dVdx_0 * dx ** 2 * dy + 12 * c2 * dVdx_1 * dx ** 2 * dy + 24 * c2 * dVdy_0 * dx ** 2 * dy + 2 * d * dUdx_0 * dUdx_1 * dVdy_0 ** 2 - 3 * d * dUdx_0 * dUdy_0 * dVdx_1 ** 2 + 3 * d * dUdx_1 * dUdy_0 * dVdx_0 ** 2 + 3 * d * dUdx_0 ** 2 * dVdx_1 * dVdy_0 - 3 * d * dUdx_1 ** 2 * dVdx_0 * dVdy_0 + 2 * d * dUdy_0 ** 2 * dVdx_0 * dVdx_1 + 2 * d * dUdx_0 * dUdx_1 * dy ** 2 + 4 * d * dUdx_0 * dVdx_1 ** 2 * dx + 6 * d * dUdx_0 * dVdy_0 ** 2 * dx + 4 * d * dUdx_1 * dVdx_0 ** 2 * dx + 3 * d * dUdy_0 * dVdx_0 ** 2 * dx + 6 * d * dUdx_1 * dVdy_0 ** 2 * dx - 3 * d * dUdy_0 * dVdx_1 ** 2 * dx - 4 * d * dVdx_0 * dVdx_1 * dx ** 2 - 6 * d * dVdx_0 * dVdy_0 * dx ** 2 + 3 * d * dUdx_0 ** 2 * dVdx_1 * dy + 4 * d * dUdx_0 ** 2 * dVdy_0 * dy - 3 * d * dUdx_1 ** 2 * dVdx_0 * dy + 6 * d * dVdx_1 * dVdy_0 * dx ** 2 + 4 * d * dUdx_1 ** 2 * dVdy_0 * dy - 6 * d * dUdx_0 * dx * dy ** 2 - 6 * d * dUdx_1 * dx * dy ** 2 + 6 * d * dVdx_0 * dx ** 2 * dy - 6 * d * dVdx_1 * dx ** 2 * dy - 12 * d * dVdy_0 * dx ** 2 * dy - 8 * c * dUdx_0 * dUdx_1 * dVdx_0 * dVdx_1 - 6 * c * dUdx_0 * dUdx_1 * dVdx_0 * dVdy_0 - 6 * c * dUdx_0 * dUdy_0 * dVdx_0 * dVdx_1 - 8 * c * dUdx_0 * dUdy_0 * dVdx_0 * dVdy_0 + 6 * c * dUdx_0 * dUdx_1 * dVdx_1 * dVdy_0 - 4 * c * dUdx_0 * dUdy_0 * dVdx_1 * dVdy_0 + 6 * c * dUdx_1 * dUdy_0 * dVdx_0 * dVdx_1 - 4 * c * dUdx_1 * dUdy_0 * dVdx_0 * dVdy_0 - 8 * c * dUdx_1 * dUdy_0 * dVdx_1 * dVdy_0 - 8 * c2 * dUdx_0 * dUdx_1 * dVdx_0 * dVdx_1 - 6 * c2 * dUdx_0 * dUdx_1 * dVdx_0 * dVdy_0 - 6 * c2 * dUdx_0 * dUdy_0 * dVdx_0 * dVdx_1 - 8 * c2 * dUdx_0 * dUdy_0 * dVdx_0 * dVdy_0 + 6 * c2 * dUdx_0 * dUdx_1 * dVdx_1 * dVdy_0 - 4 * c2 * dUdx_0 * dUdy_0 * dVdx_1 * dVdy_0 + 6 * c2 * dUdx_1 * dUdy_0 * dVdx_0 * dVdx_1 - 4 * c2 * dUdx_1 * dUdy_0 * dVdx_0 * dVdy_0 - 8 * c2 * dUdx_1 * dUdy_0 * dVdx_1 * dVdy_0 - 6 * c * dUdx_0 * dUdx_1 * dVdx_0 * dy - 8 * c * dUdx_0 * dUdy_0 * dVdx_0 * dy - 8 * c * dUdx_0 * dVdx_0 * dVdx_1 * dx - 6 * c * dUdx_0 * dVdx_0 * dVdy_0 * dx + 6 * c * dUdx_0 * dUdx_1 * dVdx_1 * dy + 8 * c * dUdx_0 * dUdx_1 * dVdy_0 * dy - 4 * c * dUdx_0 * dUdy_0 * dVdx_1 * dy + 18 * c * dUdx_0 * dVdx_1 * dVdy_0 * dx - 4 * c * dUdx_1 * dUdy_0 * dVdx_0 * dy - 8 * c * dUdx_1 * dVdx_0 * dVdx_1 * dx - 18 * c * dUdx_1 * dVdx_0 * dVdy_0 * dx - 12 * c * dUdy_0 * dVdx_0 * dVdy_0 * dx - 8 * c * dUdx_1 * dUdy_0 * dVdx_1 * dy + 6 * c * dUdx_1 * dVdx_1 * dVdy_0 * dx - 12 * c * dUdy_0 * dVdx_1 * dVdy_0 * dx - 6 * c2 * dUdx_0 * dUdx_1 * dVdx_0 * dy - 8 * c2 * dUdx_0 * dUdy_0 * dVdx_0 * dy - 8 * c2 * dUdx_0 * dVdx_0 * dVdx_1 * dx - 6 * c2 * dUdx_0 * dVdx_0 * dVdy_0 * dx + 6 * c2 * dUdx_0 * dUdx_1 * dVdx_1 * dy + 8 * c2 * dUdx_0 * dUdx_1 * dVdy_0 * dy - 4 * c2 * dUdx_0 * dUdy_0 * dVdx_1 * dy + 18 * c2 * dUdx_0 * dVdx_1 * dVdy_0 * dx - 4 * c2 * dUdx_1 * dUdy_0 * dVdx_0 * dy - 8 * c2 * dUdx_1 * dVdx_0 * dVdx_1 * dx - 18 * c2 * dUdx_1 * dVdx_0 * dVdy_0 * dx - 12 * c2 * dUdy_0 * dVdx_0 * dVdy_0 * dx - 8 * c2 * dUdx_1 * dUdy_0 * dVdx_1 * dy + 6 * c2 * dUdx_1 * dVdx_1 * dVdy_0 * dx - 12 * c2 * dUdy_0 * dVdx_1 * dVdy_0 * dx - 6 * c * dUdx_0 * dVdx_0 * dx * dy + 6 * c * dUdx_0 * dVdx_1 * dx * dy + 12 * c * dUdx_0 * dVdy_0 * dx * dy - 6 * c * dUdx_1 * dVdx_0 * dx * dy + 6 * c * dUdx_1 * dVdx_1 * dx * dy + 12 * c * dUdx_1 * dVdy_0 * dx * dy - 6 * c2 * dUdx_0 * dVdx_0 * dx * dy + 18 * c2 * dUdx_0 * dVdx_1 * dx * dy + 24 * c2 * dUdx_0 * dVdy_0 * dx * dy - 18 * c2 * dUdx_1 * dVdx_0 * dx * dy - 12 * c2 * dUdy_0 * dVdx_0 * dx * dy + 6 * c2 * dUdx_1 * dVdx_1 * dx * dy + 24 * c2 * dUdx_1 * dVdy_0 * dx * dy - 12 * c2 * dUdy_0 * dVdx_1 * dx * dy - 4 * d * dUdx_0 * dUdx_1 * dVdx_0 * dVdx_1 - 3 * d * dUdx_0 * dUdx_1 * dVdx_0 * dVdy_0 - 3 * d * dUdx_0 * dUdy_0 * dVdx_0 * dVdx_1 - 4 * d * dUdx_0 * dUdy_0 * dVdx_0 * dVdy_0 + 3 * d * dUdx_0 * dUdx_1 * dVdx_1 * dVdy_0 - 2 * d * dUdx_0 * dUdy_0 * dVdx_1 * dVdy_0 + 3 * d * dUdx_1 * dUdy_0 * dVdx_0 * dVdx_1 - 2 * d * dUdx_1 * dUdy_0 * dVdx_0 * dVdy_0 - 4 * d * dUdx_1 * dUdy_0 * dVdx_1 * dVdy_0 - 3 * d * dUdx_0 * dUdx_1 * dVdx_0 * dy - 4 * d * dUdx_0 * dUdy_0 * dVdx_0 * dy - 4 * d * dUdx_0 * dVdx_0 * dVdx_1 * dx - 3 * d * dUdx_0 * dVdx_0 * dVdy_0 * dx + 3 * d * dUdx_0 * dUdx_1 * dVdx_1 * dy + 4 * d * dUdx_0 * dUdx_1 * dVdy_0 * dy - 2 * d * dUdx_0 * dUdy_0 * dVdx_1 * dy + 9 * d * dUdx_0 * dVdx_1 * dVdy_0 * dx - 2 * d * dUdx_1 * dUdy_0 * dVdx_0 * dy - 4 * d * dUdx_1 * dVdx_0 * dVdx_1 * dx - 9 * d * dUdx_1 * dVdx_0 * dVdy_0 * dx - 6 * d * dUdy_0 * dVdx_0 * dVdy_0 * dx - 4 * d * dUdx_1 * dUdy_0 * dVdx_1 * dy + 3 * d * dUdx_1 * dVdx_1 * dVdy_0 * dx - 6 * d * dUdy_0 * dVdx_1 * dVdy_0 * dx - 3 * d * dUdx_0 * dVdx_0 * dx * dy - 3 * d * dUdx_0 * dVdx_1 * dx * dy + 3 * d * dUdx_1 * dVdx_0 * dx * dy + 6 * d * dUdy_0 * dVdx_0 * dx * dy + 3 * d * dUdx_1 * dVdx_1 * dx * dy + 6 * d * dUdy_0 * dVdx_1 * dx * dy) / (
        #                    12 * dx * dy))
        # calculate the boundary loss
        disp_x, disp_y = y_out[..., 0:1], y_out[..., 1:2]
        disp_y_right = disp_y[:, :, -1, :]
        ty = x_real[:, :, 0, 0:1]
        loss_bnd = jnp.sum(disp_y_right * ty) * (self.W / self.num_y)
        return jnp.where(loss_int - loss_bnd > 1e6, 1e6, loss_int - loss_bnd)


class DemPlateHoleLoss(PinoFGMLoss):
    def __init__(self, model, model_data, normalizers, d=2, p=2, size_average=True, reduction=True):
        super().__init__(model, model_data, normalizers, d, p, size_average, reduction)
        self.Wint = None
        self.Emod = model_data['beam']['E']
        self.get_int_weights()

    def get_int_weights(self, n=1):
        self.Wint = jnp.ones((n, self.num_y - 2, self.num_x - 2, 1), dtype=self.data_type)
        top_bottom_edge = 1 / 2 * jnp.ones((n, 1, self.num_x - 2, 1), dtype=self.data_type)
        self.Wint = jnp.concatenate((top_bottom_edge, self.Wint, top_bottom_edge), axis=1)
        left_right_edge = jnp.concatenate((jnp.array([[[[1 / 4]]]], dtype=self.data_type),
                                           1 / 2 * jnp.ones((1, self.num_y - 2, 1, 1), dtype=self.data_type),
                                           jnp.array([[[[1 / 4]]]], dtype=self.data_type)), axis=1)
        left_right_edge = jnp.repeat(left_right_edge, n, axis=0)
        self.Wint = jnp.concatenate((left_right_edge, self.Wint, left_right_edge), axis=2)

    def constitutiveEq(self, x, y_out):
        e_mat = self.Emod * self.states[self.state]
        elasticity_modulus = 1 - x
        elasticity_modulus = jnp.expand_dims(elasticity_modulus, axis=-1)
        e_mat = jnp.expand_dims(jnp.expand_dims(e_mat, axis=0), axis=0)
        e_porous = jnp.multiply(elasticity_modulus, e_mat)

        eps_xx_val, eps_yy_val, eps_xy_val = self.kinematicEq(y_out)
        eps_val = jnp.concatenate([eps_xx_val, eps_yy_val, eps_xy_val], axis=-1)
        eps_val = jnp.expand_dims(eps_val, axis=-1)

        stress_val = jnp.matmul(e_porous, eps_val).squeeze(-1)

        stress_xx_val = stress_val[..., 0:1]
        stress_yy_val = stress_val[..., 1:2]
        stress_xy_val = stress_val[..., 2:3]
        return stress_xx_val, stress_yy_val, stress_xy_val

    def loss_fn(self, params, x, y):

        if self.model_data["normalized"]:
            y_out = self.model.apply(params, x)
            y_out = self.normalizer_y.decode(y_out)
            x_real = self.normalizer_x.decode(x)
            # y_real = self.normalizer_y.decode(y)
        else:
            y_out = self.model.apply(params, x)
            x_real = x
            # y_real = y

        # mask_temp = jnp.where(x_real < 0.5, False, True)
        # mask = jnp.where(jnp.array(x_real) < 0.5, False, True)
        # mask = jnp.asarray(jnp.where(x_real < 0.5, False, True))
        # mask = jnp.asarray(jnp.where(jnp.array(x_real) < 0.5, False, True))

        # mask = jnp.nonzero(mask_temp)
        # loss_data = self.rel(y_out, y_real)
        # calculate the interior loss
        stress_xx_val, stress_yy_val, stress_xy_val = self.constitutiveEq(x_real, y_out)
        eps_xx_val, eps_yy_val, eps_xy_val = self.kinematicEq(y_out)

        # stress_xx_val = stress_xx_val[mask]
        # stress_yy_val = stress_yy_val[mask]
        # stress_xy_val = stress_xy_val[mask]
        # eps_xx_val = eps_xx_val[mask]
        # eps_yy_val = eps_yy_val[mask]
        # eps_xy_val = eps_xy_val[mask]

        # if self.Wint.shape != x_real.shape:
        #     self.get_int_weights(x_real.shape[0])
        #     self.Wint = self.Wint[mask]

        loss_int = jnp.sum(1 / 2 * (eps_xx_val * stress_xx_val + eps_yy_val * stress_yy_val + eps_xy_val
                                    * stress_xy_val) * self.Wint) * (self.L / self.num_x * self.W / self.num_y)

        # dx = self.L / (self.num_x - 1)
        dy = self.W / (self.num_y - 1)

        # calculate the boundary loss
        disp_x, disp_y = y_out[..., 0:1], y_out[..., 1:2]
        disp_x_right = disp_x[:, :, -1, :]
        tx = jnp.ones_like(disp_x_right)
        tx = tx.at[:, 0, :].set(tx[:, 0, :]/2)
        tx = tx.at[:, -1, :].set(tx[:, -1, :]/2)

        loss_bnd = jnp.sum(disp_x_right * tx) * dy
        # jax.debug.print("loss_int = {}, loss_bnd = {}, diff = {}", loss_int, loss_bnd, loss_int - loss_bnd)
        return loss_int - loss_bnd


class VinoPlateHoleLoss(DemPlateHoleLoss):

    @staticmethod
    def difference_disp(y_out):
        u = y_out[..., 0:1]
        v = y_out[..., 1:2]
        dudx = difference_x(u)
        dudy = difference_y(u)
        dvdx = difference_x(v)
        dvdy = difference_y(v)
        return dudx, dudy, dvdx, dvdy

    def loss_fn(self, params, x, y):

        if self.model_data["normalized"]:
            y_out = self.model.apply(params, x)
            y_out = self.normalizer_y.decode(y_out)
            x_real = self.normalizer_x.decode(x)
            # y_real = self.normalizer_y.decode(y)
        else:
            y_out = self.model.apply(params, x)
            x_real = x
            # y_real = y

        # loss_data = self.rel(y_out, y_real)
        elasticity_modulus = jnp.where(x_real < 0.5, 1.0, 0.0)
        e_mat = self.Emod * elasticity_modulus

        e_00 = e_mat[:, 0:-1, 0:-1, :]
        e_10 = e_mat[:, 1:, 0:-1, :]
        e_01 = e_mat[:, 0:-1, 1:, :]
        e_11 = e_mat[:, 1:, 1:, :]

        # calculate the interior loss
        dudx, dudy, dvdx, dvdy = self.difference_disp(y_out)

        dudx_0 = dudx[:, 0:-1, 0:-1, :]
        dudx_1 = dudx[:, 1:, 0:-1, :]
        dudy_0 = dudy[:, 0:-1, 0:-1, :]
        dvdx_0 = dvdx[:, 0:-1, 0:-1, :]
        dvdx_1 = dvdx[:, 1:, 0:-1, :]
        dvdy_0 = dvdy[:, 0:-1, 0:-1, :]

        dx = self.L / (self.num_x - 1)
        dy = self.W / (self.num_y - 1)
        n = dy / dx
        nu = self.nu
        batch_size = e_00.shape[0]
        loss_int = jnp.sum(
            -(3 * e_00 * dudx_0 ** 2 + 3 * e_00 * dudx_1 ** 2 + 18 * e_00 * dudy_0 ** 2 + 9 * e_01 * dudx_0 ** 2 +
              9 * e_01 * dudx_1 ** 2 + 18 * e_01 * dudy_0 ** 2 + 3 * e_10 * dudx_0 ** 2 + 3 * e_10 * dudx_1 ** 2 +
              18 * e_10 * dudy_0 ** 2 + 9 * e_11 * dudx_0 ** 2 + 9 * e_11 * dudx_1 ** 2 + 18 * e_11 * dudy_0 ** 2 +
              6 * e_00 * dvdx_0 ** 2 + 6 * e_00 * dvdx_1 ** 2 + 36 * e_00 * dvdy_0 ** 2 + 18 * e_01 * dvdx_0 ** 2 +
              18 * e_01 * dvdx_1 ** 2 + 36 * e_01 * dvdy_0 ** 2 + 6 * e_10 * dvdx_0 ** 2 + 6 * e_10 * dvdx_1 ** 2 +
              36 * e_10 * dvdy_0 ** 2 + 18 * e_11 * dvdx_0 ** 2 + 18 * e_11 * dvdx_1 ** 2 + 36 * e_11 * dvdy_0 ** 2 +
              18 * e_00 * dudx_0 ** 2 * n ** 2 + 6 * e_00 * dudx_1 ** 2 * n ** 2 + 18 * e_01 * dudx_0 ** 2 * n ** 2 +
              6 * e_01 * dudx_1 ** 2 * n ** 2 + 6 * e_10 * dudx_0 ** 2 * n ** 2 + 18 * e_10 * dudx_1 ** 2 * n ** 2 +
              6 * e_11 * dudx_0 ** 2 * n ** 2 + 18 * e_11 * dudx_1 ** 2 * n ** 2 + 9 * e_00 * dvdx_0 ** 2 * n ** 2 +
              3 * e_00 * dvdx_1 ** 2 * n ** 2 + 9 * e_01 * dvdx_0 ** 2 * n ** 2 + 3 * e_01 * dvdx_1 ** 2 * n ** 2 +
              3 * e_10 * dvdx_0 ** 2 * n ** 2 + 9 * e_10 * dvdx_1 ** 2 * n ** 2 + 3 * e_11 * dvdx_0 ** 2 * n ** 2 +
              9 * e_11 * dvdx_1 ** 2 * n ** 2 - 6 * e_00 * dudx_0 * dudx_1 - 12 * e_00 * dudx_0 * dudy_0 +
              12 * e_00 * dudx_1 * dudy_0 - 18 * e_01 * dudx_0 * dudx_1 - 24 * e_01 * dudx_0 * dudy_0 +
              24 * e_01 * dudx_1 * dudy_0 - 6 * e_10 * dudx_0 * dudx_1 - 12 * e_10 * dudx_0 * dudy_0 +
              12 * e_10 * dudx_1 * dudy_0 - 18 * e_11 * dudx_0 * dudx_1 - 24 * e_11 * dudx_0 * dudy_0 +
              24 * e_11 * dudx_1 * dudy_0 - 12 * e_00 * dvdx_0 * dvdx_1 - 24 * e_00 * dvdx_0 * dvdy_0 +
              24 * e_00 * dvdx_1 * dvdy_0 - 36 * e_01 * dvdx_0 * dvdx_1 - 48 * e_01 * dvdx_0 * dvdy_0 +
              48 * e_01 * dvdx_1 * dvdy_0 - 12 * e_10 * dvdx_0 * dvdx_1 - 24 * e_10 * dvdx_0 * dvdy_0 +
              24 * e_10 * dvdx_1 * dvdy_0 - 36 * e_11 * dvdx_0 * dvdx_1 - 48 * e_11 * dvdx_0 * dvdy_0 +
              48 * e_11 * dvdx_1 * dvdy_0 - 3 * e_00 * dudx_0 ** 2 * nu - 3 * e_00 * dudx_1 ** 2 * nu -
              18 * e_00 * dudy_0 ** 2 * nu - 9 * e_01 * dudx_0 ** 2 * nu - 9 * e_01 * dudx_1 ** 2 * nu -
              18 * e_01 * dudy_0 ** 2 * nu - 3 * e_10 * dudx_0 ** 2 * nu - 3 * e_10 * dudx_1 ** 2 * nu -
              18 * e_10 * dudy_0 ** 2 * nu - 9 * e_11 * dudx_0 ** 2 * nu - 9 * e_11 * dudx_1 ** 2 * nu -
              18 * e_11 * dudy_0 ** 2 * nu + 12 * e_00 * dudx_0 * dudx_1 * n ** 2 +
              12 * e_01 * dudx_0 * dudx_1 * n ** 2 + 12 * e_10 * dudx_0 * dudx_1 * n ** 2 +
              12 * e_11 * dudx_0 * dudx_1 * n ** 2 + 6 * e_00 * dvdx_0 * dvdx_1 * n ** 2 +
              6 * e_01 * dvdx_0 * dvdx_1 * n ** 2 + 6 * e_10 * dvdx_0 * dvdx_1 * n ** 2 +
              6 * e_11 * dvdx_0 * dvdx_1 * n ** 2 - 9 * e_00 * dvdx_0 ** 2 * n ** 2 * nu -
              3 * e_00 * dvdx_1 ** 2 * n ** 2 * nu - 9 * e_01 * dvdx_0 ** 2 * n ** 2 * nu -
              3 * e_01 * dvdx_1 ** 2 * n ** 2 * nu - 3 * e_10 * dvdx_0 ** 2 * n ** 2 * nu -
              9 * e_10 * dvdx_1 ** 2 * n ** 2 * nu - 3 * e_11 * dvdx_0 ** 2 * n ** 2 * nu -
              9 * e_11 * dvdx_1 ** 2 * n ** 2 * nu - 8 * e_00 * dudx_0 * dvdx_0 * n - 4 * e_00 * dudx_0 * dvdx_1 * n +
              8 * e_00 * dudx_1 * dvdx_0 * n + 24 * e_00 * dudy_0 * dvdx_0 * n - 16 * e_01 * dudx_0 * dvdx_0 * n +
              4 * e_00 * dudx_1 * dvdx_1 * n + 12 * e_00 * dudy_0 * dvdx_1 * n - 8 * e_01 * dudx_0 * dvdx_1 * n +
              16 * e_01 * dudx_1 * dvdx_0 * n + 24 * e_01 * dudy_0 * dvdx_0 * n + 8 * e_01 * dudx_1 * dvdx_1 * n +
              12 * e_01 * dudy_0 * dvdx_1 * n - 4 * e_10 * dudx_0 * dvdx_0 * n - 8 * e_10 * dudx_0 * dvdx_1 * n +
              4 * e_10 * dudx_1 * dvdx_0 * n + 12 * e_10 * dudy_0 * dvdx_0 * n - 8 * e_11 * dudx_0 * dvdx_0 * n +
              8 * e_10 * dudx_1 * dvdx_1 * n + 24 * e_10 * dudy_0 * dvdx_1 * n - 16 * e_11 * dudx_0 * dvdx_1 * n +
              8 * e_11 * dudx_1 * dvdx_0 * n + 12 * e_11 * dudy_0 * dvdx_0 * n + 16 * e_11 * dudx_1 * dvdx_1 * n +
              24 * e_11 * dudy_0 * dvdx_1 * n + 6 * e_00 * dudx_0 * dudx_1 * nu + 12 * e_00 * dudx_0 * dudy_0 * nu -
              12 * e_00 * dudx_1 * dudy_0 * nu + 18 * e_01 * dudx_0 * dudx_1 * nu + 24 * e_01 * dudx_0 * dudy_0 * nu -
              24 * e_01 * dudx_1 * dudy_0 * nu + 6 * e_10 * dudx_0 * dudx_1 * nu + 12 * e_10 * dudx_0 * dudy_0 * nu -
              12 * e_10 * dudx_1 * dudy_0 * nu + 18 * e_11 * dudx_0 * dudx_1 * nu + 24 * e_11 * dudx_0 * dudy_0 * nu -
              24 * e_11 * dudx_1 * dudy_0 * nu - 8 * e_00 * dudx_0 * dvdx_0 * n * nu +
              20 * e_00 * dudx_0 * dvdx_1 * n * nu + 48 * e_00 * dudx_0 * dvdy_0 * n * nu -
              16 * e_00 * dudx_1 * dvdx_0 * n * nu - 24 * e_00 * dudy_0 * dvdx_0 * n * nu -
              16 * e_01 * dudx_0 * dvdx_0 * n * nu + 4 * e_00 * dudx_1 * dvdx_1 * n * nu +
              24 * e_00 * dudx_1 * dvdy_0 * n * nu - 12 * e_00 * dudy_0 * dvdx_1 * n * nu +
              40 * e_01 * dudx_0 * dvdx_1 * n * nu + 48 * e_01 * dudx_0 * dvdy_0 * n * nu -
              32 * e_01 * dudx_1 * dvdx_0 * n * nu - 24 * e_01 * dudy_0 * dvdx_0 * n * nu +
              8 * e_01 * dudx_1 * dvdx_1 * n * nu + 24 * e_01 * dudx_1 * dvdy_0 * n * nu -
              12 * e_01 * dudy_0 * dvdx_1 * n * nu - 4 * e_10 * dudx_0 * dvdx_0 * n * nu +
              16 * e_10 * dudx_0 * dvdx_1 * n * nu + 24 * e_10 * dudx_0 * dvdy_0 * n * nu -
              20 * e_10 * dudx_1 * dvdx_0 * n * nu - 12 * e_10 * dudy_0 * dvdx_0 * n * nu -
              8 * e_11 * dudx_0 * dvdx_0 * n * nu + 8 * e_10 * dudx_1 * dvdx_1 * n * nu +
              48 * e_10 * dudx_1 * dvdy_0 * n * nu - 24 * e_10 * dudy_0 * dvdx_1 * n * nu +
              32 * e_11 * dudx_0 * dvdx_1 * n * nu + 24 * e_11 * dudx_0 * dvdy_0 * n * nu -
              40 * e_11 * dudx_1 * dvdx_0 * n * nu - 12 * e_11 * dudy_0 * dvdx_0 * n * nu +
              16 * e_11 * dudx_1 * dvdx_1 * n * nu + 48 * e_11 * dudx_1 * dvdy_0 * n * nu -
              24 * e_11 * dudy_0 * dvdx_1 * n * nu - 6 * e_00 * dvdx_0 * dvdx_1 * n ** 2 * nu -
              6 * e_01 * dvdx_0 * dvdx_1 * n ** 2 * nu - 6 * e_10 * dvdx_0 * dvdx_1 * n ** 2 * nu -
              6 * e_11 * dvdx_0 * dvdx_1 * n ** 2 * nu) / (288 * n * (nu ** 2 - 1))
        )

        # calculate the boundary loss
        disp_x, disp_y = y_out[..., 0:1], y_out[..., 1:2]
        disp_x_right = disp_x[:, :, -1, :]
        tx = jnp.ones_like(disp_x_right)
        tx = tx.at[:, 0, :].set(tx[:, 0, :] / 2)
        tx = tx.at[:, -1, :].set(tx[:, -1, :] / 2)

        loss_bnd = jnp.sum(disp_x_right * tx) * dy
        # jax.debug.print("loss_int = {}, loss_bnd = {}, diff = {}", loss_int, loss_bnd, loss_int - loss_bnd)
        # jax.debug.print("loss_physics = {}", loss_int - loss_bnd)
        return (loss_int - loss_bnd) / batch_size
        # return loss_data
