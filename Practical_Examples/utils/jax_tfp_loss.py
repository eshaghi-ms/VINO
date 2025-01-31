# import sys
import jax.debug
import jax.flatten_util
# from io import StringIO


def jax_tfp_function_factory(model, params, loss_fn, *args):
    """A factory to create a function required by tensorflow_probability.substrates.jax.optimizer.lbfgs_minimize.
    Based on the example from https://pychao.com/2019/11/02/optimize-tensorflow-keras-models-with-l-bfgs-from-tensorflow-probability/
    Args:
        model: an instance
        params: network parameters
        loss_fn: loss functions
        *args: arguments to be passed to model.get_grads method

    Returns:
        A function that has a signature of:
            loss_value, gradients = f(model_parameters).

    """
    # params = model.get_params(model.opt_state)
    init_params_1d, unflatten_params = jax.flatten_util.ravel_pytree(params)

    # now create a function that will be returned by this factory
    def f(params_1d):
        """A function that can be used by tfp.substrates.jax.optimizer.lbfgs_minimize

        This function is created by function_factory.

        Args:
           params_1d [in]: a 1D (device) array.

        Returns:
            A scalar loss and the gradients w.r.t. the `params_1d`.
        """
        # params = unflatten_params(params_1d)
        loss_value, grads = jax.value_and_grad(loss_fn)(params, *args)
        grads, _ = jax.flatten_util.ravel_pytree(grads)
        # jax.debug.print("Train loss: {} \r", loss_value)
        # print out iteration & loss
        f.iter += 1
        if f.iter % 1 == 0:
            jax.debug.print("Iteration: {}, Loss: {}", f.iter, loss_value)

        # store loss value, so we can retrieve later
        # output = StringIO()
        # sys.stdout = output  # Redirect stdout
        # jax.debug.print("Iteration: {}, Loss: {}", f.iter, loss_value)
        # f.history.append(output.getvalue())
        # sys.stdout = sys.__stdout__

        return loss_value, grads

    # store these information as members so we can use them outside the scope
    f.iter = 0
    f.history = []
    f.init_params_1d = init_params_1d

    return f
