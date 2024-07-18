

import math

import jax
import jax.numpy as jnp
from jax import grad, random, vmap
import optax


def random_layer_params(m, n, key, scale=1e-2):
    w_key, b_key = random.split(key)
    return scale * random.normal(w_key, (n, m)), scale * random.normal(b_key, (n,))

def init_network_params(sizes, key):
    keys = random.split(key, len(sizes))
    return [random_layer_params(m, n, k) for m, n, k in zip(sizes[:-1], sizes[1:], keys)]

def relu(x):
    return jnp.maximum(0, x)

def predict(params, X):
    for w, b in params[:-1]:
        outputs = jnp.dot(X, w.T) + b
        outputs = relu(outputs)
    final_w, final_b = params[-1]
    final_outputs = jnp.dot(outputs, final_w.T) + final_b
    return final_outputs

batched_predict = vmap(predict, in_axes=(None, 0))

def mse_loss(params, X, Y, l2_reg):
    preds = batched_predict(params, X)
    mse = jnp.mean((preds - Y) ** 2)

    l2_norm = 0.0
    for param in jax.tree_leaves(params):
        l2_norm += jnp.sum(param ** 2)

    loss = mse + l2_reg * l2_norm
    return loss

def update(params, X, Y, l2_reg, optimizer, opt_state):
    grads = grad(mse_loss)(params, X, Y, l2_reg)
    updates, new_opt_state = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_opt_state


def optimize(X, Y, l2_reg, layer_sizes = [10, 10, 5, 2], num_iterations = 1000, seed=42, learning_rate = 1e-3):
    key = random.PRNGKey(seed)
    params = init_network_params(layer_sizes, key)

    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)

    ls_loss = [] 

    for i in range(num_iterations):
        params, opt_state = update(params, X, Y,l2_reg, optimizer, opt_state)
        loss_ = mse_loss(params, X, Y, l2_reg)  
        ls_loss.append(loss_)
        if i % 10 == 0:  
            print(f"Iter {i+1}, Loss: {loss_:.4f}")
    return params, ls_loss