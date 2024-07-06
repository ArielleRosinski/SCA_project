

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

def mse_loss(params, X, Y):
    preds = batched_predict(params, X)
    return jnp.mean((preds - Y) ** 2)

def update(params, X, Y, optimizer, opt_state,):
    grads = grad(mse_loss)(params, X, Y)
    updates, new_opt_state = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_opt_state


def forward(X, Y, layer_sizes = [10, 10, 5, 2], num_epochs = 1000, seed=42, learning_rate = 1e-3):
    key = random.PRNGKey(seed)
    keys = random.split(key, num=num_epochs)
    params = init_network_params(layer_sizes, key)

    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)

    for epoch in range(num_epochs):
        params, opt_state = update(params, X, Y, opt_state)
        if epoch % 10 == 0:  
            current_loss = mse_loss(params, X, Y)  
            print(f"Epoch {epoch+1}, Loss: {current_loss:.4f}")