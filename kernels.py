import numpy as np 

import jax
import jax.numpy as jnp

def K_X_Y_diagonal(X, Y, sigma_sqrd):
    """For two spatial patterns X and Y, the kernel k(x_i,y_i) is equal to sum_i sigma_i^2 x_i y_i"""
    return jnp.dot(X.T * sigma_sqrd, Y) 

def K_X_Y_identity(X, Y):
    return jnp.dot(X.T, Y) 

def K_X_Y_squared_exponential(X, Y, l=1.0, sigma_f=1.0):
    sq_dist = jnp.sum((X.T[:, jnp.newaxis, :] - Y.T[jnp.newaxis, :, :])**2, axis=2)
    return sigma_f**2 * jnp.exp(-0.5 / l * sq_dist) #**2

def K_X_Y_polynomial(X,Y):
    return (1 + jnp.dot(X.T, Y) )**2