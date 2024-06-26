import numpy as np 

import optax

import math

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from kernel_sca import *
from utils import *
from kernels import * 

import wandb

from absl import app
from absl import flags
import sys

import os

os.environ['JAX_PLATFORMS'] = 'cpu'
try:
    import jax
    jax.config.update("jax_enable_x64", True)
    from jax.lib import xla_bridge
    if 'gpu' in jax.lib.xla_bridge.get_backend().platform:
        os.environ['JAX_PLATFORMS'] = 'gpu'
except Exception as e:
    print(f"Switching to GPU failed with error: {str(e)}. Continuing with CPU.")

import jax.numpy as jnp
from jax import grad, random, vmap


flags.DEFINE_integer('seed', 42, 'Random seed to set')
flags.DEFINE_integer('iterations', 500, 'training iterations')
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('d', 3, 'Subspace dimensionality')
flags.DEFINE_string('path', '/rds/user/ar2217/hpc-work/SCA/datasets/Churchland/churchland.npy',
                     'dataset path')
flags.DEFINE_string('save_path', '/rds/user/ar2217/hpc-work/SCA/outputs/Kernel',
                     'save path')
flags.DEFINE_string('name', 'a_linear_kernel',
                     'name of the run and of the saved file')
flags.DEFINE_boolean('save', True,
                     'Whether to save the learned parameters')
flags.DEFINE_string('mode', 'disabled',
                     'wanb mode')
flags.DEFINE_string('kernel', 'gaussian',
                     'type of kernel used')
flags.DEFINE_float('l', 1, 'lengthscale RGB kernel')

FLAGS = flags.FLAGS
FLAGS(sys.argv)

seed = FLAGS.seed
iterations = FLAGS.iterations
learning_rate = FLAGS.learning_rate
path = FLAGS.path 
save_path = FLAGS.save_path 
d = FLAGS.d
name = FLAGS.name
save = FLAGS.save
mode = FLAGS.mode
kernel = FLAGS.kernel
l = FLAGS.l

print(f'Using d = {d}, l = {l}')

X = jnp.array(np.load(path)) ##X, _ = pre_processing(X_init, center=False, soft_normalize='max')
K, N, T = X.shape
A = jnp.swapaxes(X, 0, 1)                  #(N, K, T)
A = A.reshape(N,-1)  

if kernel == 'gaussian':
    K_A_X = np.zeros((K*T, K, T))
    for k in range(K):
        K_A_X[:,k,:] = K_X_Y_squared_exponential(A, X[k], l = l)
    K_A_X = jnp.array(K_A_X)
    K_A_A = jnp.array(K_X_Y_squared_exponential(A, A, l = l))
elif kernel == 'linear':
    K_A_X = K_X_Y_identity(A, X)                                    #(K*T, K, T)
    K_A_A = K_X_Y_identity(A, A)

K_A_A_reshaped = K_A_A.reshape(K,T,K,T)                          #(K,T,K,T)
means = jnp.mean(K_A_A_reshaped, axis=(0, 2), keepdims=True)     #(1, T, 1, T)
K_A_A_tilde = (K_A_A_reshaped - means).reshape(K*T,K*T)          #(K*T,K*T)
P, S, Pt = jnp.linalg.svd(K_A_A_tilde, full_matrices=False)      #P is (K*T, K*T) and S is (K*T,)

wandb.init(project="SCA-project-kernel", name=name, mode=mode)
optimized_alpha_tilde, _,  _ = optimize(P, S, K_A_X, X, iterations= iterations, learning_rate= learning_rate, seed = seed )
wandb.finish()

if save: 
    np.save(f'{save_path}/alpha_tilde_{d}d_l{l}', optimized_alpha_tilde)

    alpha_tilde_QR, _ = jnp.linalg.qr(optimized_alpha_tilde) 
    alpha = (P / jnp.sqrt(S)) @ alpha_tilde_QR

    alpha_reshaped = alpha.reshape(K,T,d)                                           #(K, T, d)
    mean = jnp.mean(alpha_reshaped, axis=(0), keepdims=True)                        #(1, T, d)
    optimized_alpha_H = (alpha_reshaped - mean).reshape(K*T,d)                      #(K*T,d)
    projection = jnp.einsum('ij,imk->mjk', optimized_alpha_H, K_A_X)                #(K*T,d) @ (K*T, K, T) --> (K, d, T)

    plot_3D(projection)
    plt.savefig(f'{save_path}/projection_{d}d_l{l}_fig.png')

    np.save(f'{save_path}/projection_{d}d_l{l}', projection)