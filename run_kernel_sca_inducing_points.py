import numpy as np 

import optax

import math

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from kernel_sca_inducing_points import *
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
flags.DEFINE_integer('iterations', 10000, 'training iterations')
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
flags.DEFINE_integer('d', 3, 'Subspace dimensionality')
flags.DEFINE_string('path_X', '/rds/user/ar2217/hpc-work/SCA/datasets/MC_Maze_20ms/X_softNormMax_centerFalse.npy',
                     'dataset path')
flags.DEFINE_string('path_A', '/rds/user/ar2217/hpc-work/SCA/datasets/MC_Maze_20ms/A_softNormMax_centerFalse.npy',
                     'dataset path')
flags.DEFINE_string('save_path', '/rds/user/ar2217/hpc-work/SCA/outputs/Kernel',
                     'save path')
flags.DEFINE_string('name', '',
                     'name of wanb run')
flags.DEFINE_string('save', 'True',
                     'Whether to save the learned parameters')
flags.DEFINE_string('mode', 'disabled',
                     'wanb mode')
flags.DEFINE_string('kernel', 'gaussian',
                     'type of kernel used')
flags.DEFINE_integer('c', 30, 'number of inducing points')

FLAGS = flags.FLAGS
FLAGS(sys.argv)

seed = FLAGS.seed
iterations = FLAGS.iterations
learning_rate = FLAGS.learning_rate
path_X = FLAGS.path_X 
path_A = FLAGS.path_A 
save_path = FLAGS.save_path 
d = FLAGS.d
name = FLAGS.name
save = FLAGS.save
mode = FLAGS.mode
kernel = FLAGS.kernel
c = FLAGS.c

print(f'Using d = {d}, kernel={kernel}')

X = jnp.array( np.load(path_X) )
K, N, T = X.shape
A = jnp.array( np.load(path_A) )


if kernel == 'gaussian':
    kernel_function = K_X_Y_squared_exponential
elif kernel == 'linear':
    kernel_function = K_X_Y_identity


wandb.init(project="SCA-project-kernel", name=name, mode=mode)
params, ls_loss,  ls_S_ratio = optimize(X, A, iterations=iterations, learning_rate=learning_rate, d=d, c=c, kernel_function=kernel_function, seed=seed )
wandb.finish()

if save == 'True': 
    np.save(f'{save_path}/params_{d}d_{kernel}', params)

    _, u, l2 = get_params(params, kernel_function=kernel_function)
    K_u_u_K_u_A_alpha_H  = get_alpha(params, A, X, kernel_function, d)

    X_reshaped = X.swapaxes(0,1).reshape(N,-1)
    K_u_X = K_X_Y_identity(u, X_reshaped, l2=l2).reshape(-1,K,T).swapaxes(0,1)  
    Y = jnp.einsum('ji,kjm->kim',  K_u_u_K_u_A_alpha_H, K_u_X)
    Y = center(Y)

    plt.figure()
    plot_3D(Y)
    plt.title(f's = {compute_S_all_pairs(Y)}')
    plt.savefig(f'{save_path}/projection_fig_{d}d_{kernel}.png')

    np.save(f'{save_path}/projection_{d}d_{kernel}', Y)

    plt.figure()
    get_loss_fig(ls_loss, ls_S_ratio)
    plt.savefig(f'{save_path}/loss_fig_{d}d_{kernel}.png')

    np.save(f'{save_path}/ls_loss_{d}d_{kernel}', np.array(ls_loss))
    np.save(f'{save_path}/ls_S_ratio_{d}d_{kernel}', np.array(ls_S_ratio))