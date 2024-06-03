import numpy as np 

import jax
import jax.numpy as jnp
from jax import grad, random, vmap

import optax

import math

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from kernel_sca import *
from utils import *

import wandb

from absl import app
from absl import flags
import sys

flags.DEFINE_integer('seed', 42, 'Random seed to set')
flags.DEFINE_integer('iterations', 10000, 'training iterations')
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
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

X_init = np.load(path)

X, _ = pre_processing(X_init, center=False, soft_normalize='max')
K, N, T = X.shape
A = jnp.swapaxes(pre_processing(X_init,soft_normalize='max')[0], 0, 1)                  #(N, K, T)
A = A.reshape(N,-1)                        #(N, K*T)

K_A_X = K_X_Y_identity(A, X)                                    #(K*T, K, T)
K_A_A = K_X_Y_identity(A, A)
K_A_A_reshaped = K_A_A.reshape(K,T,K,T)                          #(K,T,K,T)
means = jnp.mean(K_A_A_reshaped, axis=(0, 2), keepdims=True)     #(1, T, 1, T)
K_A_A_tilde = (K_A_A_reshaped - means).reshape(K*T,K*T)          #(K*T,K*T)
P, S, Pt = jnp.linalg.svd(K_A_A_tilde, full_matrices=False)      #P is (K*T, K*T) and S is (K*T,)

# X = jnp.array(np.load('/rds/user/ar2217/hpc-work/SCA/datasets/Churchland/X_centerFalse_softNormMax.npy'))
# A = jnp.array(np.load('/rds/user/ar2217/hpc-work/SCA/datasets/Churchland/A_softNormMax.npy'))
# K, N, T = X.shape
# K_A_X = K_X_Y_identity(A, X)                                    #(K*T, K, T)
# K_A_A = K_X_Y_identity(A, A)
# P = jnp.array(np.load('/rds/user/ar2217/hpc-work/SCA/datasets/Churchland/P_centerFalse_softNormMax.npy'))
# S = jnp.array(np.load('/rds/user/ar2217/hpc-work/SCA/datasets/Churchland/S_centerFalse_softNormMax.npy'))

wandb.init(project="SCA-project-kernel", name=name, mode=mode)
optimized_alpha_tilde, _,  _ = optimize(P, S, K_A_X, X, iterations= iterations, learning_rate= learning_rate, seed = seed )
wandb.finish()

if save: 
    np.save(f'{save_path}/{name}', optimized_alpha_tilde)

    alpha_tilde_QR, _ = jnp.linalg.qr(optimized_alpha_tilde) 
    alpha = (P / jnp.sqrt(S)) @ alpha_tilde_QR

    alpha_reshaped = alpha.reshape(K,T,d)                                           #(K, T, d)
    mean = jnp.mean(alpha_reshaped, axis=(0), keepdims=True)                        #(1, T, d)
    optimized_alpha_H = (alpha_reshaped - mean).reshape(K*T,d)                      #(K*T,d)
    projection = jnp.einsum('ij,imk->mjk', optimized_alpha_H, K_A_X)                #(K*T,d) @ (K*T, K, T) --> (K, d, T)

    plot_3D(projection)
    plt.savefig(f'{save_path}/{name}_fig.png')