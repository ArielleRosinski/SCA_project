import numpy as np

import jax
import jax.numpy as jnp


from absl import app
from absl import flags
import sys

import os

from kernel_sca_inducing_points import *
from utils import *


flags.DEFINE_integer('d', 3, 'Subspace dimensionality')
flags.DEFINE_string('kernel', 'gaussian',
                     'type of kernel used')


FLAGS = flags.FLAGS
FLAGS(sys.argv)

d = FLAGS.d
kernel = FLAGS.kernel

if kernel == 'gaussian':
    kernel_function = K_X_Y_squared_exponential
elif kernel == 'linear':
    kernel_function = K_X_Y_identity
elif kernel == 'rational_quadratic':
    kernel_function = K_X_Y_rational_quadratic

A_train = jnp.load('/rds/user/ar2217/hpc-work/SCA/datasets/MC_Maze_20ms/A_softNormMax_centerFalse_pcaFalse.npy')
X_train = jnp.load('/rds/user/ar2217/hpc-work/SCA/datasets/MC_Maze_20ms/X_softNormMax_centerFalse_pcaFalse.npy')


params = jnp.load(f'/rds/user/ar2217/hpc-work/SCA/outputs/MC_Maze/neural_spikes_kernel_pcaFalse/params_{d}d_{kernel}.npy', allow_pickle=True).tolist()

_, _, _, _, alpha  = get_alpha(params, A_train, X_train, kernel_function, d)
_, _, l2, scale = get_params(params, kernel_function=kernel_function)

var = var_explained_kernel(alpha, kernel_function, A_train, X_train, l2, scale, compute_matrix=False)
np.save(f'/rds/user/ar2217/hpc-work/SCA/outputs/MC_Maze/var_explained_spikes/var_{d}d', var)
