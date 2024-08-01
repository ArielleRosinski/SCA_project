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

#A_train = jnp.load('/rds/user/ar2217/hpc-work/SCA/datasets/MC_Maze_20ms/A_softNormMax_centerFalse_spikes.npy')
#X_train = jnp.load('/rds/user/ar2217/hpc-work/SCA/datasets/MC_Maze_20ms/X_softNormMax_centerFalse_spikes.npy')
A_train = jnp.load('/Users/ariellerosinski/My Drive/Cambridge/Project/datasets/MC_Maze/A_softNormMax_centerFalse_spikes.npy')
X_train = jnp.load('/Users/ariellerosinski/My Drive/Cambridge/Project/datasets/MC_Maze/X_softNormMax_centerFalse_spikes.npy')

#params = jnp.load(f'/rds/user/ar2217/hpc-work/SCA/outputs/MC_Maze/neural_spikes/params_{d}d_{kernel}.npy', allow_pickle=True).tolist()
params = jnp.load(f'/Users/ariellerosinski/My Drive/Cambridge/Project/outputs/kernel_SCA/MC_Maze_3_7_2024/kernel_/neural_spikes/params_{d}d_{kernel}.npy', allow_pickle=True).tolist()

_, _, _, _, alpha  = get_alpha(params, A_train, X_train, kernel_function, d)
_, _, l2, scale = get_params(params, kernel_function=kernel_function)

var = var_explained_kernel(alpha, kernel_function, A_train, X_train, l2, scale, compute_matrix=False)
print(var)
#np.save(f'/rds/user/ar2217/hpc-work/SCA/outputs/MC_Maze/var_explained_spikes/var_{d}d_{kernel}', var)
np.save(f'/Users/ariellerosinski/My Drive/Cambridge/Project/outputs/kernel_SCA/MC_Maze_3_7_2024/results/R2/spikes_hand_vel/var_explained_spikes/var_{d}d_{kernel}', var)