import numpy as np 
from numpy.linalg import qr, svd

from scipy.integrate import solve_ivp
from scipy.linalg import hadamard, subspace_angles

import math

import jax
import jax.numpy as jnp
from jax import grad, random, vmap
import optax

from linear_sca import *
from utils import *

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV

import wandb

from absl import app
from absl import flags
import sys

flags.DEFINE_integer('seed', 42, 'Random seed to set')
flags.DEFINE_integer('d', 3, 'Subspace dimensionality')
flags.DEFINE_integer('iterations', 5000, 'training iterations')
flags.DEFINE_string('dataset_path', " ",
                     'dataset train path')
flags.DEFINE_string('dataset_test_path', " ",
                     'dataset test path')
flags.DEFINE_string('save_path', "/rds/user/ar2217/hpc-work/SCA/outputs/MC_Maze",
                     'dataset path')
flags.DEFINE_string('mode', "disabled",
                     'wandb mode')
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')


FLAGS = flags.FLAGS
FLAGS(sys.argv)

dataset_path = FLAGS.dataset_path 
dataset_test_path = FLAGS.dataset_test_path 
save_path = FLAGS.save_path 
d = FLAGS.d
seed = FLAGS.seed
iterations = FLAGS.iterations
mode = FLAGS.mode
learning_rate = FLAGS.learning_rate


X = np.load(dataset_path) 
X = jnp.array(X)

X_test = jnp.array(np.load(dataset_test_path)) 


K, _, T = X.shape  

wandb.init(project=" ", name=f"d={d}_seed{seed}", mode=mode)
U, ls_loss, ls_S_ratio = optimize(X,d=d, learning_rate=learning_rate, seed=seed, iterations=iterations) 
wandb.finish()

U_qr, _ = jnp.linalg.qr(U)        
Y = jnp.einsum('ji,kjl->kil', U_qr, X)  
np.save(f'{save_path}/Y_{d}d', Y)

np.save(f'{save_path}/U_{d}d', U)

np.save(f'{save_path}/ls_loss_{d}d', np.array(ls_loss))
np.save(f'{save_path}/ls_S_ratio_{d}d', np.array(ls_S_ratio))

Y_test = jnp.einsum('ji,kjl->kil', U_qr, X_test)  
np.save(f'{save_path}/Y_test_{d}d', Y_test)