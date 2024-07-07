import numpy as np 

import optax

import math

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from utils import *

import wandb

import pickle

from absl import app
from absl import flags
import sys

import os
import time

import jax
import jax.numpy as jnp
from jax import grad, random, vmap
import optax

from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score

from NN_jax import *


flags.DEFINE_string('path_X', '',
                     'kSCA/SCA/PCA-projected data')
flags.DEFINE_string('path_Y', '/rds/user/ar2217/hpc-work/SCA/datasets/MC_Maze_20ms/behaviour/aug_behaviour.npy',
                     'behaviour')
flags.DEFINE_string('save_path', '/rds/user/ar2217/hpc-work/SCA/outputs/motor_cortex/R2_nonlinear/aug_behaviour',
                     'save path')
flags.DEFINE_string('name', '',
                     'file name')
flags.DEFINE_integer('d', 3, 'subspace dimensionality')
flags.DEFINE_integer('split', 10, 'test/train split')
flags.DEFINE_integer('lag', 5, 'behaviour/neural prediction time lag')
flags.DEFINE_integer('iterations', 15000, 'training iterations')

FLAGS = flags.FLAGS
FLAGS(sys.argv)

path_X = FLAGS.path_X
path_Y = FLAGS.path_Y
save_path = FLAGS.save_path
name = FLAGS.name
d = FLAGS.d
iterations = FLAGS.iterations
split = FLAGS.split
lag = FLAGS.lag


# try:
#     os.makedirs(save_path)
# except FileExistsError:
#     print("Directory already exists")

behaviour = np.load(path_Y)
y_train = behaviour[split:,:,lag:].swapaxes(1,2).reshape(-1, behaviour.shape[1])
y_test = behaviour[:split,:,lag:].swapaxes(1,2).reshape(-1, behaviour.shape[1])

X = np.load(path_X)
X_train = X[split:,:,:-lag].swapaxes(1,2).reshape(-1, X.shape[1])
X_test = X[:split,:,:-lag].swapaxes(1,2).reshape(-1, X.shape[1])

params, ls_loss = optimize(X_train, y_train, layer_sizes = [d, 10, 2], num_iterations = iterations, seed=42, learning_rate = 1e-2)

predictions = predict(params, X_test)
r2 = r2_score(y_test, predictions)

np.save(f'{save_path}/{name}_{d}_ls_loss', np.array(ls_loss))

plt.figure()
plt.plot(ls_loss)
plt.savefig(f'{save_path}/{name}_{d}_loss_fig.png')

#np.save(f'{save_path}/{name}_{d}_params', np.array(params))
with open(f"{save_path}/{name}_{d}_params.pkl", "wb") as file:
    pickle.dump(params, file)

np.save(f'{save_path}/{name}_{d}_predictions', np.array(predictions))

np.save(f'{save_path}/{name}_{d}_r2 ', np.array(r2))