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
# flags.DEFINE_string('path_Y', '/rds/user/ar2217/hpc-work/SCA/datasets/MC_Maze_20ms/behaviour/aug_behaviour.npy',
#                      'behaviour')
# flags.DEFINE_string('save_path', '/rds/user/ar2217/hpc-work/SCA/outputs/motor_cortex/R2_nonlinear/aug_behaviour',
#                      'save path')
flags.DEFINE_string('behaviour', 'hand_vel',
                     'behaviour type incl. hand_vel, augmented behaviour, torques')
flags.DEFINE_string('name', '',
                     'file name')
flags.DEFINE_integer('d', 3, 'subspace dimensionality')
flags.DEFINE_integer('split', 20, 'test/train split')
flags.DEFINE_integer('lag', 5, 'behaviour/neural prediction time lag')
flags.DEFINE_integer('iterations', 20000, 'training iterations')
flags.DEFINE_string('spikes', 'True', 'whether to use psths (False) or spikes (True)')
flags.DEFINE_string('fullX', 'False', 'whether to use psths (False) or spikes (True)')

FLAGS = flags.FLAGS
FLAGS(sys.argv)

path_X = FLAGS.path_X
name = FLAGS.name
d = FLAGS.d
iterations = FLAGS.iterations
split = FLAGS.split
lag = FLAGS.lag
behaviour = FLAGS.behaviour
spikes = FLAGS.spikes
fullX = FLAGS.fullX

if spikes == 'False':
    path_Y = f'/rds/user/ar2217/hpc-work/SCA/datasets/MC_Maze_20ms/behaviour/{behaviour}.npy'
    behaviour_ = np.load(path_Y)
    save_path = f'/rds/user/ar2217/hpc-work/SCA/outputs/motor_cortex/R2_nonlinear/{behaviour}'
elif spikes == 'True':
    path_Y = '/rds/user/ar2217/hpc-work/SCA/datasets/MC_Maze_20ms/train_behavior.npy'
    behaviour_ = np.load(path_Y).swapaxes(1,2)
    save_path = f'/rds/user/ar2217/hpc-work/SCA/outputs/motor_cortex/R2_nonlinear/{behaviour}_spikes'
    split = 200


t = 6 if behaviour == 'aug_behaviour' else 2

# try:
#     os.makedirs(save_path)
# except FileExistsError:
#     print("Directory already exists")


y_train = behaviour_[split:,:,lag:].swapaxes(1,2).reshape(-1, behaviour_.shape[1])
y_test = behaviour_[:split,:,lag:].swapaxes(1,2).reshape(-1, behaviour_.shape[1])

if fullX == 'True':
    X = np.load(path_X).swapaxes(1,2)
else:
    X = np.load(path_X)
X_train = X[split:,:,:-lag].swapaxes(1,2).reshape(-1, X.shape[1])
X_test = X[:split,:,:-lag].swapaxes(1,2).reshape(-1, X.shape[1])

ls_l2_reg = [1e-2, 1e-3, 1e-4, 1e-6, 1e-8, 0]
r2 = float('-inf') 
for l2_reg in ls_l2_reg:
    params_temp, ls_loss_temp = optimize(X_train, y_train, l2_reg, layer_sizes = [d, 10, t], num_iterations = iterations, seed=42, learning_rate = 1e-3)

    predictions_temp = predict(params_temp, X_test)
    r2_temp = r2_score(y_test, predictions_temp)
    if r2_temp > r2:
        r2 = r2_temp
        params, ls_loss, predictions = params_temp, ls_loss_temp, predictions_temp
        l2_reg_final = l2_reg
        print(f'r2_temp: {r2_temp}')
        print(f'l2_reg updated to {l2_reg_final}')

print(f'l2_reg final is {l2_reg_final}')
np.save(f'{save_path}/{name}_{d}_ls_loss', np.array(ls_loss))

plt.figure()
plt.plot(ls_loss)
plt.savefig(f'{save_path}/{name}_{d}_loss_fig.png')

#np.save(f'{save_path}/{name}_{d}_params', np.array(params))
with open(f"{save_path}/{name}_{d}_params.pkl", "wb") as file:
    pickle.dump(params, file)

np.save(f'{save_path}/{name}_{d}_predictions', np.array(predictions))

np.save(f'{save_path}/{name}_{d}_r2 ', np.array(r2))