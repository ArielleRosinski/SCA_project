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
flags.DEFINE_string('dataset_path', "/rds/user/ar2217/hpc-work/SCA/datasets/MC_Maze/X_softNormMax.npy",
                     'dataset path')
flags.DEFINE_string('dataset_path_pcaFalse', "/rds/user/ar2217/hpc-work/SCA/datasets/MC_Maze/X_softNormMax_pcaFalse.npy",
                     'dataset path with no pre-computed pca')
flags.DEFINE_string('save_path', "/rds/user/ar2217/hpc-work/SCA/outputs/MC_Maze",
                     'dataset path')
flags.DEFINE_boolean('pre_processing_', False,
                     'Use pre_processing function versus load pre-processed X')
flags.DEFINE_boolean('pca', True,
                     'conduct and save pca')
flags.DEFINE_string('mode', "disabled",
                     'wandb mode')
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
flags.DEFINE_float('noise', 0.0, 'Noise added to psth.')
flags.DEFINE_integer('split', 10, 'Train/test split')

FLAGS = flags.FLAGS
FLAGS(sys.argv)

dataset_path = FLAGS.dataset_path 
dataset_path_pcaFalse = FLAGS.dataset_path_pcaFalse
save_path = FLAGS.save_path 
d = FLAGS.d
seed = FLAGS.seed
iterations = FLAGS.iterations
pre_processing_ = FLAGS.pre_processing_
mode = FLAGS.mode
learning_rate = FLAGS.learning_rate
pca = FLAGS.pca
noise = FLAGS.noise
split = FLAGS.split

print(f'noise: {noise}')
print(f'pre_processing_: {pre_processing_}')

np.random.seed(seed)

if pre_processing_:
    X_raw = np.load(dataset_path).swapaxes(1,2)
    X_raw += noise * np.random.normal(size=X_raw.shape)

    X, _ = pre_processing(X_raw, soft_normalize='max')
    X_train = X[split:]
    X_test = X[:split]  
    
    X_pre_pca, _ = pre_processing(X_raw, soft_normalize='max', pca=False)
    X_pre_pca_train = X_pre_pca[split:]
    X_pre_pca_test = X_pre_pca[:split]
else: 
    X = np.load(dataset_path) 
    X += noise * np.random.normal(size=X.shape) 
    X = jnp.array(X)
    X_train = X[split:]
    X_test = X[:split]

    X_pre_pca = np.load(dataset_path_pcaFalse)
    X_pre_pca += noise * np.random.normal(size=X_pre_pca.shape) 
    X_pre_pca_train = X_pre_pca[split:]
    X_pre_pca_test = X_pre_pca[:split]


K, _, T = X.shape   #X_train

wandb.init(project="SCA-project-MC_Maze", name=f"d={d}_seed{seed}", mode=mode)
U, ls_loss, ls_S_ratio = optimize(X,d=d, learning_rate=learning_rate, seed=seed) #X_train
np.save(f'{save_path}/U_psth_{d}d', U)
wandb.finish()

U_qr, _ = jnp.linalg.qr(U)        
Y = jnp.einsum('ji,kjl->kil', U_qr, X)  #X_test
np.save(f'{save_path}/Y_{d}d', Y)

if pca: 
   #X_reshaped_train = np.concatenate(X_pre_pca.swapaxes(1,2))       #X_pre_pca_train
    #X_reshaped_test = np.concatenate(X_pre_pca.swapaxes(1,2))       #X_pre_pca_test
    X_reshaped = np.concatenate(X_pre_pca.swapaxes(1,2))
    pca = PCA(d)
    X_pca = pca.fit(X_reshaped).transform(X_reshaped)
    PCs = pca.components_
    X_pca = X_pca.reshape(K, T, d).swapaxes(1,2)            #split
    pca_variance_captured = pca.explained_variance_ratio_
    np.save(f'{save_path}/pca_variance_captured_{d}d', pca_variance_captured)
    np.save(f'{save_path}/PCs_{d}d', PCs)
    np.save(f'{save_path}/X_pca_{d}d', X_pca)