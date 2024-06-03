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
flags.DEFINE_string('dataset_path', "/rds/user/ar2217/hpc-work/SCA/datasets/MC_Maze/psth.npy",
                     'dataset path')
flags.DEFINE_string('save_path', "/rds/user/ar2217/hpc-work/SCA/outputs/MC_Maze",
                     'dataset path')
flags.DEFINE_boolean('pre_processing_', False,
                     'Use pre_processing function versus load pre-processed X')


FLAGS = flags.FLAGS
FLAGS(sys.argv)

dataset_path = FLAGS.dataset_path 
save_path = FLAGS.save_path 
d = FLAGS.d
seed = FLAGS.seed
iterations = FLAGS.iterations
pre_processing_ = FLAGS.pre_processing_

X_raw = np.load(dataset_path).swapaxes(1,2)
K, N, T = X_raw.shape

if pre_processing_:
    X, _ = pre_processing(X_raw, soft_normalize='max')
    X_pre_pca, _ = pre_processing(X_raw, soft_normalize='max', pca=False)
else: 
    X = np.load("/rds/user/ar2217/hpc-work/SCA/datasets/MC_Maze/X_softNormMax.npy")
    X_pre_pca = np.load("/rds/user/ar2217/hpc-work/SCA/datasets/MC_Maze/X_softNormMax_pcaFalse.npy")

wandb.init(project="SCA-project-MC_Maze", name=f"d={d}.5", mode="disabled")
U, ls_loss, ls_S_ratio = optimize(X,d=d, seed=seed, iterations=100) 
np.save(f'{save_path}/U_psth_{d}d', U)
wandb.finish()

U_qr, _ = jnp.linalg.qr(U)        
Y = jnp.einsum('ji,kjl->kil', U_qr, X)
plot_3D(Y[:,:,:])
plt.title(f"sca {var_explained(X, U_qr):.2f}")
# plt.savefig(f'{save_path}/proj_{d}d')
# np.save(f'{save_path}/Y_{d}d', Y)
np.save(f'{save_path}/X_{d}d_2', X)
np.save(f'{save_path}/Xraw_{d}d', X_raw)

X_reshaped = np.concatenate(X_pre_pca.swapaxes(1,2))
pca = PCA(d)
X_pca = pca.fit_transform(X_reshaped)
PCs = pca.components_
X_pca = X_pca.reshape(K, T, d).swapaxes(1,2)
pca_variance_captured = pca.explained_variance_ratio_

# plt.figure()
# plot_3D(X_pca[:,:,:])
# plt.savefig(f'{save_path}/proj_pca_{d}d')

# np.save(f'{save_path}/pca_variance_captured_{d}d', pca_variance_captured)
# np.save(f'{save_path}/PCs_{d}d', PCs)
# np.save(f'{save_path}/X_pca_{d}d', X_pca)