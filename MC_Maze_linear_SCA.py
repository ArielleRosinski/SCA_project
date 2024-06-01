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
flags.DEFINE_string('path', "/Users/ariellerosinski/My Drive/Cambridge/Project/datasets/MC_Maze/psth.npy",
                     'dataset path')


FLAGS = flags.FLAGS
FLAGS(sys.argv)

path = FLAGS.path 
d = FLAGS.d
seed = FLAGS.seed


X_raw = np.load(path).swapaxes(1,2)
K, N, T = X_raw.shape

X, _ = pre_processing(X_raw, soft_normalize='max')
X = jnp.array(X)
X_pre_pca, _ = pre_processing(X_raw, soft_normalize='max', pca=False)

wandb.init(project="SCA-project-MC_Maze", name=f"d={d}", mode="disabled")
U, ls_loss, ls_S_ratio = optimize(X,d=d, seed=seed) 
np.save(f'../outputs/MC_Maze/U_psth_{d}d', U)
wandb.finish()

X_reshaped = np.concatenate(X_pre_pca.swapaxes(1,2))
pca = PCA(d)
X_pca = pca.fit_transform(X_reshaped)
PCs = pca.components_
X_pca = X_pca.reshape(K, T, d).swapaxes(1,2)
pca_variance_captured = pca.explained_variance_ratio_

plot_2D(X_pca)
plt.title(f"pca {var_explained(X_pre_pca, PCs.T):.2f}")

np.save(f'../outputs/MC_Maze/pca_variance_captured_{d}d', pca_variance_captured)
np.save(f'../outputs/MC_Maze/PCs_{d}d', PCs)
np.save(f'../outputs/MC_Maze/X_pca_{d}d', X_pca)