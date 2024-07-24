import numpy as np

import jax
import jax.numpy as jnp
from jax import random

import matplotlib as mpl
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap, BoundaryNorm


from absl import app
from absl import flags
import sys

import os
import time

from kernels import *
from utils import *

flags.DEFINE_float('sigma', 0.35, 'DDM param')
flags.DEFINE_integer('d', 3, 'subspace dimensionality')
flags.DEFINE_integer('c', 40, 'inducing points')
flags.DEFINE_integer('iterations', 300, 'iters')
flags.DEFINE_float('learning_rate', 1e-2, 'Initial learning rate.')
flags.DEFINE_string('save_path', '/rds/user/ar2217/hpc-work/SCA/outputs/DDM', 'save path')
flags.DEFINE_integer('proj_dims', 10, 'proj dims from DDM to neurons')
flags.DEFINE_float('sigma_noise', 0.5, 'low rank noise param')

FLAGS = flags.FLAGS
FLAGS(sys.argv)

sigma = FLAGS.sigma
d = FLAGS.d
c = FLAGS.c
iterations = FLAGS.iterations
learning_rate = FLAGS.learning_rate
save_path = FLAGS.save_path
proj_dims = FLAGS.proj_dims
sigma_noise = FLAGS.sigma_noise

def DDM(mu, sigma, dt, total_time, key):
    num_trajectories = len(mu)
    num_steps = int(total_time / dt)
    x = jnp.zeros((num_trajectories, num_steps + 1))
    ls_RT = []
    ls_acc = []

    keys = random.split(key, num_trajectories)

    for i in range(num_trajectories):
        current_key = keys[i]
        for t in range(num_steps): 
            if jnp.abs(x[i, t]) < 1:
                normal_sample = random.normal(current_key, shape=())
                x = x.at[i, t + 1].set(x[i, t] + mu[i] * dt + sigma * jnp.sqrt(dt) * normal_sample)
                current_key, subkey = random.split(current_key)
            else:
                ls_RT.append(t)
                ls_acc.append(1 if jnp.sign(x[i, t]) == jnp.sign(mu[i]) else -1)
                
                x = x.at[i, t:].set(jnp.sign(x[i, t]))
                break
    return x, ls_RT, ls_acc

mu = np.array([-0.64, -0.32, -0.16, -0.08, -0.04, 0.0, 0.04, 0.08, 0.16, 0.32, 0.64])  
dt = 0.1
total_time = 100
trials = 150

ls_RTs = []
ls_accs = []
paths = jnp.zeros((trials, len(mu), int(total_time / dt) + 1))
for i in range(trials):
    key = random.PRNGKey(42 + i)
    path, ls_RT, ls_acc = DDM(mu, sigma, dt, total_time, key) 
    paths = paths.at[i].set(path)
    ls_RTs.append(ls_RT)
    ls_accs.append(ls_acc)

accs = jnp.array(ls_accs)
RTs = jnp.array(ls_RTs)
paths = paths[:,:, :jnp.max(RTs)]



def project(paths, proj_dims = 10):
    proj_matrix = random.normal(key, (proj_dims, 1))
    proj_matrix , _ = jnp.linalg.qr(proj_matrix)                                        #(N',N)
    return jnp.einsum('dn,lknt->lkdt', proj_matrix, paths[:,:,jnp.newaxis,:])           #(trial, K, N=1, T)

def relu(x):
    return jnp.maximum(0, x)

def add_low_rank_noise(X, key1, key2, proj_dims = 3, sigma_noise= 1 ):
    trials, K, N, T = X.shape    
    B = random.normal(key1, (N, proj_dims))
    B, _ = jnp.linalg.qr(B)

    epsilon_t = random.normal(key2, (trials, K, T, proj_dims)) * sigma_noise
    noise = jnp.einsum('lktd,nd->lknt', epsilon_t, B)             
    
    X += noise                                                   
    return X


neural_traces = relu(project(paths, proj_dims=proj_dims))
neural_traces = neural_traces * 10

key = random.PRNGKey(42)
key, subkey = random.split(key)
neural_traces = add_low_rank_noise(neural_traces, key, subkey, sigma_noise = sigma_noise)        #(trials, K, N, T)

X = jnp.mean( neural_traces, axis=0 )
K, N, T = X.shape
A = jnp.swapaxes(X, 0, 1)               
A = A.reshape(N,-1)      

### kSCA ###
kernel_function = K_X_Y_squared_exponential

from kernel_sca_inducing_points import *
wandb.init(project="", name="", mode="disabled")
params, ls_loss, ls_S_ratio = optimize(X, A, iterations=iterations, learning_rate=learning_rate, d=d, c=c, kernel_function=kernel_function)
wandb.finish()

_, u, l2, scale = get_params(params, kernel_function=kernel_function)
K_u_u_K_u_A_alpha_H, K_A_u, K_u_u, _  = get_alpha(params, A, X, kernel_function, d)

X_reshaped = X.swapaxes(0,1).reshape(N,-1)
K_u_X = kernel_function(u, X_reshaped, l2=l2, scale=scale).reshape(-1,K,T).swapaxes(0,1)  
Y = jnp.einsum('ji,kjm->kim',  K_u_u_K_u_A_alpha_H, K_u_X)
Y = center(Y)

np.save(f'{save_path}/kSCA/params_kSCA_{d}d_sigma{sigma_noise}_proj_dims{proj_dims}', params)
np.save(f'{save_path}/kSCA/Y_kSCA_{d}d_sigma{sigma_noise}_proj_dims{proj_dims}', Y)

for i in range(Y.shape[0]):
    Y = Y[:,:,:int(jnp.mean(RTs, axis=0).squeeze()[i])]

np.save(f'{save_path}/kSCA/Y_kSCA_cut_{d}d_sigma{sigma_noise}_proj_dims{proj_dims}', Y)

np.save(f'{save_path}/kSCA/ls_loss_kSCA_{d}d_sigma{sigma_noise}_proj_dims{proj_dims}', np.array(ls_loss))
np.save(f'{save_path}/kSCA/ls_S_ratio_kSCA_{d}d_sigma{sigma_noise}_proj_dims{proj_dims}', np.array(ls_S_ratio))

plt.figure()
get_loss_fig(ls_loss, ls_S_ratio)
plt.savefig(f'{save_path}/kSCA/loss_fig_kSCA_{d}d_sigma{sigma_noise}_proj_dims{proj_dims}.png')

plt.figure()
plot_3D_K_coded(Y)
plt.savefig(f'{save_path}/kSCA/projection_fig_kSCA_{d}d_sigma{sigma_noise}_proj_dims{proj_dims}.png')

### SCA ###
from linear_sca import *
wandb.init(project="", name="", mode="disabled")
U, ls_loss, ls_S_ratio = optimize(center(X), d=d, learning_rate=learning_rate, iterations=iterations) 
wandb.finish

U_qr, _ = jnp.linalg.qr(U)        
Y = jnp.einsum('ji,kjl->kil', U_qr, center(X))

np.save(f'{save_path}/SCA/U_SCA_{d}d_sigma{sigma_noise}_proj_dims{proj_dims}', U)
np.save(f'{save_path}/SCA/Y_SCA_{d}d_sigma{sigma_noise}_proj_dims{proj_dims}', Y)

for i in range(Y.shape[0]):
    Y = Y[:,:,:int(jnp.mean(RTs, axis=0).squeeze()[i])]

np.save(f'{save_path}/SCA/Y_SCA_cut_{d}d_sigma{sigma_noise}_proj_dims{proj_dims}', Y)

np.save(f'{save_path}/SCA/ls_loss_SCA_{d}d_sigma{sigma_noise}_proj_dims{proj_dims}', np.array(ls_loss))
np.save(f'{save_path}/SCA/ls_S_ratio_SCA_{d}d_sigma{sigma_noise}_proj_dims{proj_dims}', np.array(ls_S_ratio))

plt.figure()
get_loss_fig(ls_loss, ls_S_ratio)
plt.savefig(f'{save_path}/SCA/loss_fig_SCA_{d}d_sigma{sigma_noise}_proj_dims{proj_dims}.png')
    
plt.figure()
plot_3D_K_coded(Y)
plt.savefig(f'{save_path}/SCA/projection_fig_SCA_{d}d_sigma{sigma_noise}_proj_dims{proj_dims}.png')

### PCA ###
Y_pca, PCs = get_pca(X, num_pcs=d)

np.save(f'{save_path}/PCA/PCs_PCA_{d}d_sigma{sigma_noise}_proj_dims{proj_dims}', PCs)
np.save(f'{save_path}/PCA/Y_PCA_{d}d_sigma{sigma_noise}_proj_dims{proj_dims}', Y_pca)

for i in range(Y_pca.shape[0]):
    Y_pca = Y_pca[:,:,:int(jnp.mean(RTs, axis=0).squeeze()[i])]
    
np.save(f'{save_path}/PCA/Y_PCA_cut_{d}d_sigma{sigma_noise}_proj_dims{proj_dims}', Y_pca)

plt.figure() 
plot_3D_K_coded(jnp.array(Y_pca))
plt.savefig(f'{save_path}/PCA/projection_fig_PCA_{d}d_sigma{sigma_noise}_proj_dims{proj_dims}.png')
