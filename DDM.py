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
flags.DEFINE_integer('c', 30, 'inducing points')
flags.DEFINE_integer('iterations', 1000, 'iters')
flags.DEFINE_float('learning_rate', 1e-2, 'Initial learning rate.')
flags.DEFINE_string('save_path', '/rds/user/ar2217/hpc-work/SCA/outputs/DDM', 'save path')
flags.DEFINE_integer('proj_dims', 10, 'proj dims from DDM to neurons')
flags.DEFINE_float('sigma_noise', 0.5, 'low rank noise param')
flags.DEFINE_float('l2_', 1e-1, 'l2 param in RBF kernel when generating low rank correlated noise')

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
l2_ = FLAGS.l2_

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
            #if jnp.abs(x[i, t]) < 1:
            normal_sample = random.normal(current_key, shape=())
            x = x.at[i, t + 1].set(x[i, t] + mu[i] * dt + sigma * jnp.sqrt(dt) * normal_sample)
            current_key, subkey = random.split(current_key)
            # else:
            #     ls_RT.append(t)
            #     ls_acc.append(1 if jnp.sign(x[i, t]) == jnp.sign(mu[i]) else -1)
                
            #     x = x.at[i, t:].set(jnp.sign(x[i, t]))
            #     break
    return x, ls_RT, ls_acc

mu = np.array([-0.64, -0.32, -0.16, -0.08, -0.04, -0.02, 0.0, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64])  
dt = 0.1
total_time = 10
trials = 30    
split = 10

paths = jnp.zeros((trials, len(mu), int(total_time / dt) + 1))
for i in range(trials):
    key = random.PRNGKey(42 + i)
    path, ls_RT, ls_acc = DDM(mu, sigma, dt, total_time, key) 
    paths = paths.at[i].set(path)


def project(paths, key, proj_dims = 50):
    proj_matrix = random.normal(key, (proj_dims, 1))
    proj_matrix , _ = jnp.linalg.qr(proj_matrix)                                        #(N',N)
    return jnp.einsum('dn,lknt->lkdt', proj_matrix, paths[:,:,jnp.newaxis,:])           #(trial, K, N=1, T)

def relu(x):
    return jnp.maximum(0, x)

def add_low_rank_noise(X, key1, key2, proj_dims = 3, sigma_noise= 1 , l2_=0.1):
    trials, K, N, T = X.shape    
    B = random.normal(key1, (N, proj_dims))
    B, _ = jnp.linalg.qr(B)

    time_points = jnp.linspace(0, 1, T)[None, :]
    cov_matrix = K_X_Y_squared_exponential(time_points, time_points, l2=l2_)
    L = jnp.linalg.cholesky(cov_matrix + jnp.identity(T) * 1e-5)

    epsilon_t_uncorr = random.normal(key2, (trials, K, T, proj_dims)) * sigma_noise
    epsilon_t = jnp.einsum("ts,lksd->lktd", L, epsilon_t_uncorr)
    noise = jnp.einsum('lktd,nd->lknt', epsilon_t, B)             
    
    X += noise                                                   
    return X

key = random.PRNGKey(0)
key1, key2, key3 = random.split(key, 3)


neural_traces = relu(project(paths, key=key1, proj_dims=proj_dims))
neural_traces = neural_traces * 5
#neural_traces = add_low_rank_noise(neural_traces, key2, key3, sigma_noise = sigma_noise, l2_=l2_)        #(trials, K, N, T)

X_train = neural_traces[split:,:,:,:].reshape(-1, neural_traces.shape[-2],neural_traces.shape[-1] )
X_test = neural_traces[:split,:,:,:].reshape(-1, neural_traces.shape[-2],neural_traces.shape[-1] )

K, N, T = X_train.shape
A = jnp.swapaxes(X_train, 0, 1)                  #(N, K, T)
A = A.reshape(N,-1)                              #(N, K*T)

### kSCA ###
kernel_function = K_X_Y_squared_exponential

from kernel_sca_inducing_points import *
wandb.init(project="", name="", mode="disabled")
params, ls_loss, ls_S_ratio = optimize(X_train, A, iterations=iterations, learning_rate=learning_rate, d=d, c=c, kernel_function=kernel_function)
wandb.finish()

np.save(f'{save_path}/kSCA/params_kSCA_{d}d_sigma{sigma_noise}_proj_dims{proj_dims}_l2{l2_}', params)

_, u, l2, scale = get_params(params, kernel_function=kernel_function)
K_u_u_K_u_A_alpha_H, K_A_u, K_u_u, _, _  = get_alpha(params, A, X_train, kernel_function, d)

X_reshaped = X_train.swapaxes(0,1).reshape(N,-1)
K_u_X = kernel_function(u, X_reshaped, l2=l2, scale=scale).reshape(-1,K,T).swapaxes(0,1)  
Y = jnp.einsum('ji,kjm->kim',  K_u_u_K_u_A_alpha_H, K_u_X)
Y = center(Y)

np.save(f'{save_path}/kSCA/Y_kSCA_{d}d_sigma{sigma_noise}_proj_dims{proj_dims}_l2{l2_}', Y)

Y = jnp.mean(Y.reshape(trials - split, -1, d, T), axis=0)
np.save(f'{save_path}/kSCA/Y_mean_kSCA_{d}d_sigma{sigma_noise}_proj_dims{proj_dims}_l2{l2_}', Y)

plt.figure()
plot_3D_K_coded(Y)
plt.savefig(f'{save_path}/kSCA/projection_fig_kSCA_{d}d_sigma{sigma_noise}_proj_dims{proj_dims}_l2{l2_}.png')

K_u_u_K_u_A_alpha_H, K_A_u, K_u_u, _ ,_ = get_alpha(params, A, X_test, kernel_function, d)
K_test, _, _ = X_test.shape
X_reshaped = X_test.swapaxes(0,1).reshape(N,-1)
K_u_X = kernel_function(u, X_reshaped, l2=l2, scale=scale).reshape(-1,K_test,T).swapaxes(0,1)  
Y_test = jnp.einsum('ji,kjm->kim',  K_u_u_K_u_A_alpha_H, K_u_X)
Y_test = center(Y_test)

np.save(f'{save_path}/kSCA/Y_test_kSCA_{d}d_sigma{sigma_noise}_proj_dims{proj_dims}_l2{l2_}', Y_test)
Y_test = jnp.mean(Y_test.reshape(split, -1, d, T), axis=0)
np.save(f'{save_path}/kSCA/Y_mean_test_kSCA_{d}d_sigma{sigma_noise}_proj_dims{proj_dims}_l2{l2_}', Y_test)

np.save(f'{save_path}/kSCA/ls_loss_kSCA_{d}d_sigma{sigma_noise}_proj_dims{proj_dims}_l2{l2_}', np.array(ls_loss))
np.save(f'{save_path}/kSCA/ls_S_ratio_kSCA_{d}d_sigma{sigma_noise}_proj_dims{proj_dims}_l2{l2_}', np.array(ls_S_ratio))

plt.figure()
get_loss_fig(ls_loss, ls_S_ratio)
plt.savefig(f'{save_path}/kSCA/loss_fig_kSCA_{d}d_sigma{sigma_noise}_proj_dims{proj_dims}_l2{l2_}.png')

### SCA ###
from linear_sca import *
wandb.init(project="", name="", mode="disabled")
U, ls_loss, ls_S_ratio = optimize(center(X_train), d=d, learning_rate=learning_rate, iterations=iterations) 
wandb.finish

np.save(f'{save_path}/SCA/U_SCA_{d}d_sigma{sigma_noise}_proj_dims{proj_dims}_l2{l2_}', U)

U_qr, _ = jnp.linalg.qr(U)        
Y = jnp.einsum('ji,kjl->kil', U_qr, center(X_train))

np.save(f'{save_path}/SCA/Y_SCA_{d}d_sigma{sigma_noise}_proj_dims{proj_dims}_l2{l2_}', Y)

Y = jnp.mean(Y.reshape(trials - split, -1, d, T), axis=0)
np.save(f'{save_path}/SCA/Y_mean_SCA_{d}d_sigma{sigma_noise}_proj_dims{proj_dims}_l2{l2_}', Y)

plt.figure()
plot_3D_K_coded(Y)
plt.savefig(f'{save_path}/SCA/projection_fig_SCA_{d}d_sigma{sigma_noise}_proj_dims{proj_dims}_l2{l2_}.png')
   
Y_test = jnp.einsum('ji,kjl->kil', U_qr, center(X_test))
np.save(f'{save_path}/SCA/Y_test_SCA_{d}d_sigma{sigma_noise}_proj_dims{proj_dims}_l2{l2_}', Y_test)
Y_test = jnp.mean(Y_test.reshape(split, -1, d, T), axis=0)
np.save(f'{save_path}/SCA/Y_mean_test_SCA_{d}d_sigma{sigma_noise}_proj_dims{proj_dims}_l2{l2_}', Y_test)

np.save(f'{save_path}/SCA/ls_loss_SCA_{d}d_sigma{sigma_noise}_proj_dims{proj_dims}_l2{l2_}', np.array(ls_loss))
np.save(f'{save_path}/SCA/ls_S_ratio_SCA_{d}d_sigma{sigma_noise}_proj_dims{proj_dims}_l2{l2_}', np.array(ls_S_ratio))

plt.figure()
get_loss_fig(ls_loss, ls_S_ratio)
plt.savefig(f'{save_path}/SCA/loss_fig_SCA_{d}d_sigma{sigma_noise}_proj_dims{proj_dims}_l2{l2_}.png')
    


### PCA ###
#Y_pca, PCs = get_pca(center(X_train), num_pcs=d)

X_pca_train = center(X_train).swapaxes(1,2).reshape(-1, N)
X_pca_test = center(X_test).swapaxes(1,2).reshape(-1, N)

pca = PCA(d)
Y_pca = pca.fit(X_pca_train).transform(X_pca_train)
PCs = pca.components_
Y_pca = Y_pca.reshape(-1, T, d).swapaxes(1,2)

np.save(f'{save_path}/PCA/PCs_PCA_{d}d_sigma{sigma_noise}_proj_dims{proj_dims}_l2{l2_}', PCs)
np.save(f'{save_path}/PCA/Y_PCA_{d}d_sigma{sigma_noise}_proj_dims{proj_dims}_l2{l2_}', Y_pca)


Y_pca = jnp.mean(Y_pca.reshape(trials - split, -1, d, T), axis=0)
np.save(f'{save_path}/PCA/Y_mean_PCA_{d}d_sigma{sigma_noise}_proj_dims{proj_dims}_l2{l2_}', Y_pca)

plt.figure() 
plot_3D_K_coded(jnp.array(Y_pca))
plt.savefig(f'{save_path}/PCA/projection_fig_PCA_{d}d_sigma{sigma_noise}_proj_dims{proj_dims}_l2{l2_}.png')

Y_pca_test = pca.fit(X_pca_train).transform(X_pca_test)
Y_pca_test = Y_pca_test.reshape(-1, T, d).swapaxes(1,2)
np.save(f'{save_path}/PCA/Y_test_PCA_{d}d_sigma{sigma_noise}_proj_dims{proj_dims}_l2{l2_}', Y_pca_test)
Y_pca_test = jnp.mean(Y_pca_test.reshape(split, -1, d, T), axis=0)
np.save(f'{save_path}/PCA/Y_mean_test_PCA_{d}d_sigma{sigma_noise}_proj_dims{proj_dims}_l2{l2_}', Y_pca_test)



