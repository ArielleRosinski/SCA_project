import math

import jax
import jax.numpy as jnp
from jax import grad, random, vmap

from utils import *
from kernels import *

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from scipy.ndimage import gaussian_filter

from absl import app
from absl import flags
import sys
import os

flags.DEFINE_integer('seed', 41, 'Random seed to set')
flags.DEFINE_string('traj', 'rotation',
                     'toy data set to generate')
flags.DEFINE_string('save_path', '/rds/user/ar2217/hpc-work/SCA/outputs/toy_data',
                     'save path')
flags.DEFINE_string('kernel', 'gaussian',
                     'type of kernel used')
flags.DEFINE_integer('iterations', 10000, 'training iterations')
flags.DEFINE_float('learning_rate', 1e-3, 'Initial learning rate.')
flags.DEFINE_string('linear_run', 'True', 'Run linear methods')
flags.DEFINE_integer('sigma', 4, 'Gaussian smoothing')
flags.DEFINE_float('sigma_low_rank', 0.75, 'Low rank noise parameter')

FLAGS = flags.FLAGS
FLAGS(sys.argv)

seed = FLAGS.seed
traj = FLAGS.traj
save_path = FLAGS.save_path
kernel = FLAGS.kernel
iterations = FLAGS.iterations
learning_rate = FLAGS.learning_rate
linear_run = FLAGS.linear_run
sigma = FLAGS.sigma
sigma_low_rank = FLAGS.sigma_low_rank

def get_rotation_params(K, T, key):
    time = jnp.linspace(0, 2 *jnp.pi, T)[:, jnp.newaxis] #4
    radii = jnp.linspace(0.1, 2, K)
    radii = random.permutation(key, radii)
    phases = jnp.linspace(0, 2*jnp.pi, K)
    return time, radii, phases

def get_rotations(K, T, key):
    time, radii, phases = get_rotation_params(K, T, key)

    sine_waves = jnp.sin(time + phases) * radii
    cosine_waves = jnp.cos(time + phases) * radii
    X = jnp.stack([cosine_waves.T, sine_waves.T], axis=1)  
    return X

def get_infty_traj(K, T, key):
    time, radii, phases = get_rotation_params(K, T, key)

    r = radii * jnp.cos(time + phases)
    theta = radii * jnp.sin(time + phases)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    X = jnp.stack([x.T, y.T], axis=1)
    return X

def get_expansions(K, T, key1, key2, oneD=False):
    if oneD == False:
        initial_rates = 0.05 * (random.uniform(key1, shape=(K, 2, 1)) - 0.25)
        initial_rates = random.permutation(key2, initial_rates)
    else:
        initial_rates = 0.1 * (random.uniform(key1, shape=(K, 1, 1)) - 0.5)
        initial_rates = random.permutation(key2, initial_rates)
    time_steps = jnp.arange(T)
    X = jnp.exp(initial_rates * time_steps)
    return X

def rotation_system(t, y):
    x1, x2 = y
    dx1_dt = - x2 
    dx2_dt = x1 
    return [dx1_dt, dx2_dt]

def pendulum_system(t, y):
    x1, x2 = y
    dx1_dt = x2 
    dx2_dt = -np.sin(x1) 
    return [dx1_dt, dx2_dt]

def duffing_oscillator(t, y):
    x1, x2 = y
    dx1_dt = x2 
    dx2_dt = x1 - x1**3 
    return [dx1_dt, dx2_dt]

def van_der_pol(t, y):
    x1, x2 = y
    dx1_dt = x2 
    dx2_dt = (1 - x1**2) * x2 - x1 
    return [dx1_dt, dx2_dt]

def get_oscillator(K, T, seed = seed, type=van_der_pol):
    np.random.seed(seed) 
    initial_conditions_list = np.random.uniform(low=-np.pi, high=np.pi, size=(K, 2))

    t_span = (0, 10)   
    t_eval = np.linspace(t_span[0], t_span[1], T)  
    X = np.zeros((K, 2, T))
    for i, initial_conditions in enumerate(initial_conditions_list):
        solution = solve_ivp(type, t_span, initial_conditions, t_eval=t_eval)
        x1 = solution.y[0]
        x2 = solution.y[1]

        X[i, 0, :] = x1
        X[i, 1, :] = x2
    X = jnp.array(X)
    return X

def project_X(X, key, proj_dims = 50):
    proj_matrix = random.normal(key, (proj_dims, X.shape[1]))
    proj_matrix , _ = jnp.linalg.qr(proj_matrix)                    #(N',N)
    X = jnp.einsum('lj,ijk->ilk', proj_matrix, X)                   #(K, N', T)
    return X

def add_isotropic_noise(X, key):
    noise = random.normal(key, (X.shape)) 
    X += 0.01 * (noise) 
    return X 

def add_random_modes(X, key1, key2, mean=0, std_dev=1):
    K, _, T = X.shape
    trajectories_1 = random.normal(key1, (K, T), dtype=X.dtype) * std_dev + mean
    trajectories_2 = random.normal(key2, (K, T), dtype=X.dtype) * std_dev + mean
    combined_trajectories = jnp.stack((trajectories_1, trajectories_2), axis=1)
    X = jnp.concatenate((X, combined_trajectories), axis=1)
    return X

def add_low_rank_noise(X, key1, key2, proj_dims = 3, sigma = 0.75 ):
    K, N, T = X.shape    
    B = random.normal(key1, (N, proj_dims))
    B, _ = jnp.linalg.qr(B)

    epsilon_t = random.normal(key2, (K, T, proj_dims)) * sigma  
    noise = jnp.einsum('ktd,nd->knt', epsilon_t, B)             
    
    X += noise                                                   
    return X




key = random.PRNGKey(seed)
key1, key2, key3, key4, key5 = random.split(key, 5)

K = 100
T = 50
split = 20
d = 2 
c = 30

dir_path = os.path.join(save_path, traj, kernel)
if not os.path.exists(dir_path):
    os.makedirs(dir_path)

dir_path = os.path.join(save_path, traj, 'linear')
if not os.path.exists(dir_path):
    os.makedirs(dir_path)

dir_path = os.path.join(save_path, traj, 'pca')
if not os.path.exists(dir_path):
    os.makedirs(dir_path)

if traj == 'rotation':
    X = get_rotations(K, T, key1)
    np.save(f'{save_path}/{traj}/X', X)
    X = project_X(X, key2)
    X = add_low_rank_noise(X, key3, key4, sigma=sigma_low_rank)
elif traj == 'infty':
    X = get_infty_traj(K, T, key1)
    np.save(f'{save_path}/{traj}/X', X)
    X = project_X(X, key2)
    X = add_low_rank_noise(X, key3, key4, sigma=sigma_low_rank)
elif traj == 'expansion':
    X = get_expansions(K, T, key1, key2)
    np.save(f'{save_path}/{traj}/X', X)
    X = project_X(X, key3)
    X = add_low_rank_noise(X, key4, key5, sigma=sigma_low_rank)
elif traj == 'expansion1D':
    X = get_expansions(K, T, key1, key2, oneD=True)
    np.save(f'{save_path}/{traj}/X', X)
    X = project_X(X, key3)
    X = add_low_rank_noise(X, key4, key5, sigma=sigma_low_rank)
elif traj == 'vdp_isotropic':
    X = get_oscillator(K, T, type=van_der_pol, seed = seed)
    np.save(f'{save_path}/{traj}/X', X)
    X = project_X(X, key1)
    X = add_isotropic_noise(X, key2)
elif traj == 'duffing_isotropic':
    X = get_oscillator(K, T, type=duffing_oscillator, seed = seed)
    np.save(f'{save_path}/{traj}/X', X)
    X = project_X(X, key1)
    X = add_isotropic_noise(X, key2)
elif traj == 'rotation_isotropic':
    X = get_rotations(K, T, key1)
    np.save(f'{save_path}/{traj}/X', X)
    X = project_X(X, key2)
    X = add_isotropic_noise(X, key3)
elif traj == 'expansion1D_isotropic':
    X = get_expansions(K, T, key1, key2, oneD=True)
    np.save(f'{save_path}/{traj}/X', X)
    X = project_X(X, key3)
    X = add_isotropic_noise(X, key4)
  
X_train=X[split:]
X_test=X[:split]
K, N, T = X_train.shape
A = jnp.swapaxes(X_train, 0, 1)                  #(N, K, T)
A = A.reshape(N,-1)                              #(N, K*T)

np.save(f'{save_path}/{traj}/X_train_{sigma_low_rank}', X_train)
np.save(f'{save_path}/{traj}/X_test_{sigma_low_rank}', X_test)

if kernel =='RQ':
    kernel_function=K_X_Y_rational_quadratic
elif kernel =='gaussian':
    kernel_function=K_X_Y_squared_exponential

### kSCA ###
from kernel_sca_inducing_points import *
wandb.init(project="", name="", mode="disabled")
params, ls_loss, ls_S_ratio = optimize(X_train, A, iterations=iterations, learning_rate=learning_rate, d=d, c=c, kernel_function=kernel_function)
wandb.finish()

plt.figure()
get_loss_fig(ls_loss, ls_S_ratio)
plt.savefig(f'{save_path}/{traj}/{kernel}/loss_fig_{sigma_low_rank}.png')

_, u, l2, scale = get_params(params, kernel_function=kernel_function)
K_u_u_K_u_A_alpha_H, K_A_u, K_u_u  = get_alpha(params, A, X_train, kernel_function, d)
X_reshaped = X_train.swapaxes(0,1).reshape(N,-1)
K_u_X = kernel_function(u, X_reshaped, l2=l2, scale=scale).reshape(-1,K,T).swapaxes(0,1)  
Y = jnp.einsum('ji,kjm->kim',  K_u_u_K_u_A_alpha_H, K_u_X)
Y = center(Y)

plt.figure()
plot_2D(Y)
plt.title(f'kSCA; s = {compute_S_all_pairs(Y)}')
plt.savefig(f'{save_path}/{traj}/{kernel}/projection_fig_{sigma_low_rank}.png')

Y_smoothed = apply_gaussian_smoothing(Y, sigma=sigma)
plt.figure()
plot_2D(Y_smoothed)
plt.savefig(f'{save_path}/{traj}/{kernel}/projection_smoothed_fig_{sigma_low_rank}.png')

np.save(f'{save_path}/{traj}/{kernel}/Y_train_{sigma_low_rank}', Y)

_, u, l2, scale = get_params(params, kernel_function=kernel_function)
K_u_u_K_u_A_alpha_H, K_A_u, K_u_u  = get_alpha(params, A, X_test, kernel_function, d)
K_test, _, _ = X_test.shape
X_reshaped = X_test.swapaxes(0,1).reshape(N,-1)
K_u_X = kernel_function(u, X_reshaped, l2=l2, scale=scale).reshape(-1,K_test,T).swapaxes(0,1)  
Y = jnp.einsum('ji,kjm->kim',  K_u_u_K_u_A_alpha_H, K_u_X)
Y = center(Y)

np.save(f'{save_path}/{traj}/{kernel}/Y_test_{sigma_low_rank}', Y)

np.save(f'{save_path}/{traj}/{kernel}/ls_loss_{sigma_low_rank}', np.array(ls_loss))
np.save(f'{save_path}/{traj}/{kernel}/ls_S_ratio_{sigma_low_rank}', np.array(ls_S_ratio))

if linear_run == 'True':
    ### LINEAR SCA ###
    from linear_sca import *
    wandb.init(project="", name="", mode="disabled")
    U, ls_loss, ls_S_ratio = optimize(center(X_train), d=d, iterations=iterations, learning_rate=learning_rate) 
    wandb.finish

    plt.figure()
    get_loss_fig(ls_loss, ls_S_ratio)
    plt.savefig(f'{save_path}/{traj}/linear/loss_fig_{sigma_low_rank}.png')

    U_qr, _ = jnp.linalg.qr(U)        
    Y = jnp.einsum('ji,kjl->kil', U_qr, center(X_train))

    plt.figure()
    plot_2D(Y)
    plt.title(f'SCA; s = {compute_S_all_pairs(Y)}')
    plt.savefig(f'{save_path}/{traj}/linear/projection_fig_{sigma_low_rank}.png')

    Y_smoothed = apply_gaussian_smoothing(Y, sigma=sigma)
    plt.figure()
    plot_2D(Y_smoothed)
    plt.savefig(f'{save_path}/{traj}/linear/projection_smoothed_fig_{sigma_low_rank}.png')

    np.save(f'{save_path}/{traj}/linear/Y_train_{sigma_low_rank}', Y)

    U_qr, _ = jnp.linalg.qr(U)        
    Y = jnp.einsum('ji,kjl->kil', U_qr, center(X_test))

    np.save(f'{save_path}/{traj}/linear/Y_test_{sigma_low_rank}', Y)

    np.save(f'{save_path}/{traj}/linear/ls_loss_{sigma_low_rank}', np.array(ls_loss))
    np.save(f'{save_path}/{traj}/linear/ls_S_ratio_{sigma_low_rank}', np.array(ls_S_ratio))

    ### PCA ###
    X_pca_train = center(X_train).swapaxes(1,2).reshape(-1, N)
    X_pca_test = center(X_test).swapaxes(1,2).reshape(-1, N)

    pca = PCA(d)
    Y_pca = pca.fit(X_pca_train).transform(X_pca_train)
    Y_pca = Y_pca.reshape(-1, T, d).swapaxes(1,2)

    plot_2D(Y_pca)
    plt.title(f'PCA; s = {compute_S_all_pairs(jnp.array(Y_pca))}')
    plt.savefig(f'{save_path}/{traj}/pca/projection_fig_{sigma_low_rank}.png')

    Y_smoothed = apply_gaussian_smoothing(Y_pca, sigma=sigma)
    plt.figure()
    plot_2D(Y_smoothed)
    plt.savefig(f'{save_path}/{traj}/pca/projection_smoothed_fig_{sigma_low_rank}.png')

    np.save(f'{save_path}/{traj}/pca/Y_train_{sigma_low_rank}', Y_pca)

    pca = PCA(d)
    Y_pca = pca.fit(X_pca_train).transform(X_pca_test)
    Y_pca = Y_pca.reshape(-1, T, d).swapaxes(1,2)

    np.save(f'{save_path}/{traj}/pca/Y_test_{sigma_low_rank}', Y_pca)