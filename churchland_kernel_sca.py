import numpy as np 

import jax
import jax.numpy as jnp
from jax import grad, random, vmap

import optax

import math

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from linear_sca import *

import wandb

from absl import app
from absl import flags
import sys

flags.DEFINE_integer('seed', 42, 'Random seed to set')
flags.DEFINE_integer('iterations', 10000, 'training iterations')
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
flags.DEFINE_integer('d', 3, 'Subspace dimensionality')
flags.DEFINE_string('path', '../datasets/churchland.npy',
                     'dataset path')
flags.DEFINE_string('run_name', 'a_linear_kernel',
                     'name of the run and of the saved file')
flags.DEFINE_boolean('save', True,
                     'Whether to save the learned parameters')

FLAGS = flags.FLAGS
FLAGS(sys.argv)

seed = FLAGS.seed
iterations = FLAGS.iterations
learning_rate = FLAGS.learning_rate
path = FLAGS.path 
d = FLAGS.d
name = FLAGS.run_name
save = FLAGS.save

X_init = np.load(path) 

X, _ = pre_processing(X_init, center=True)
X = jnp.array(X)
K, N, T = X.shape
A = jnp.swapaxes(pre_processing(X_init)[0], 0, 1)       #(N, K, T)
A = A.reshape(N,-1)                                                #(N, K*T)

def K_X_Y_diagonal(X, Y, sigma_sqrd):
    """For two spatial patterns X and Y, the kernel k(x_i,y_i) is equal to sum_i sigma_i^2 x_i y_i"""
    return jnp.dot(X.T * sigma_sqrd, Y) 

def K_X_Y_identity(X, Y):
    return jnp.dot(X.T, Y) 


K_A_X = K_X_Y_identity(A, X)                                    #(K*T, K, T)

K_A_A = K_X_Y_identity(A, A)
K_A_A_reshaped = K_A_A.reshape(K,T,K,T)                          #(K,T,K,T)
means = jnp.mean(K_A_A_reshaped, axis=(0, 2), keepdims=True)     #(1, T, 1, T)
K_A_A_tilde = (K_A_A_reshaped - means).reshape(K*T,K*T)          #(K*T,K*T)
P, S, Pt = jnp.linalg.svd(K_A_A_tilde, full_matrices=False)      #P is (K*T, K*T) and S is (K*T,)

def single_pair_loss(alpha_H, K_A_X, id_1, id_2, operator = 'minus'):
    K_A_X_i = K_A_X[:,id_1,:]
    K_X_A_i = K_A_X[:,id_2,:].T
    
    #Q = alpha_H.T @ K_A_X_i @ K_X_A_i @ alpha_H                         #(KT,D).T @ (KT,T) and (T,KT) @ (KT,D) --> (D,T) @ (T,D) --> (D,D)
    Q = jnp.einsum('kd,kt,tj,jm->dm', alpha_H, K_A_X_i, K_X_A_i, alpha_H)
    #QQ_product = jnp.einsum('ij,lm->im', Q, Q)
    QQ_product = Q @ Q


    if operator == 'minus':
        return jnp.trace(Q)**2 - jnp.trace(QQ_product)
    
    elif operator == 'plus':
        return jnp.trace(Q)**2 + jnp.trace(QQ_product)
 

def loss(alpha_tilde, P, S, K_A_X, X, d, key, normalized = False):  
    K, N, T = X.shape
    
    alpha_tilde_QR, _ = jnp.linalg.qr(alpha_tilde) 
    #alpha = jnp.dot(P , 1/jnp.sqrt(S))[:,None] * alpha_tilde_QR
    alpha = (P / jnp.sqrt(S)) @ alpha_tilde_QR


    alpha_reshaped = alpha.reshape(K,T,d)                           #(K, T, D)
    mean = jnp.mean(alpha_reshaped, axis=(0), keepdims=True)        #(1, T, D)
    alpha_H = (alpha_reshaped - mean).reshape(K*T,d)                #(K*T,D)

    num_pairs = 100  
    indices = random.randint(key, shape=(num_pairs*2,), minval=0, maxval=N)
    index_pairs = indices.reshape((num_pairs, 2))

    batched_loss = vmap(single_pair_loss, in_axes=(None, None, 0, 0))(alpha_H, K_A_X, index_pairs[:, 0], index_pairs[:, 1]) #(num_pairs)

    if normalized == False:
        S = (2 / (num_pairs**2) ) * jnp.sum(batched_loss)
        return -S
    else: 
        batched_normalizer = vmap(single_pair_loss, in_axes=(None, None, 0, 0, None))(alpha_H, K_A_X, index_pairs[:, 0], index_pairs[:, 1], 'plus')
        return jnp.sum(batched_loss) / jnp.sum(batched_normalizer)

def update(alpha_tilde, P, S, K_A_X, X, d, optimizer, opt_state, key):
    grad_loss = grad(loss)(alpha_tilde, P, S, K_A_X, X, d, key)
  
    updates, opt_state_updated = optimizer.update(grad_loss, opt_state, alpha_tilde)
    alpha_tilde_updated = optax.apply_updates(alpha_tilde, updates)
    return alpha_tilde_updated, opt_state_updated

def optimize(P, S, K_A_X, X, iterations=10000, learning_rate=0.001, d=3, seed=42):
    K, N, T = X.shape
    key = random.PRNGKey(seed)
    
    alpha_tilde = random.normal(key, (K*T, d))
    
    keys = random.split(key, num=iterations)

    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(alpha_tilde)

    ls_loss = []
    ls_S_ratio = []
    
    for i in range(iterations):
        alpha_tilde, opt_state = update(alpha_tilde, P, S, K_A_X, X, d, optimizer, opt_state, keys[i])        

        loss_ = loss(alpha_tilde, P, S, K_A_X, X, d, keys[i])
        S_ratio = loss(alpha_tilde, P, S, K_A_X, X, d, keys[i], normalized = True)

        wandb.log({"loss_": loss_, "S_ratio": S_ratio})

        ls_loss.append(loss_)
        ls_S_ratio.append(S_ratio)
        
        if i % 10 == 0:
            print(f"Iteration {i}, S: {-loss_}, S_ratio: {S_ratio}")

    return alpha_tilde, ls_loss, ls_S_ratio


wandb.init(project="SCA-project-kernel", name=name, mode="online")
optimized_alpha_tilde, _,  _ = optimize(P, S, K_A_X, X, iterations= iterations, learning_rate= learning_rate, seed = seed )
wandb.finish()

if save: 
    np.save(f'../outputs/{name}', optimized_alpha_tilde)

