import numpy as np

import jax
from jax import jit
import jax.numpy as jnp
from jax import grad, random, vmap
from jax.scipy.linalg import solve_triangular
import optax

import math

from utils import *
from kernels import * 

from itertools import combinations

import wandb

def center(x, axis=0):
    mean_x = jnp.mean(x, axis=(axis), keepdims=True)        
    return (x - mean_x)

def get_params(params, kernel_function):
    alpha_tilde = params['alpha_tilde']
    u = params['u']

    if kernel_function == 'linear':
        l2 = None
    elif kernel_function == 'gaussian':
        l_tilde = params['l_tilde']      
        l2 = l_tilde**2 + 0.01

    return alpha_tilde, u, l2

def get_alpha(params, A, X, kernel_function, d):
    K, N, T = X.shape

    alpha_tilde, u, l2 = get_params(params, kernel_function)
    c = u.shape[-1]

    if kernel_function == 'linear':
        K_u_u =  K_X_Y_identity(u, u)                
        K_A_u =  K_X_Y_identity(A, u) 
    elif kernel_function == 'gaussian':
        K_u_u =  K_X_Y_squared_exponential(u, u, l=l2)                 #(c, c)          
        K_A_u =  K_X_Y_squared_exponential(A, u, l=l2)                 #(K*T, c)

    K_A_u_reshaped = K_A_u.reshape(K,T,c)                                #(K, T, c)
    mean = jnp.mean(K_A_u_reshaped, axis=(0), keepdims=True)             #(1, T, c)
    H_K_A_u = (K_A_u_reshaped - mean).reshape(K*T,c)                     #(K*T,c)          
    L = jnp.linalg.cholesky(K_u_u + jnp.identity(c) * 1e-5)
    Q_, R = jnp.linalg.qr(H_K_A_u, mode='reduced')                                                                                        
    
    tmp = solve_triangular(L, H_K_A_u.T, lower=True)        #(c, KT)
    K_u_u_K_u_A = solve_triangular(L.T, tmp, lower=False)   #(c, KT)
    
    alpha_tilde_QR, _ = jnp.linalg.qr(alpha_tilde, mode='reduced') 
    alpha = jnp.einsum('ij,jm->im', Q_, solve_triangular(R.T, jnp.dot(L, alpha_tilde_QR), lower=True))                            #(K*T, c) @ (c, c) @ (c, c) @ (c, d) 
    alpha_reshaped = alpha.reshape(K,T,d)                           #(K, T, D)
    mean = jnp.mean(alpha_reshaped, axis=(0), keepdims=True)        #(1, T, D)
    alpha_H = (alpha_reshaped - mean).reshape(K*T,d)                #(K*T,D)

    K_u_u_K_u_A_alpha_H =  jnp.einsum('ij,jm->im',  K_u_u_K_u_A, alpha_H)   #(c, KT) @ (KT, d) --> (c, d)                       
    
    return  K_u_u_K_u_A_alpha_H
 
def loss(params, X, A, d,kernel_function, key, normalized = False):  
    K, N, T = X.shape

    _, u, l2 = get_params(params, kernel_function)

    K_u_u_K_u_A_alpha_H = get_alpha(params, A, X, kernel_function, d) 


    num_pairs = 100  
    indices = random.randint(key, shape=(num_pairs*2,), minval=0, maxval=K) 
    index_pairs = indices.reshape((num_pairs, 2))

    X1 = X[index_pairs[:, 0]].swapaxes(0,1).reshape(N,-1)
    X2 = X[index_pairs[:, 1]].swapaxes(0,1).reshape(N,-1)
    K_u_X1 = K_X_Y_squared_exponential(u, X1, l=l2).reshape(-1,num_pairs,T).swapaxes(0,1)    #(pairs, c, T)
    K_u_X2 = K_X_Y_squared_exponential(u, X2, l=l2).reshape(-1,num_pairs,T).swapaxes(0,1)  

    k1 = jnp.einsum('lji,jm->lim',  K_u_X1, K_u_u_K_u_A_alpha_H)                #(pair, T, d)
    k2 = jnp.einsum('lji,jm->lim',  K_u_X2, K_u_u_K_u_A_alpha_H) 
    
    k1 = center(k1)
    k2 = center(k2)

    Q = jnp.einsum('ktn,ltm->klnm', k1,k2)                  #(pair, pair, d, d)
    term2 = jnp.einsum('klnm,klmn->', Q,Q)
    term1 = jnp.square(jnp.einsum('klnn->kl', Q)).sum()

    if normalized == False:
        S = (2 / (num_pairs**2) ) * (term1-term2)
        return -S
    
    else:
        return (term1-term2) / (term1+term2)

def update(params, X, A, d, kernel_function, optimizer, opt_state, key):
    grad_loss = grad(loss)(params, X, A, d, kernel_function, key)
  
    updates, opt_state_updated = optimizer.update(grad_loss, opt_state, params)
    params_updated = optax.apply_updates(params, updates)
    return params_updated, opt_state_updated

def optimize(X, A, kernel_function='gaussian', iterations=10000, learning_rate=0.001, d=3, c=10, seed=42):
    K, N, T = X.shape

    key = random.PRNGKey(seed)
    
    alpha_tilde = random.normal(key, (c, d))             

    l_tilde = 0.1

    indices = jax.random.choice(key, A.shape[1], shape=(c,), replace=False)
    u = A[:, indices]    

    params = {
        'alpha_tilde': alpha_tilde,
        'u': u
    }

    if kernel_function == 'gaussian':
        params['l_tilde'] = l_tilde
    
    keys = random.split(key, num=iterations)

    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)

    ls_loss = []
    ls_S_ratio = []
    
    for i in range(iterations):
        params, opt_state = update(params, X, A, d, kernel_function, optimizer, opt_state, keys[i])        

        loss_ = loss(params, X, A, d, kernel_function,keys[i])
        S_ratio = loss(params, X, A, d, kernel_function, keys[i], normalized = True)

        wandb.log({"loss_": loss_, "S_ratio": S_ratio})

        ls_loss.append(loss_)
        ls_S_ratio.append(S_ratio)
        
        if i % 10 == 0:
            print(f"Iteration {i}, S: {-loss_}, S_ratio: {S_ratio}")

    return params, ls_loss, ls_S_ratio