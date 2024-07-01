import numpy as np

import jax
from jax import jit
import jax.numpy as jnp
from jax import grad, random, vmap
from jax.lax.linalg import triangular_solve
import optax

import math

from utils import *
from kernels import * 

import wandb

def get_params(params, kernel_function):
    alpha_tilde = params['alpha_tilde']
    u = params['u']

    if kernel_function == 'linear':
        l = None
        sigma_f = None 
    elif kernel_function == 'gaussian':
        l_tilde = params['l_tilde']      
        l2 = l_tilde**2 + 0.01

        #sigma_f_tilde = params['sigma_f_tilde'] 
        #sigma_f = jnp.exp(sigma_f_tilde)
        sigma_f = 0.1
    return alpha_tilde, u, l2, sigma_f

def get_alpha(params, A, X, kernel_function):
    K, N, T = X.shape

    alpha_tilde, u, l, sigma_f = get_params(params, kernel_function)
    c = u.shape[-1]

    if kernel_function == 'linear':
        K_u_u =  K_X_Y_identity(u, u)                
        K_A_u =  K_X_Y_identity(A, u) 
    elif kernel_function == 'gaussian':
        K_u_u =  K_X_Y_squared_exponential(u, u, l=l, sigma_f=sigma_f)                 #(c, c)          
        K_A_u =  K_X_Y_squared_exponential(A, u, l=l, sigma_f=sigma_f)                 #(K*T, c)

    K_A_u_reshaped = K_A_u.reshape(K,T,c)                                #(K, T, c)
    mean = jnp.mean(K_A_u_reshaped, axis=(0), keepdims=True)             #(1, T, c)
    H_K_A_u = (K_A_u_reshaped - mean).reshape(K*T,c)                     #(K*T,c)           #K_A_A = jnp.einsum('kc,cj,mj ->km',  H_K_A_u, jnp.linalg.inv(K_u_u), H_K_A_u)       #K_A_u @ K_u_u @ K_A_u.T (K*T, K*T)

    L = jnp.linalg.cholesky(K_u_u + jnp.identity(c) * 1e-5)
    Q, R = jnp.linalg.qr(H_K_A_u, mode='reduced')                                                                                        #(mode reduced Q(kt, c) R(c,c) versus complete Q (kt,kt) and R(KT, c))
    alpha_tilde_QR, _ = jnp.linalg.qr(alpha_tilde) 
    alpha = jnp.einsum('kc, cl, lj, jd -> kd', Q, triangular_solve(R.T, jnp.eye(R.shape[0]), lower=True), L, alpha_tilde_QR) #(K*T, c) @ (c, c) @ (c, c) @ (c, d) 
    K_u_u_inv = jnp.linalg.solve(K_u_u, jnp.eye(K_u_u.shape[0]))                                                             #K_u_u_inv = jnp.linalg.inv(K_u_u)
    return alpha, K_A_u, K_u_u, H_K_A_u, K_u_u_inv

def single_pair_loss(alpha_H, X, params, kernel_function, K_A_u, K_u_u_inv, id_1, id_2, operator = 'minus'):
    
    _, u, l, sigma_f = get_params(params, kernel_function)

    if kernel_function == 'linear':
        K_u_X_i =  K_X_Y_identity(u, X[id_1])                              #(c, K, T)
        K_X_u_i =  K_X_Y_identity(u, X[id_2]).T
    elif kernel_function == 'gaussian':
        K_u_X_i =  K_X_Y_squared_exponential(u, X[id_1], l=l, sigma_f=sigma_f)                              #(c, K, T)
        K_X_u_i =  K_X_Y_squared_exponential(u, X[id_2], l=l, sigma_f=sigma_f).T

    K_A_X_i = jnp.einsum('kc,cj,jm ->km',  K_A_u, K_u_u_inv, K_u_X_i) 
    K_X_A_i = jnp.einsum('kc,cj,mj ->km',  K_X_u_i, K_u_u_inv, K_A_u) 

                  
    Q = jnp.einsum('kd,kt,tj,jm->dm', alpha_H, K_A_X_i, K_X_A_i, alpha_H)       #(KT,D).T @ (KT,T) and (T,KT) @ (KT,D) --> (D,T) @ (T,D) --> (D,D)
    QQ_product = Q @ Q                                                          #jnp.einsum('ij,jm->im', Q, Q)

    if operator == 'minus':
        return jnp.trace(Q)**2 - jnp.trace(QQ_product) # last term can be einsum(Q,Q,'nm,mn->')
    
    elif operator == 'plus':
        return jnp.trace(Q)**2 + jnp.trace(QQ_product)
 

def loss(params, X, A, d,kernel_function, key, normalized = False):  
    K, N, T = X.shape

    alpha, K_A_u, _, _, K_u_u_inv = get_alpha(params, A, X, kernel_function)

    alpha_reshaped = alpha.reshape(K,T,d)                           #(K, T, D)
    mean = jnp.mean(alpha_reshaped, axis=(0), keepdims=True)        #(1, T, D)
    alpha_H = (alpha_reshaped - mean).reshape(K*T,d)                #(K*T,D)

    num_pairs = 100  
    indices = random.randint(key, shape=(num_pairs*2,), minval=0, maxval=K) 
    index_pairs = indices.reshape((num_pairs, 2))

    batched_loss = vmap(single_pair_loss, in_axes=(None, None, None, None, None, None, 0, 0))(alpha_H, X, params, kernel_function, K_A_u, K_u_u_inv, index_pairs[:, 0], index_pairs[:, 1]) #(num_pairs)

    if normalized == False:
        S = (2 / (num_pairs**2) ) * jnp.sum(batched_loss)
        return -S 
    else: 
        batched_normalizer = vmap(single_pair_loss, in_axes=(None, None, None, None, None, None, 0, 0, None))(alpha_H, X, params, kernel_function, K_A_u, K_u_u_inv, index_pairs[:, 0], index_pairs[:, 1], 'plus')
        return jnp.sum(batched_loss) / jnp.sum(batched_normalizer)

def update(params, X, A, d, kernel_function, optimizer, opt_state, key):
    grad_loss = grad(loss)(params, X, A, d, kernel_function, key)
  
    updates, opt_state_updated = optimizer.update(grad_loss, opt_state, params)
    params_updated = optax.apply_updates(params, updates)
    return params_updated, opt_state_updated

def optimize(X, A, kernel_function='gaussian', iterations=10000, learning_rate=0.001, d=3, c=10, seed=42):
    K, N, T = X.shape

    key = random.PRNGKey(seed)
    
    alpha_tilde = random.normal(key, (c, d))             
    #l_tilde, sigma_f_tilde = random.normal(key, (2,))     initialize at 1   
    l_tilde = jnp.sqrt(0.1)
    sigma_f_tilde = 0.1

    indices = jax.random.choice(key, A.shape[1], shape=(c,), replace=False)
    u = A[:, indices]    

    params = {
        'alpha_tilde': alpha_tilde,
        'u': u
    }

    if kernel_function == 'gaussian':
        params['l_tilde'] = l_tilde
        params['sigma_f_tilde'] = sigma_f_tilde  
    
    keys = random.split(key, num=iterations)

    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)

    ls_loss = []
    ls_S_ratio = []
    
    for i in range(iterations):
        params, opt_state = update(params, X, A, d, kernel_function, optimizer, opt_state, keys[i])        

        loss_ = loss(params, X, A, d, kernel_function,keys[i])
        S_ratio = loss(params, X, A, d, kernel_function, keys[i], normalized = True)

        #wandb.log({"loss_": loss_, "S_ratio": S_ratio})

        ls_loss.append(loss_)
        ls_S_ratio.append(S_ratio)
        
        if i % 10 == 0:
            print(f"Iteration {i}, S: {-loss_}, S_ratio: {S_ratio}")

    return params, ls_loss, ls_S_ratio

