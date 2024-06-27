import numpy as np 
from numpy.linalg import qr, svd
from scipy.linalg import subspace_angles

import jax
import jax.numpy as jnp
from jax import grad, random, vmap, jit
import optax

import wandb

from itertools import combinations

@jit
def qr_decomposition(A):
    m, n = A.shape
    Q = jnp.zeros((m, n))
    R = jnp.zeros((n, n))
    A_copy = A.copy()

    for j in range(n):
        R = R.at[j, j].set(jnp.linalg.norm(A_copy[:, j]))
        Q = Q.at[:, j].set(A_copy[:, j] / R[j, j])
        for i in range(j+1, n):
            R = R.at[j, i].set(jnp.dot(Q[:, j].T, A_copy[:, i]))
            A_copy = A_copy.at[:, i].set(A_copy[:, i] - Q[:, j] * R[j, i])

    return Q, R

def single_pair_loss(alpha_H, K_A_X, id_1, id_2, operator = 'minus'):
    K_A_X_i = K_A_X[:,id_1,:]
    K_X_A_i = K_A_X[:,id_2,:].T
                  
    Q = jnp.einsum('kd,kt,tj,jm->dm', alpha_H, K_A_X_i, K_X_A_i, alpha_H)    #(KT,D).T @ (KT,T) and (T,KT) @ (KT,D) --> (D,T) @ (T,D) --> (D,D)
    QQ_product = jnp.einsum('ij,jm->im', Q, Q)                                # jnp.einsum('ij,jm->im', Q, Q) == Q @ Q   

    if operator == 'minus':
        return jnp.trace(Q)**2 - jnp.trace(QQ_product)
    
    elif operator == 'plus':
        return jnp.trace(Q)**2 + jnp.trace(QQ_product)
 

def loss(alpha_tilde, P, S, K_A_X, X, d, key, normalized = False):  
    K, N, T = X.shape
    
    alpha_tilde_QR, _ = jnp.linalg.qr(alpha_tilde) 
    alpha = (P / jnp.sqrt(S)) @ alpha_tilde_QR
    #jax.debug.print("alpha: {}", alpha)

    alpha_reshaped = alpha.reshape(K,T,d)                           #(K, T, D)
    mean = jnp.mean(alpha_reshaped, axis=(0), keepdims=True)        #(1, T, D)
    alpha_H = (alpha_reshaped - mean).reshape(K*T,d)                #(K*T,D)
    #jax.debug.print("alpha_H: {}", alpha_H)

    num_pairs = 100  
    indices = random.randint(key, shape=(num_pairs*2,), minval=0, maxval=K)  #--> this implementation works too but there was a bug (N instead of K)
    index_pairs = indices.reshape((num_pairs, 2))
    # all_combinations = jnp.array(list(combinations(range(K), 2)))
    # indices = random.randint(key, shape=(num_pairs,), minval=0, maxval=all_combinations.shape[0])
    # index_pairs = all_combinations[indices]

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