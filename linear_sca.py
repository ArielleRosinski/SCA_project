import numpy as np 
import jax
import jax.numpy as jnp
from jax import grad, random, vmap
import optax
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import wandb
from itertools import combinations

                   
def loss(params, X, key, s_learn, normalized = False):  
    K, N, T = X.shape

    if s_learn:
        U = params['U']
        s = params['s'] 

        s_normalized = jnp.sqrt(N) * (s**2) /  jnp.linalg.norm(s**2)
        X = s_normalized[None, :, None] * X
    else:
        U = params
    
    U_tilde, _ = jnp.linalg.qr(U)

    num_pairs = 100  
    indices = random.randint(key, shape=(num_pairs*2,), minval=0, maxval=K)
    index_pairs = indices.reshape((num_pairs, 2))
    # all_combinations = jnp.array(list(combinations(range(K), 2)))
    # indices = random.randint(key, shape=(num_pairs,), minval=0, maxval=all_combinations.shape[0])
    # index_pairs = all_combinations[indices]

    Y1 = jnp.einsum('ji,ljk->lik', U_tilde, X[index_pairs[:, 0], :, :])           #(pairs, d,T)
    Y2 = jnp.einsum('ji,ljk->lik', U_tilde, X[index_pairs[:, 1], :, :])           #(pairs, d,T)

    YY = jnp.einsum('lij,mkj->lmik', Y1, Y2)                                      #(pairs, pairs, d,d)

    term2 = jnp.einsum('klnm,klmn->', YY,YY)
    term1 = jnp.square(jnp.einsum('klnn->kl', YY)).sum()

    if normalized == False:
        S = (2 / (num_pairs**2) ) * (term1-term2)
        return -S
    
    else:
        return (term1-term2) / (term1+term2)

def update(params, X, optimizer, opt_state, key, s_learn):
    grad_ = grad(loss)(params,X,key, s_learn)
  
    updates, opt_state_updated = optimizer.update(grad_, opt_state, params)
    params_updated = optax.apply_updates(params, updates)
    return params_updated, opt_state_updated

def optimize(X, s_learn=False, iterations=10000, learning_rate=0.001, d=3, seed=42):
    K, N, T = X.shape

    key = random.PRNGKey(seed)
    
    U = random.normal(key, (N, d))
    s = random.normal(key, (N,)) 

    if s_learn:
        params = {
            'U': U,
            's': s
        }
    else: 
        params = U 

    keys = random.split(key, num=iterations)

    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)
    
    ls_loss = []
    ls_S_ratio = []

    for i in range(iterations):
        params, opt_state = update(params, X, optimizer, opt_state, keys[i], s_learn)
        
        loss_ = loss(params, X, keys[i], s_learn)
        S_ratio = loss(params, X, keys[i], s_learn, normalized = True)

        ls_loss.append(loss_)
        ls_S_ratio.append(S_ratio)

        wandb.log({"loss_": loss_, "S_ratio": S_ratio})
        if i % 10 == 0:
            print(f"Iteration {i}, S: {-loss_}, S_ratio: {S_ratio}")
    
    
    return params, ls_loss, ls_S_ratio


