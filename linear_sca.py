import numpy as np 
import jax
import jax.numpy as jnp
from jax import grad, random, vmap
import optax
from sklearn.decomposition import PCA

def pre_processing(X,
               soft_normalize = 'churchland',
               soft_normalize_value = 5,
               center=True,
               pca=True,
               num_pcs=30):
        
        K, N, T = X.shape
        if soft_normalize == 'churchland': 
            """ soft-normalized to approximately unity firing rate range (divided by a normalization factor equal 
            to the firing rate range + 5 spikes per s) (Elsayed, 2017)"""
            range = jnp.max(X, axis=(0,2), keepdims=True) - jnp.min(X, axis=(0,2), keepdims=True)
            X = X / (range + soft_normalize_value)
        else:
            norm_const = jnp.maximum(jnp.max(X, axis=(0,2),  keepdims=True ), 0.1)
            X = X / norm_const
                
        if center:
            condition_mean = jnp.mean(X, axis=0, keepdims=True)
            X = X - condition_mean

        pca_variance_captured = None

        if pca:
            X_reshaped = np.concatenate(X.swapaxes(1,2))
            pca = PCA(num_pcs)
            X = pca.fit_transform(X_reshaped)
            X = X.reshape(K, T, num_pcs).swapaxes(1,2)
            pca_variance_captured = pca.explained_variance_

        return jnp.array(X), pca_variance_captured


                   
def single_pair_S(U_tilde, X, id_1, id_2, operator = 'plus'):                           #U (N,d); X(K,N,T)

    Y = jnp.einsum('ji,jk->ik', U_tilde, X[id_1, :, :])                 #(d,T)
    Y_prime = jnp.einsum('ji,jk->ik', U_tilde, X[id_2, :, :])           #(d,T)

    YY = jnp.einsum('ij,kj->ik', Y, Y_prime)                            #(d,d)
    YY_product = jnp.einsum('ij,lm->im', YY, YY)                        #(d,d)

    if operator == 'plus':
        return jnp.trace(YY)**2 - jnp.trace(YY_product)
    
    elif operator == 'minus':
        return jnp.trace(YY)**2 + jnp.trace(YY_product)
    
def loss(params, X, key, s_learn, normalized = False):  
    K, N, T = X.shape

    if s_learn:
        U = params['U']
        s = params['s'] 

        s_normalized = jnp.sqrt(N) * (s**2) /  jnp.linalg.norm(s**2)
        X = s_normalized[None, :, None] * X
    else:
        U = params
        X = X
    
    U_tilde, _ = jnp.linalg.qr(U)

    num_pairs = 10  
    indices = random.randint(key, shape=(num_pairs*2,), minval=0, maxval=N)
    index_pairs = indices.reshape((num_pairs, 2))

    batched_loss = vmap(single_pair_S, in_axes=(None, None, 0, 0))(U_tilde, X, index_pairs[:, 0], index_pairs[:, 1]) #(num_pairs)
    
    if normalized == False:
        S = (2 / (num_pairs**2) ) * jnp.sum(batched_loss)
        return -S
    else: 
        batched_normalizer = vmap(single_pair_S, in_axes=(None, None, 0, 0, None))(U_tilde, X, index_pairs[:, 0], index_pairs[:, 1], 'minus')
        return jnp.sum(batched_loss) / jnp.sum(batched_normalizer)

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
        
        if i % 10 == 0:
            print(f"Iteration {i}, S: {-loss_}, S_ratio: {S_ratio}")
    
    
    return params, ls_loss, ls_S_ratio
