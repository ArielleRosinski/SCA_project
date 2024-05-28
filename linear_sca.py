import numpy as np 
import jax
import jax.numpy as jnp
from jax import grad, random, vmap
import optax
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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


def single_pair_S(X, id_1, id_2, operator):
    XX = jnp.einsum('ij,kj->ik', X[id_1, :, :], X[id_2, :, :])          #(N,N)
    XX_product = jnp.einsum('ij,lm->im', XX, XX)                        #(N,N)

    if operator == 'minus':
        return jnp.trace(XX)**2 - jnp.trace(XX_product)
    
    elif operator == 'plus':
        return jnp.trace(XX)**2 + jnp.trace(XX_product)


def compute_S(X, seed=42, iterations=100):
    K, N, T = X.shape
    key = random.PRNGKey(seed)
    keys = random.split(key, num=iterations)

    S_list = []
    for i in range(iterations):
        num_pairs = 100  
        indices = random.randint(keys[i], shape=(num_pairs*2,), minval=0, maxval=N)
        index_pairs = indices.reshape((num_pairs, 2))

        batched_numerator = vmap(single_pair_S, in_axes=(None, 0, 0, None))(X, index_pairs[:, 0], index_pairs[:, 1], 'minus')
        batched_denominator = vmap(single_pair_S, in_axes=(None, 0, 0, None))(X, index_pairs[:, 0], index_pairs[:, 1], 'plus') 
        S_list.append( jnp.sum(batched_numerator) / jnp.sum(batched_denominator) )
    return S_list


                   
def single_pair_loss(U_tilde, X, id_1, id_2, operator = 'minus'):                           #U (N,d); X(K,N,T)

    Y = jnp.einsum('ji,jk->ik', U_tilde, X[id_1, :, :])                 #(d,T)
    Y_prime = jnp.einsum('ji,jk->ik', U_tilde, X[id_2, :, :])           #(d,T)

    YY = jnp.einsum('ij,kj->ik', Y, Y_prime)                            #(d,d)
    YY_product = jnp.einsum('ij,lm->im', YY, YY)                        #(d,d)

    if operator == 'minus':
        return jnp.trace(YY)**2 - jnp.trace(YY_product)
    
    elif operator == 'plus':
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
    
    U_tilde, _ = jnp.linalg.qr(U)

    num_pairs = 10  
    indices = random.randint(key, shape=(num_pairs*2,), minval=0, maxval=N)
    index_pairs = indices.reshape((num_pairs, 2))

    batched_loss = vmap(single_pair_loss, in_axes=(None, None, 0, 0))(U_tilde, X, index_pairs[:, 0], index_pairs[:, 1]) #(num_pairs)
    
    if normalized == False:
        S = (2 / (num_pairs**2) ) * jnp.sum(batched_loss)
        return -S
    else: 
        batched_normalizer = vmap(single_pair_loss, in_axes=(None, None, 0, 0, None))(U_tilde, X, index_pairs[:, 0], index_pairs[:, 1], 'plus')
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

def var_explained(X, U):
    """ X is K, N, T 
        U is N, d """
    X_reshaped = np.concatenate(X.swapaxes(1,2))    #(K*T, N)                #X_reshaped -= np.mean(X_reshaped, axis = 0) sigma = np.dot(X_reshaped.T, X_reshaped)
    sigma = np.cov(X_reshaped.T)
    return np.trace(U.T @ sigma @ U) / np.trace(sigma)

def plot_2D(Y):
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111)
    cmap = plt.get_cmap('viridis')      #RdBu

    num_time_points = Y.shape[-1]
    indices_to_plot = np.arange(0, Y.shape[0], 1)

    for i in indices_to_plot:
        x = Y[i, 0, :]  
        y = Y[i, 1, :]  
        
        for t in range(num_time_points - 1):
            ax.plot(x[t:t+2], y[t:t+2], color=cmap(t / (num_time_points - 1)))

    #ax.set_xlabel('SC 1')
    #ax.set_ylabel('SC 2')
    ax.spines[['top','right']].set_visible(False)
    #ax.set_xticks([])
    #ax.set_yticks([])
    #plt.grid()

def plot_3D(Y):
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection='3d')
    cmap = plt.get_cmap('viridis')

    num_time_points = Y.shape[-1]
    indices_to_plot = np.arange(0,Y.shape[0],1)

    for i in indices_to_plot:
        x = Y[i, 0, :]  
        y = Y[i, 1, :] 
        z = Y[i, 2, :]  

        
        for t in range(num_time_points - 1):
            ax.plot(x[t:t+2], y[t:t+2], z[t:t+2], color=cmap(t / (num_time_points - 1)))
    
    ax.spines[['top','right']].set_visible(False)

