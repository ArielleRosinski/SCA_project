import numpy as np 
from numpy.linalg import qr, svd
import jax
import jax.numpy as jnp
from jax import grad, random, vmap
from jax.scipy.linalg import solve_triangular
import optax
from sklearn.decomposition import PCA
from sklearn.linear_model import ElasticNetCV, LinearRegression, RidgeCV
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import wandb
from itertools import combinations
from scipy.integrate import solve_ivp
from scipy.ndimage import gaussian_filter

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
        elif soft_normalize == 'max':
            norm_const = jnp.maximum(jnp.max(X, axis=(0,2),  keepdims=True ), 0.1)
            X = X / norm_const
                
        if center:
            condition_mean = jnp.mean(X, axis=0, keepdims=True)
            X = X - condition_mean

        pca_variance_captured = None

        if pca:
            X_reshaped = X.swapaxes(1,2).reshape(-1, N)
            pca = PCA(num_pcs)
            X = pca.fit_transform(X_reshaped)
            X = X.reshape(K, T, num_pcs).swapaxes(1,2)
            pca_variance_captured = pca.explained_variance_

        return jnp.array(X), pca_variance_captured

def center(x, axis=0):
    mean_x = jnp.mean(x, axis=(axis), keepdims=True)        
    return (x - mean_x)

def var_explained(X, U):
    """ X is K, N, T 
        U is N, d """
    _, N, _ = X.shape
    X_reshaped = X.swapaxes(1,2).reshape(-1, N)    #(K*T, N)                #X_reshaped -= np.mean(X_reshaped, axis = 0) sigma = np.dot(X_reshaped.T, X_reshaped)
    sigma = np.cov(X_reshaped.T)
    return np.trace(U.T @ sigma @ U) / np.trace(sigma)

# def var_explained_kernel(alpha, kernel_function, A, X, l2, scale):
#     K, _, T = X.shape

#     K_A_A = kernel_function(A, A, l2=l2, scale=scale)
#     K_A_A_reshaped = K_A_A.reshape(K, T, K, T)
#     mean = jnp.mean(K_A_A_reshaped, axis=(0,2), keepdims=True)   
#     K_A_A_tilde = (K_A_A_reshaped - mean).reshape(K*T, K*T)       

#     var_explained = jnp.trace(alpha.T @ K_A_A_tilde @ K_A_A_tilde @ alpha) / jnp.trace(K_A_A_tilde)
#     return var_explained

def H_mult(K_X_Y, K, T, K_prime):
    K_X_Y_reshaped = K_X_Y.reshape(K, T, K_prime, T)
    mean = jnp.mean(K_X_Y_reshaped, axis=(0,2), keepdims=True)   
    K_X_Y_tilde = (K_X_Y_reshaped - mean).reshape(K*T, K_prime*T)  
        
    return K_X_Y_tilde

def squared_frobenius_norm(matrix):
    return jnp.sum(matrix**2)

def get_numerator(K, X, A_train, X_train, T, l2, scale, alpha, kernel_function, batch_size=10):
    K_train, N, T = X_train.shape
    numerator = 0

    for start in range(0, K, batch_size):
        end = min(start + batch_size, K)
       
        Y = jnp.swapaxes(X[start:end, :, :], 0, 1).reshape(N, -1)           ##for k in range(K): Y = jnp.swapaxes(X[k,:,:][None,:,:], 0, 1).reshape(N,-1)      
        K_X_Y = kernel_function(A_train, Y, l2=l2, scale=scale)      
  
        K_X_Y_tilde = H_mult(K_X_Y, K_train, T, end-start )
      
        numerator += squared_frobenius_norm(alpha.T @ K_X_Y_tilde)  
    return numerator    

def get_denominator(A, X, K, T, l2, scale, kernel_function):
    _, N, T = X.shape
    diag_entries = jnp.array([kernel_function(A[:,i][:,None], A[:,i][:,None], l2=l2, scale=scale) for i in range(A.shape[-1])]).reshape(K, T)

    mean = jnp.zeros((T, T))

    for t in range(T):
        for t_prime in range(T):
            A_t = jnp.swapaxes(X[:,:,t][:,:,None], 0, 1).reshape(N,-1)     
            A_t_prime = jnp.swapaxes(X[:,:,t_prime][:,:,None], 0, 1).reshape(N,-1)      

            kernel_values = kernel_function(A_t, A_t_prime, l2=l2, scale=scale)

            mean = mean.at[t, t_prime].set(jnp.sum(kernel_values) / (K * K))

    centered_diag_entries = diag_entries - jnp.diag(mean)
    return jnp.sum(centered_diag_entries)

def var_explained_kernel(alpha, kernel_function, A_train, X_train, l2, scale, A_test=None, X_test = None, test=False, compute_matrix = False):
    K, _, T = X_train.shape

    if test == False:
        if compute_matrix:
            K_X_X = kernel_function(A_train, A_train, l2=l2, scale=scale)
            K_X_X_tilde = H_mult(K_X_X, K, T, K)
            numerator = jnp.trace(alpha.T @ K_X_X_tilde @ K_X_X_tilde @ alpha)

            denominator = jnp.trace(K_X_X_tilde)
        else: 
            numerator = get_numerator(K, X_train, A_train, X_train, T, l2, scale, alpha, kernel_function)
            denominator = get_denominator(A_train, X_train, K, T,  l2, scale, kernel_function)

    else:
        K_prime, _, _ = X_test.shape
        if compute_matrix:
            K_X_Y = kernel_function(A_train, A_test, l2=l2, scale=scale)
            K_X_Y_tilde = H_mult(K_X_Y, K, T, K_prime)
            numerator = jnp.trace(alpha.T @ K_X_Y_tilde @ K_X_Y_tilde.T @ alpha)

            K_Y_Y = kernel_function(A_test, A_test, l2=l2, scale=scale)
            K_Y_Y_tilde = H_mult(K_Y_Y, K_prime, T, K_prime)
            denominator = jnp.trace(K_Y_Y_tilde)
        
        else:
            numerator = get_numerator(K_prime, X_test, A_train, X_train, T, l2, scale, alpha, kernel_function)

            denominator = get_denominator(A_test, X_test, K_prime, T,  l2, scale,kernel_function)


    var_explained = numerator / denominator

    return var_explained

def autocorr(x, lags):
    corr = []
    for l in lags:
        if l == 0:
            corr_coef = np.corrcoef(x, x)[0, 1]
        else:
            corr_coef = np.corrcoef(x[:-l], x[l:])[0, 1]
        corr.append(corr_coef)
    return np.array(corr)

def get_pca(X_train, X_test=None, num_pcs = 2, test=False):
    _, N, T = X_train.shape
    X_pca_train = center(X_train).swapaxes(1,2).reshape(-1, N)

    
    pca = PCA(num_pcs)
    if test:
        X_pca_test = center(X_test).swapaxes(1,2).reshape(-1, N)
        Y_pca = pca.fit(X_pca_train).transform(X_pca_test)
    else:
        Y_pca = pca.fit(X_pca_train).transform(X_pca_train)
    PCs = pca.components_
    Y_pca = Y_pca.reshape(-1, T, num_pcs).swapaxes(1,2)
    return Y_pca, PCs

def single_pair_S(X, id_1, id_2, operator):
    XX = jnp.einsum('ij,kj->ik', X[id_1, :, :], X[id_2, :, :])          #(N,N)
    XX_product = XX @ XX

    if operator == 'minus':
        return jnp.trace(XX)**2 - jnp.trace(XX_product)
    
    elif operator == 'plus':
        return jnp.trace(XX)**2 + jnp.trace(XX_product)


def compute_S(X, seed=42, iterations=1000, num_pairs = 100, ratio=True):
    K, N, T = X.shape
    key = random.PRNGKey(seed)
    keys = random.split(key, num=iterations)

    S_list = []
    for i in range(iterations):
        indices = random.randint(keys[i], shape=(num_pairs*2,), minval=0, maxval=K)
        index_pairs = indices.reshape((num_pairs, 2))
        # all_combinations = jnp.array(list(combinations(range(K), 2)))
        # indices = random.randint(keys[i], shape=(num_pairs,), minval=0, maxval=all_combinations.shape[0])
        # index_pairs = all_combinations[indices]

        batched_numerator = vmap(single_pair_S, in_axes=(None, 0, 0, None))(X, index_pairs[:, 0], index_pairs[:, 1], 'minus')
        if ratio:
            batched_denominator = vmap(single_pair_S, in_axes=(None, 0, 0, None))(X, index_pairs[:, 0], index_pairs[:, 1], 'plus') 
            S_list.append( jnp.sum(batched_numerator) / jnp.sum(batched_denominator) )
        else: 
            S_list.append( (2 / (num_pairs**2) ) * jnp.sum(batched_numerator) )

    return S_list

def compute_S_all_pairs(X):
    K, _, _ = X.shape
    index_pairs = jnp.array(list(combinations(range(K), 2)))

    batched_numerator = vmap(single_pair_S, in_axes=(None, 0, 0, None))(X, index_pairs[:, 0], index_pairs[:, 1], 'minus')
    batched_denominator = vmap(single_pair_S, in_axes=(None, 0, 0, None))(X, index_pairs[:, 0], index_pairs[:, 1], 'plus') 

    return jnp.sum(batched_numerator) / jnp.sum(batched_denominator) 

def get_reg(X_train,y_train,X_test, y_test):
    regr = RidgeCV()
    reg = regr.fit(X_train, y_train)   
    return reg.score(X_test, y_test), reg.predict(X_test)   

def principal_angle(Y, X_pca, d, smaller=True):
    Q_A, _ = qr(Y.swapaxes(1,2).reshape(-1,d), mode='reduced')
    Q_B, _ = qr(X_pca.swapaxes(1,2).reshape(-1,d), mode='reduced')
        
    M = np.dot(Q_A.T, Q_B)
        
    _, singular_values, _ = svd(M)
    
    angles = np.arccos(singular_values)
    angles = np.rad2deg(np.sort(angles)[::-1])
    
    if smaller:
        return angles[-1]
    else:
        return angles
    
def get_loss_fig(ls_loss, ls_S_ratio):
    plt.figure(figsize=(10,5))
    plt.subplot(211)
    plt.plot(ls_loss)
    plt.grid()
    plt.title(r"$-S(U) = -\|C^{(-)}\|_\mathrm{F}^2$")
    plt.gca().spines[['top','right']].set_visible(False)
    plt.subplot(212)
    plt.plot(ls_S_ratio)
    plt.title(r"$\frac{\|C^{(-)}\|_\mathrm{F}^2}{\|C^{(+)}\|_\mathrm{F}^2}$")
    plt.gca().spines[['top','right']].set_visible(False)
    plt.grid()
    plt.subplots_adjust(hspace=0.5)

def plot_1D(X):
    K, _, T = X.shape
    cmap = plt.cm.viridis  
    fig, ax = plt.subplots() 

    for k in range(K):
        for t in range(T - 1):
            ax.plot([t, t + 1], [X[k, 0, t], X[k, 0, t + 1]], color=cmap(t / (T - 1)))

    ax.spines[['top','right']].set_visible(False)

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

    ax.spines[['top','right']].set_visible(False)

def plot_2D_K_coded(Y):
    K, _,_=Y.shape
    fig = plt.figure()
    ax = fig.add_subplot(111) #projection='3d'
    cmap = plt.get_cmap('coolwarm', K)
    for k in range(K):
        x = Y[k, 0, :]
        y = Y[k, 1, :]
        
        #z = Y[k, 2, :] 
        #color = cmap(k / K)
        #ax.plot(x, y, z, linestyle='-', marker='.', linewidth=1, color=color)
        color = cmap(k / (K - 1)) 
        ax.plot(x, y, linestyle='-', marker='.', linewidth=1, color=color)
    plt.title(f'kSCA; s = {compute_S_all_pairs(Y)}')
    
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


def plot_3D_K_coded(Y, elevation=30, azimuth=30, rotate=False):
    K, _,_=Y.shape
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, projection='3d') 
    cmap = plt.get_cmap('coolwarm', K)
    for k in range(K):
        x = Y[k, 0, :]
        y = Y[k, 1, :]
        z = Y[k, 2, :] 
       

        color = cmap(k / (K - 1)) 
        ax.plot(x, y, z, linestyle='-', marker='.', linewidth=1, color=color)

        if rotate:
            ax.view_init(elev=elevation, azim=azimuth)

    ax.set_xlabel('Dim. 1')
    ax.set_ylabel('Dim. 2')
    ax.set_zlabel('Dim. 3')
    ax.set_box_aspect(aspect=None, zoom=0.85)
    plt.title(f's = {compute_S_all_pairs(Y)}')
    
def apply_gaussian_smoothing(data, sigma=1, axes=-1):
    smoothed_data = gaussian_filter(np.array(data), sigma=sigma, axes=axes)
    return jnp.array(smoothed_data)