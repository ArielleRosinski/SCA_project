import numpy as np 
from numpy.linalg import qr, svd
import jax
import jax.numpy as jnp
from jax import grad, random, vmap
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
    X_reshaped = np.concatenate(X.swapaxes(1,2))    #(K*T, N)                #X_reshaped -= np.mean(X_reshaped, axis = 0) sigma = np.dot(X_reshaped.T, X_reshaped)
    sigma = np.cov(X_reshaped.T)
    return np.trace(U.T @ sigma @ U) / np.trace(sigma)

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

def plot_3D_clean(Y, fontsize=13):
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection='3d')
    cmap = plt.get_cmap('Blues_r')

    num_time_points = Y.shape[-1]
    indices_to_plot = np.arange(0,Y.shape[0],1)

    for i in indices_to_plot:
        x = Y[i, 0, :]  
        y = Y[i, 1, :] 
        z = Y[i, 2, :]  
        
        for t in range(num_time_points - 1):
            ax.plot(x[t:t+2], y[t:t+2], z[t:t+2], color=cmap(t / (num_time_points - 1)), linewidth = 1)
            #ax.plot(x[t:t+2], y[t:t+2], color=cmap(t / (num_time_points - 1)), linewidth = 1)
    
    ax.spines[['top','right']].set_visible(False)
    ax.grid(False)  # Turn off the grid
    # ax.set_xticks([])  # Remove x-axis ticks
    # ax.set_yticks([])  # Remove y-axis ticks
    # ax.set_zticks([])  # Remove z-axis ticks

    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    # Removing the color of the panes (set to white to match most backgrounds)
    ax.xaxis.pane.set_edgecolor('white')
    ax.yaxis.pane.set_edgecolor('white')
    ax.zaxis.pane.set_edgecolor('white')

    ax.tick_params(axis='both', labelsize=fontsize)

    # ax.xaxis.line.set_linewidth(0)
    # ax.yaxis.line.set_linewidth(0)
    # ax.zaxis.line.set_linewidth(0)


def single_pair_S(X, id_1, id_2, operator):
    XX = jnp.einsum('ij,kj->ik', X[id_1, :, :], X[id_2, :, :])          #(N,N)
    #XX_product = jnp.einsum('ij,lm->im', XX, XX)                        #(N,N)
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

def apply_gaussian_smoothing(data, sigma=1, axes=-1):
    smoothed_data = gaussian_filter(np.array(data), sigma=sigma, axes=axes)
    return jnp.array(smoothed_data)