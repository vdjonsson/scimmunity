import os
import logging
import numpy as np
import matplotlib.pyplot as plt

# calinski_harabasz_score for sklearn version higher than v0.19
from sklearn.metrics import calinski_harabaz_score 
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.metrics import pairwise_distances

# https://stackoverflow.com/questions/15450192/fastest-way-to-compute-entropy-in-python
def shannon_entropy(labels, base=None):
    """ Computes entropy of label distribution. """

    n_labels = len(labels)

    if n_labels <= 1:
        return 0

    value, counts = np.unique(labels, return_counts=True)
    probs = counts / n_labels
    n_classes = np.count_nonzero(probs)

    if n_classes <= 1:
        return 0

    ent = 0.

    # Compute entropy
    base = e if base is None else base
    for i in probs:
        ent -= i * log(i, base)

    return ent

def entropy_mixing(adata, label='sample_name', layer='X', key='neighbors', \
    key_added='entropy_mixing', inplace=True):
    """
    Inspired by Azizi et al., 2018
    Use precomputed neighbors connectivies.
    compute the distribution of labels in the neighborhood of each cell j
    compute Shannon entropy per cell
    """
    knn = (adata.uns[key]['connectivities'] > 0).toarray()
    labels = adata.obs[label].values
    H = []
    
    for i in range(adata.shape[0]):
        knn_labels = labels[knn[i, :]]
        ent = shannon_entropy(knn_labels)
        H.append(ent)
    
    H = np.array(H)
    if inplace:
        adata.obs[key_added] = H 
        return
    else:
        return H

def plot_variance_ratio(adata, res_list, X='latent', out='./clustering/', 
    prefix='', rep='latent', save=True):
    """
    res_list (list of float): list of resolution
    X (str): representation or layer to use {'latent', 'X', 'raw'}
    """
    if 'X_{}'.format(X) in adata.obsm:
        data = adata.obsm['X_{}'.format(X)]
    elif X=='X':
        data = adata.X
    else:
        data = adata.layers[X]

    fig, ax = plt.subplots()
    for method in ['Louvain', 'Leiden']:
        keys = []
        resolution = []
        for res in res_list:
            key = prefix+'{}Res{}_{}'.format(method, res, rep)
            # include resolution with more than one cluster
            if len(adata.obs[key].cat.categories) > 1:
                keys.append(key)
                resolution.append(res)
        scores = [calinski_harabaz_score(data, adata.obs[key].values) 
            for key in keys]
        ax.plot(resolution, scores, label=method)
    ax.legend()
    ax.set_ylabel('Variance Ratio Criterion')
    ax.set_xlabel('Resolution')
    ax.set_title(X)
    if save:
        fig.savefig(os.path.join(out, f'variance_ratio_criterion_{X}.png'))
        plt.close()
    else:
        plt.show()
    return 

def plot_silhouette_coeff(adata, res_list, X='latent', out='./clustering/', 
    prefix='', rep='latent', save=True, precompute=True):
    """
    res_list (list of float): list of resolution
    X (str): representation or layer to use {'latent', 'X', 'raw'}
    """
    if 'X_{}'.format(X) in adata.obsm:
        data = adata.obsm['X_{}'.format(X)]
    elif X=='X':
        data = adata.X
    else:
        data = adata.layers[X]

    if precompute:
        logging.warning('Calculating pairwise distance')
        data = pairwise_distances(data)
        logging.warning('Finished calculating pairwise distance')
        metric = 'precomputed'
    else:
        metric = 'euclidean'
    
    plot_mean_silhouette_coeff(adata, data, metric, res_list, 
        out=out, prefix=prefix, rep=rep, save=save)

    for method in ['Louvain', 'Leiden']:
        keys = [prefix+'{}Res{}_{}'.format(method, res, rep) for res in res_list]
        keys = [key for key in keys if len(adata.obs[key].cat.categories) > 1]
        for key in keys:
            silhouette_per_cluster(adata, data, key, metric, 
                out=out, save=save, figsize=(6,6))
            
    return 

def plot_mean_silhouette_coeff(adata, data, metric, res_list, 
    out='./clustering/', prefix='', rep='latent', save=True):

    fig, ax = plt.subplots()
    for method in ['Louvain', 'Leiden']:
        keys = []
        resolution = []
        for res in res_list:
            key = prefix+'{}Res{}_{}'.format(method, res, rep)
            # include resolution with more than one cluster
            if len(adata.obs[key].cat.categories) > 1:
                keys.append(key)
                resolution.append(res)
        scores = [silhouette_score(data, adata.obs[key].values, 
            metric=metric) for key in keys]
        ax.plot(resolution, scores, label=method)
    ax.legend()
    ax.set_ylabel('Mean Silhouette Coefficient')
    ax.set_xlabel('Resolution')
    if save:
        fig.savefig(os.path.join(out, 'mean_silhouette_coeff.png'))
        plt.close()
    else:
        plt.show()
    return 

def silhouette_per_cluster(adata, data, key, metric, out='./', save=True, figsize=(6,6)):
    # https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html
    
    cluster_labels = adata.obs[key].values
    colors = adata.uns[key+'_colors']
    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(data, cluster_labels, metric=metric)
    n_clusters = len(cluster_labels.unique())
    y_lower=10

    fig, ax1 = plt.subplots(figsize=figsize)

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-0.1, 1]
    ax1.set_xlim([-0.1, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, len(data) + (n_clusters + 1) * 10])

    # silhouette_avg = silhouette_score(data, cluster_labels)
    silhouette_avg = np.mean(sample_silhouette_values)
    
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = \
            sample_silhouette_values[cluster_labels == str(i)]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = colors[i]
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                        0, ith_cluster_silhouette_values,
                        facecolor=color, edgecolor=color, alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
    ax1.set_title('{}'.format(key))
    ax1.set_xlabel("Silhouette Cefficient")
    ax1.set_ylabel("Cluster")
    if save:
        fig.savefig(os.path.join(out, f'silhouette_coeff_{key}.png'))
        plt.close()
    else:
        plt.show()
    return 