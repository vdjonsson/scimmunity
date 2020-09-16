import os
import operator

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc

from sklearn.linear_model import LinearRegression

def calculate_complexity(adata):
    """
    adapted from SEQC
    """
    data = adata.obs
    x = np.log(data['n_counts']).values.reshape(-1,1)
    y = np.log(data['n_genes']).values.reshape(-1,1)
    regr = LinearRegression()
    regr.fit(x, y)
    # mark large residuals as failing
    yhat = regr.predict(x)
    residuals = yhat - y
    
    mean_res = np.mean(residuals)
    std_res = np.std(residuals)

    failing = residuals > (mean_res + 3*(std_res))

    return x, y, yhat, residuals, failing

def plot_complexity(adata, sample_name, outdir):

    """
    adapted from SEQC
    """
    x, y, yhat, residuals, failing = calculate_complexity(adata)
    g = sns.jointplot(x=x, y=y, kind="kde")
    g.ax_joint.plot(x, yhat, '-', color='r')
    g.ax_joint.scatter(x=x, y=yhat, color='white', marker='+', s=3)
    g.ax_joint.scatter(x[failing], y[failing], c='indianred', s=3)

    g.ax_joint.set_xlabel('log_n_counts')
    g.ax_joint.set_ylabel('log_n_genes')
    plt.tight_layout()
    path = os.path.join(outdir, f'{sample_name}_complexity.png')
    plt.savefig(path)
    plt.close()
    return

def preprocess_adata(adata):
    # basic filtering
    # get rid of cells with 0 genes
    sc.pp.filter_cells(adata, min_genes=1)
    # calculate n_cells for each gene but no gene filter yet
    sc.pp.filter_genes(adata, min_cells=0)

    # calculate QC metrics
    adata.obs['n_counts'] = adata.X.sum(axis=1).A1
    mito_genes = [name for name in adata.var_names if name.startswith('MT-')]
    # for each cell compute fraction of counts in mito genes vs. all genes
    adata.obs['percent_mito'] = np.sum(adata[:, mito_genes].X, axis=1).A1 \
        / np.sum(adata.X, axis=1).A1 * 100
    # calculate complexity (i.e. geen abundance)
    _, _, _, residuals, failing = calculate_complexity(adata)
    adata.obs['complexity_residuals'] = residuals
    adata.obs['complexity_failing'] = failing

    return adata

def plot_mt(adata, sample_name, outdir):
    data = adata.obs
    sns.jointplot(x='n_counts', y='percent_mito', data=data, kind="kde")
    path = os.path.join(outdir, f'{sample_name}_mt.png')
    plt.savefig(path)
    plt.close()
    return

def ecdf(data):
    x = np.sort(data)
    y = np.arange(1, len(x)+1) / len(x)
    return x, y

def plot_thresh(adata, sample_name, feature, thresh, sign, kind, outdir, 
    xlim=None, axs=None, log=False, logfile=False, save=True):
    
    ops = {'>': operator.gt, '<':operator.lt}

    if kind == 'obs':
        values = adata.obs[feature]
        unit = 'cells'
        
    if kind == 'var':
        values = adata.var[feature]
        unit = 'genes'

    if type(axs)==type(None):
        fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    sns.distplot(values, ax=axs[0], bins=100, hist_kws={'range':xlim})
    axs[0].axvline(x=thresh, color='r')
    x, y = ecdf(values)
    axs[1].plot(x, y, '.', linestyle='none', alpha=0.6)
    axs[1].axvline(x=thresh, color='r')
    axs[1].set(xlabel=feature, ylabel='ECDF')
    axs[1].legend(['unfiltered','threshold', 'filtered'])
    if xlim :
        axs[0].set_xlim(xlim)
        axs[1].set_xlim(xlim)
    plt.tight_layout()
    if save:
        path = os.path.join(outdir, f'{sample_name}_{feature}_{thresh}.png')
        plt.savefig(path, bbox_inches="tight")
        plt.close()
    if log:
        logfile.write(f'{feature} threshold = {thresh}\n')
        count = sum(ops[sign](values, thresh))
        logfile.write('Number of {} {} {} = {} ({:.2f}%)\n'.format(
            unit, sign, feature, count, count/len(values)*100) )
    return

def filter_adata(adata, sample_name, outdir, logfile,
    min_counts=1000, min_genes=300, 
    min_cells=0, max_mito=20):

    print('-'*50, file=logfile)
    print(sample_name, file=logfile)
    print('Before: {} cells {} genes\n'.format(*adata.shape), file=logfile)
    
    res = adata.obs['complexity_residuals'].values
    # 3 std dev below mean
    max_residual = np.mean(res) + np.std(res) * 3

    figs = []
    axes = []
    for i in range(7):
        fig, axs = plt.subplots(1, 2, figsize=(12, 4))
        figs.append(fig)
        axes.append(axs)

    # feature, thresh, sign, kind, xlim
    filters = [
        ('percent_mito', max_mito, '>', 'obs', None),
        ('n_counts', min_counts, '<', 'obs', None),
        ('n_genes', min_genes, '<', 'obs', None),
        ('n_cells', min_cells, '<', 'var', (0,100)),
        ('complexity_residuals', max_residual, '>', 'obs', None),
        # zoomed in
        ('n_counts', min_counts, '<', 'obs', (0, 5000)),
        ('n_genes', min_genes, '<', 'obs', (0, 1000))
    ]

    # plot distribution before filtering
    for i, args in enumerate(filters):
        feature, thresh, sign, kind, xlim = args 
        plot_thresh(adata, sample_name, feature, thresh, sign, kind, outdir, 
            xlim=xlim, axs=axes[i], log=True, logfile=logfile, save=False)

    sc.pp.filter_cells(adata, min_genes=min_genes)
    sc.pp.filter_cells(adata, min_counts=min_counts)
    sc.pp.filter_genes(adata, min_cells=min_cells)
    adata = adata[adata.obs['percent_mito'] < max_mito, :]
    adata = adata[adata.obs['complexity_residuals'] < max_residual, :]
    
    # plot distribution after filtering
    for i, args in enumerate(filters):
        feature, thresh, sign, kind, xlim = args 
        plot_thresh(adata, sample_name, feature, thresh, sign, kind, outdir, 
            xlim=xlim, axs=axes[i], log=False, logfile=None, save=False)
        path = os.path.join(outdir, f'{sample_name}_{feature}_{thresh:.2f}.png')
        figs[i].savefig(path, bbox_inches='tight')

    print('\nAfter: {} cells {} genes'.format(*adata.shape), file=logfile)
    print('-'*50, file=logfile)
    return adata

def filter_adatas(adatas, sample_names, outdir, 
    min_counts=1000, min_genes=300, min_cells=0, max_mito=20):
    """
    Args:
        min_counts (int or list of int): cell filter n_counts (umi) threshold
        min_genes (int or list of int): cell filter n_genes threshold
        min_cells (int or list of int): gene filter n_cells threshold
        max_mito (inr or list of int): cell filter percent_mito threshold
    """

    # make filter log file
    logfile = open(os.path.join(outdir, 'filter.log'), 'w')

    # convert int to list of int
    if type(min_counts) != list:
        min_counts = [min_counts]*len(adatas)
    if type(min_genes) != list:
        min_genes = [min_genes]*len(adatas)
    if type(min_cells) != list:
        min_cells = [min_cells]*len(adatas)
    if type(max_mito) != list:
        max_mito = [max_mito]*len(adatas)

    adatas_filtered = []
    for adata, sample_name, min_counts_i, min_genes_i, min_cells_i, max_mito_i, \
        in zip(adatas, sample_names, min_counts, min_genes, min_cells, max_mito):
        plot_mt(adata, sample_name, outdir)
        plot_complexity(adata, sample_name, outdir)
        # filter edach adata
        adata_filtered = filter_adata(adata, sample_name, outdir, logfile, 
            min_counts=min_counts_i, min_genes=min_genes_i, 
            min_cells=min_cells_i, max_mito=max_mito_i)
        adatas_filtered.append(adata_filtered)
    logfile.close()
    return adatas_filtered

def merge_adatas(adatas, outdir, log=True):
    adata = adatas[0].concatenate(adatas[1:], join='inner', index_unique=None, \
        batch_key='batch_indices')
    # adata.obs['sample_name'] = [sample_names[int(i)] \
    #     for i in adata.obs['batch_indices']]
    # for key in self.metadata:
    #     adata.obs[key] = [self.metadata[key][int(i)] for i in adata.obs['batch_indices']]
    if log:
        # append to filter log file
        logfile = open(os.path.join(outdir, 'filter.log'), 'a')
        print('Merged adata: {} cells {} genes'.format(*adata.shape), file=logfile)

        # get rid of genes expressed in less than 2 cells
        sc.pp.filter_genes(adata, min_cells=2)

        print('Remove genes expressed in less than 2 cells', file=logfile)
        print('Merged adata: {} cells {} genes'.format(*adata.shape), file=logfile)
        logfile.close()
    return adata

def filter_and_merge(adatas, sample_names, outdir, 
    subsample=False, n_obs=1000, 
    min_counts=1000, min_genes=300, min_cells=0, max_mito=20):
    """
    outdir (str): path to output folder for filtering logfile and plots
    subsample (bool): If True, subsample per sample
    n_obs (int): number of cells per sample to be subsampled
    """
    # preproccess each adata
    adatas = [preprocess_adata(adata) for adata in adatas]

    # filter each adata
    adatas = filter_adatas(adatas, sample_names, outdir,
        min_counts=min_counts, min_genes=min_genes, 
        min_cells=min_cells, max_mito=max_mito)

    # subsample each adata
    if subsample:
        for adata in adatas:
            sc.pp.subsample(adata, n_obs=min(adata.shape[0], n_obs))
            # append to filter log file
            logfile = open(os.path.join(outdir, 'filter.log'), 'a')
            print('subsampled adata: {} cells {} genes'.format(*adata.shape), file=logfile)
            logfile.close()
        
    # merge adatas
    adata = merge_adatas(adatas, outdir)

    return adata