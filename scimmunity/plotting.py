import os
import scanpy as sc
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rcParams

from scanpy.plotting._tools.scatterplots import plot_scatter

import scimmunity.palette as palettes 
from scimmunity.utils import clean_up_str

def set_plotting_params(dpi=300):
    sc.settings.set_figure_params(dpi=dpi, dpi_save=dpi, format='png', 
        color_map=palettes.gene_cmap, frameon=False, transparent=True)
    sns.set_context('paper')
    sns.set_palette(palettes.cluster_palette)
    return

def plot_reps(adata, color, save_name=None, outdir='./', 
    reps=['tsne', 'umap', 'diffmap'], 
    use_raw=False, layer=None, figsize=None, **kwargs):
    temp = sc.settings.figdir
    figsize_temp = rcParams['figure.figsize']
    if figsize is not None:
        rcParams['figure.figsize'] = figsize
    if save_name is None:
        save_name=color
    # Clean up save_name
    save_name = clean_up_str(save_name)
    if layer is not None:
        save_name += '_'+layer
    for rep in reps:
        sc.settings.figdir = os.path.join(outdir, rep)
        plot_scatter(adata, basis=rep, color=color, 
            save='_'+save_name+'_raw'*use_raw, 
            layer=layer, use_raw=use_raw, **kwargs)
    # restore default plot settings
    sc.settings.figdir = temp
    rcParams['figure.figsize'] = figsize_temp
    return 

def plot_bool(adata, key, basis, groups, order=True, save_name=None, title=None, out='./'):
    temp = sc.settings.figdir
    sc.settings.figdir = out

    if save_name is None:
        save_name=clean_up_str(key)

    ax = plot_scatter(adata, basis=basis, color=None, show=False)
    markersize = ax.collections[0]._sizes[0]
    if order:
        for i, group in enumerate(groups):
            if i!=(len(groups)-1):
                inds = adata.obs[key]==group
                ax = plot_scatter(adata[inds], basis=basis, color=key, \
                    ax=ax, size=markersize, title=title, show=False)
            else:
                inds = adata.obs[key]==group
                ax = plot_scatter(adata[inds], basis=basis, color=key, \
                    ax=ax, size=markersize, save='_'+save_name, title=title)
    else:
        inds = adata.obs[key].isin(groups)
        ax = plot_scatter(adata[inds], basis=basis, color=key, \
            ax=ax, size=markersize, save='_'+save_name, title=title)
    sc.settings.figdir = temp
    plt.close()
    return

def plot_reps_markers(adata, markers, save_name, outdir='./', 
    reps=['tsne', 'umap', 'diffmap'], use_raw=False, layer=None, figsize=(4,4)):
    # make sure markers are detected 
    markers = [g for g in markers if g in adata.var_names]
    plot_reps(adata, markers, save_name=save_name, outdir=outdir, reps=reps, 
        use_raw=use_raw, layer=layer, figsize=figsize)
    return 


def plot_density(adata, key, reps, components='1,2', outdir='./', size=20):
    temp = sc.settings.figdir
    for rep in reps:
        out = os.path.join(outdir,rep)
        mkdir(out)
        sc.settings.figdir = out
        # diffmap offset by 1 by default
        sc.tl.embedding_density(adata, basis=rep, groupby=key, components=components)
        groups =  adata.obs[key].cat.categories
        for group in groups:
            save_name = '{}_{}_{}'.format('_'.join(components.split(',')), key, group, )
            sc.pl.embedding_density(adata, basis=rep, key=rep+'_density_'+key, \
                group=group, show=False, frameon=True, save=save_name, \
                bg_dotsize=size, fg_dotsize=size)
    sc.settings.figdir = temp
    return    