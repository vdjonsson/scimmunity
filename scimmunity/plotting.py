import os
import scanpy as sc
import seaborn as sns

from scanpy.plotting._tools.scatterplots import plot_scatter

import scimmunity.palette as palettes 

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
    save_name = save_name.replace(' ', '_').replace('/', '_')
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