import os
import numpy as np
import pandas as pd
import pkg_resources
import scanpy as sc
import matplotlib.pyplot as plt
import mygene
import anndata

from scimmunity.utils import reorder_obs
import scimmunity.palette as palettes

# imports in functions: scrublet

def cellcycle(adata, layer='normalized'):
    file = pkg_resources.resource_filename('scimmunity', 
        'data/markersets/regev_lab_cell_cycle_genes.txt')
    cell_cycle_genes = [x.strip() for x in open(file)]
    s_genes = cell_cycle_genes[:43]
    g2m_genes = cell_cycle_genes[43:]
    cell_cycle_genes = [x for x in cell_cycle_genes if x in adata.var_names]

    # make copy from layer
    adata_copy = anndata.AnnData(adata.layers[layer])
    adata_copy.var_names = adata.var_names
    adata_copy.obs_names = adata.obs_names

    # score for S phase, a score for G2M phase and the predicted cell cycle phase 
    sc.tl.score_genes_cell_cycle(adata_copy, s_genes=s_genes, g2m_genes=g2m_genes, \
        use_raw=False)
    
    # copy to original adata
    adata.obs['phase'] = adata_copy.obs['phase']
    adata.obs['S_score'] = adata_copy.obs['S_score']
    adata.obs['G2M_score'] = adata_copy.obs['G2M_score']

    # add renamed cell cycle
    adata.obs['Cell Cycle'] = adata.obs['phase']
    for key in ['phase', 'Cell Cycle']:
        reorder_obs(adata, key,['G1', 'S', 'G2M'], palettes.feature2color['cellcycle'])

    return

def housekeeping(adata, layer='normalized'):
    file = pkg_resources.resource_filename('scimmunity', 
        'data/markersets/regev_lab_housekeeping_genes.xlsx')
    df_housekeeping = pd.read_excel(file)
    housekeeping_genes = list(df_housekeeping.iloc[:,0])

    housekeeping_present = [gene for gene in housekeeping_genes if gene in adata.raw.var_names]
    # get average housekeeping gene expression
    housekeeping_exp = adata.raw[:, housekeeping_present].X.mean(axis=1)
    adata.obs['housekeeping_raw'] = housekeeping_exp
    sc.pl.scatter(adata, x='n_counts', y='n_genes', size=12, \
        color='housekeeping_raw', color_map='viridis_r', save='_housekeeping_raw')

    adata_copy = anndata.AnnData(adata.layers[layer])
    adata_copy.var_names = adata.var_names
    adata_copy.obs_names = adata.obs_names
    housekeeping_present = [gene for gene in housekeeping_genes if gene in adata.var_names]
    # get average housekeeping gene expression
    housekeeping_exp = adata_copy[:, housekeeping_present].X.mean(axis=1)
    adata.obs['housekeeping'] = housekeeping_exp
    return

def heatshock(adata, species='human', layer='normalized'):
    adata_copy = anndata.AnnData(adata.layers[layer])
    adata_copy.var_names = adata.var_names
    adata_copy.obs_names = adata.obs_names
    # adata_copy.X = adata.layers[layer] 
    genes = adata.var['gene_ids']
    mg = mygene.MyGeneInfo()
    fields = ['symbol', 'name', 'go.BP.term']
    df = mg.querymany(genes, scopes='ensembl.gene', species=species, fields=fields, \
                                dotfield=True, as_dataframe=True)
    heat = df['go.BP.term'].apply(lambda x:'cellular response to heat' in x if type(x)==list else False)
    heat_genes = adata.var[adata.var['gene_ids'].isin(heat.index[heat])].index
    sc.tl.score_genes(adata_copy, heat_genes, score_name='heatshock_score', use_raw=False)
    adata.obs['heatshock_score'] = adata_copy.obs['heatshock_score']
    return 

### Doublet detection with Scrublet ###
def doublet(adata, sample='', layer='raw', thresh=None, out='./'):
    import scrublet as scr
    counts_matrix = adata.layers[layer]
    scrub = scr.Scrublet(counts_matrix, expected_doublet_rate=0.06)
    doublet_scores, predicted_doublets = scrub.scrub_doublets(min_counts=2, 
                                                          min_cells=3, 
                                                          min_gene_variability_pctl=85, 
                                                          n_prin_comps=30)
    scrub.plot_histogram()
    plt.savefig(os.path.join(out, f"{sample}_doublet_auto_cutoff.png"))
    plt.close()
    if thresh is not None:
        scrub.call_doublets(threshold=thresh)
        scrub.plot_histogram()
        plt.savefig(os.path.join(out, f"{sample}_doublet_{thresh}_cutoff.png"))
        plt.close()
    # add doublet information to adata
    adata.obs['doublet'] = scrub.predicted_doublets_
    adata.obs['doublet_score'] = doublet_scores
    return

def score_doublet_per_sample(adata, sample_key, thresh_list=[], out='./'):
    samples = adata.obs[sample_key].cat.categories
    doublet_calls = []
    doublet_scores = []
    if len(thresh_list) == 0:
        thresh_list = [None] * len(samples)
    for sample, thresh in zip(samples, thresh_list):
        subset = adata[adata.obs[sample_key]==sample]
        doublet(subset, sample=sample, thresh=thresh, out=out)
        doublet_calls.append(subset.obs['doublet'])
        doublet_scores.append(subset.obs['doublet_score'])
    adata.obs['doublet'] = pd.concat(doublet_calls)[adata.obs.index]
    adata.obs['doublet_score'] = pd.concat(doublet_scores)[adata.obs.index]
    return

def call_doublet_per_sample(adata, sample_key, thresh_list, out='./'):
    samples = adata.obs[sample_key].cat.categories
    doublet_calls = []
    for sample, thresh in zip(samples, thresh_list):
        subset = adata[adata.obs[sample_key]==sample]
        subset.obs['doublet'] = subset.obs['doublet_score']>thresh 
        plot_doublet_histogram(subset, sample, thresh, out=out)
        doublet_calls.append(subset.obs['doublet'])
    adata.obs['doublet'] = pd.concat(doublet_calls)[adata.obs.index]
    return 

def plot_doublet_histogram(adata, sample, thresh, out='./', scale_hist_obs='log'):
    ''' 
    Adapted from scrublet plot_histagram.
    Plot histogram of doublet scores for observed transcriptomes. 
    '''
    fig, ax = plt.subplots()
    ax.hist(adata.obs['doublet_score'], np.linspace(0, 1, 50), color='gray', linewidth=0, density=True)
    ax.set_yscale(scale_hist_obs)
    plt.axvline(thresh, c='black', linewidth=1)
    ax.set_xlabel('Doublet score')
    ax.set_ylabel('Prob. density')

    fig.tight_layout()
    plt.savefig(os.path.join(out, f"{sample}_doublet_{thresh}_cutoff.png"))
    return 