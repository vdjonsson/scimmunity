import gseapy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.stats import pearsonr
import seaborn as sns

from scimmunity.utils import mkdir
from scimmunity.annotation import get_signature_dict

###### Component Annotation ########

def corr_comp_gene(adata, rep, i, offset=1, layer=None):
    comps = adata.obsm['X_'+rep][:, offset:]
    genes = adata.var_names
    
    comp = comps[:, i]
    corr = []
    pvals = []
    for j, g in enumerate(genes):
#             if (j % 1000) == 0:
#                 print(j)
        exp = adata.obs_vector(g, layer=layer)
        r, p = pearsonr(comp, exp)
        corr.append(r)
        pvals.append(p)
    df = pd.DataFrame({'gene':genes, 'R':corr, 'P-value':pvals})
    
    return df.set_index('gene')

def df_corr_rep_gene(adata, rep, prefix='', dims=[0,1], offset=1, layer=None, returnR=True):
    """
    Use gseapy.get_library_name() to see more gene set option
    """
    genes = adata.var_names
    df_R = pd.DataFrame(index=genes)
    df_pvalue = pd.DataFrame(index=genes)
    for i in dims:
        df = corr_comp_gene(adata, rep, i, offset=offset, layer=layer)
        df_R['{}{}'.format(prefix, i+1)] = df['R'].values
        df_pvalue['{}{}'.format(prefix, i+1)] = df['P-value'].values
    if returnR:
        return df_R
    else:
        return df_R, df_pvalue
    
def corr_rep_gene(adata, rep, prefix='', dims=[0,1], offset=1, layer=None, thresh=0.5, out='./', \
                  gsets=['GO_Biological_Process_2018', 'KEGG_2019_Human', 'WikiPathways_2019_Human']):
    """
    Use gseapy.get_library_name() to see more gene set option
    """
    for i in dims:
        df = corr_comp_gene(adata, rep, i, offset=offset, layer=layer)
        df.to_csv('{}{}_{}{}_corr_gene.csv'.format(out, rep, prefix, i+1))
        df.sort_values('R', ascending=False).to_csv('{}{}_{}{}_corr_gene_sorted.csv'.format(out, rep, prefix, i+1))
        df_pos = df.loc[df['R']>thresh, :]
        df_neg = df.loc[df['R']<-thresh, :]
        for sign, df in zip(['pos','neg'], [df_pos, df_neg]):
            if len(df)>0:
                gseapy.enrichr(gene_list=list(df.index), gene_sets=gsets, outdir='{}{}{}_{}'.format(out, prefix, i+1, sign))
    return

def corr_rep_gene_dict(adata, markerset, rep, prefix='', dims=[0,1], offset=1, layer=None, out='./', save=True):
    """
    Calculate correlation coefficient R of each component with each gene. 
    Then for each component take the average of R for each gene signature/phenotype in the markerset.
    Args:
        adata (AnnData)
        markerset: name of markerset to annotate with
        rep (str): name of dimension reduction representation
        dims (list of int): indices (0-based) of components from dimension reduction 
        offset (int): Offset indices to skip unwanted component (ex. 1 for diffmap, 0 for others)
        layer (str): adata.layers key to get gene expression from
        out (str): output path
    Return:
        Saves correlation matrix as csv and heatmap.
    """
    mkdir(out)
    # load markerset
    gene_dict = get_signature_dict(markerset, adata=adata)
    phenotypes = list(gene_dict.keys())
    df = pd.DataFrame(index=phenotypes)
    
    for i in dims:
        df_list = []
        corr = corr_comp_gene(adata, rep, i, offset=offset, layer=layer)
        rs = []
        for phenotype in phenotypes:
            genes = gene_dict[phenotype]
            subset = corr.loc[genes, :]
            subset['phenotype'] = phenotype
            avg_r = corr.loc[genes, 'R'].mean()
            rs.append(avg_r)
            df_list.append(subset.sort_values('R'))
        df['{}{}'.format(prefix, i+1)] = rs
        pd.concat(df_list).to_csv('{}{}_{}{}.csv'.format(out, markerset, prefix, i+1))
    fig, ax = plt.subplots(figsize=(6,6))
    sns.heatmap(df, cmap='coolwarm', center=0, ax=ax)
    plt.tight_layout()
    plt.savefig('{}{}_{}_{}_avg_gene_corr.png'.format(out, rep, markerset, layer))
    plt.close()
    if save:
        df.to_csv('{}{}_{}_{}_avg_gene_corr.csv'.format(out, rep, markerset, layer))
    return df 

def pseudotime(adata, key, group):
    """
    key (str): adata.obs key (ex. "Phenotype")
    group (str): group in adata.obs key to select initial seed from (ex. Naive)
    Return:
        Update adata.obs with dpt_pseudotime
    """
    adata.uns['iroot'] = np.flatnonzero(adata.obs[key] == group)[0]
    sc.tl.dpt(adata, n_dcs=15)
    return 

def lineplot(x, y, error):
    with sns.axes_style('ticks'):
        plt.plot(x, y, '-')
        plt.fill_between(x, y-error, y+error, alpha=0.2)
    return

def gene_lineplot(dcs_mean, dcs_std, genes, figsize=None):
    if type(genes)==str:
        genes = [genes]
    plt.figure(figsize=figsize)
    for gene in genes:
        x = dcs_mean.index
        y = dcs_mean[gene]
        error = dcs_std[gene]
        lineplot(x, y, error)
    plt.ylabel('Average expression')
    plt.legend(genes, bbox_to_anchor=(1.1,1))
    sns.despine()
    return

def rolling_mean_dc(adata, basis, comp, genes, layer, window=10, win_type=None, plot=True, cellorder=False, out='./', figsize=None):
    x = "X_{}-{}".format(basis, comp)

    dcs = sc.get.obs_df(adata, keys=genes, obsm_keys=[("X_"+basis, comp)], layer=layer)

    # dcs = dcs.set_index(x).sort_index()
    if cellorder:
        dcs = dcs.sort_values(x).reset_index()[genes]
    else:
        dcs = dcs.sort_values(x).set_index(x)[genes]

    dcs_mean = dcs.rolling(window, win_type=win_type, axis=0).mean()
    dcs_std = dcs.rolling(window, win_type=win_type, axis=0).std()
    if plot:
        gene_lineplot(dcs_mean, dcs_std, genes, figsize=figsize)
        plt.xlabel(x)
        # plt.tight_layout()
        plt.savefig('{}{}{}_{}'.format(out, x,'_cellorder'*cellorder, '_'.join(genes)) , bbox_inches='tight')
        plt.close()
    return 