import os
import collections 

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns

import scanpy as sc

from scipy.sparse import issparse

from scimmunity.config import MARKERSETS, POP2MARKERSETCHOICES, POP2MARKERSETS, POP2PHENOTYPE
from scimmunity.config import BULKPROFILES, POP2BULKPROFILES

from scimmunity.plotting import plot_reps_markers, plot_reps
from scimmunity.utils import mkdir
import scimmunity.palette as palettes
from scimmunity.de import convert_filtered_de, revert_filtered_de

def df_cluster_mean_expression(adata, genes, clustering, layer='normalized', \
    use_raw=False, var_col='gene_names'):
    """
    Return:
        average_obs (ndarray): genes x clusters dataframe of mean expression
        with cluster as columns, genes as index  
    """
    if use_raw:
        adata = adata.raw.copy()
    else:
        adata = adata.copy()
    
    adata.var.index = adata.var[var_col]

    genes_detected = []
    genes_missing = []
    for g in genes:
        if g in adata.var_names:
            genes_detected.append(g)
        else:
            genes_missing.append(g)
    if len(genes_missing) > 0:
        print(len(genes_missing), 'genes not detected')

    if layer is None:
        obs =  adata[:,genes_detected].X
    else:
        obs = adata[:, genes_detected].layers[layer]
    if issparse(obs):
        obs = obs.toarray()
        
    obs = pd.DataFrame(obs,columns=genes_detected, index=adata.obs[clustering])
    average_obs = obs.groupby(level=0).mean().T
    return average_obs

def gene_dict_from_df(df):
    gene_dict = collections.defaultdict(list)
    for i, row in df.iterrows():
        gene_dict[row.phenotype].append(row.symbol)
    return gene_dict

def get_signature_dict(markerset, adata=None):
    df_markers = pd.read_excel(MARKERSETS[markerset]).astype(str)
    # remove extra spaces
    df_markers['symbol'] = df_markers['symbol'].apply(lambda x:x.strip())
    if adata is not None:
        inds = df_markers['symbol'].isin(adata.var_names.values)
        if sum(~inds) > 0:
            print('{} genes not detected:'.format(markerset))
            print(*df_markers.loc[~inds, 'symbol'].values)
        df_markers = df_markers.loc[inds]
    signature_dict = gene_dict_from_df(df_markers)
    return signature_dict

def df_signature_expression(adata, gene_dict, layer=None, use_raw=False):
    """
    Args:
        adata (AnnData): anndata.AnnData instance
        gene_dict (str : list of str): dictionary mapping phenotype to genes
        layer (str): adata layer to be used {'raw', 'normalized', 'corrected', etc} 
        use_raw (bool): use adata.raw.X if True 
    Return
        genes x cells dataframe for genes in gene_dict, 
        with celltype as index, barcodes as columns
    """
    if use_raw:
        adata = adata.raw
    celltype_detected = []
    genes_detected = []
    genes_missing = collections.defaultdict(list)
    for celltype, genes in gene_dict.items():
        for g in genes:
            if g in adata.var_names:
                genes_detected.append(g)
                celltype_detected.append(celltype)
            else:
                genes_missing[celltype].append(g)
    if len(genes_missing) > 0:
        print('genes not detected:')
        for celltype in genes_missing:
            print(celltype, genes_missing[celltype])

    if layer is None:
        obs =  adata[:,genes_detected].X.T
    else:
        obs = adata[:, genes_detected].layers[layer].T
    if issparse(obs):
        obs = obs.toarray()
    
    # index by phenotype for downstream groupby operations
    obs = pd.DataFrame(obs,columns=adata.obs_names,index=celltype_detected)
    return obs 

def df_signature_mean_expression(adata, gene_dict, layer=None, use_raw=False):
    """
    Return:
        average_obs : signatures x cells array of mean expression, \
            with celltype as index, barcodes as col names
    """
    obs = df_signature_expression(adata, gene_dict, layer=layer, use_raw=use_raw)
    # average per phenotype
    average_obs = obs.groupby(level=0).mean().T
    return average_obs

def df_signature_mean_detection(adata, gene_dict, layer=None, use_raw=False, limit='zero', normalize=False):
    """
    limit (str): {zero, mean}
    Return:
        signatures x cells array of mean detection 
    """
    
    obs = df_signature_expression(adata, gene_dict, layer=layer, use_raw=use_raw)

    if limit == 'zero':
        lim = 0 
    elif limit == 'mean':
        lim = obs.mean(axis=1)[:, None] # global mean expression
         
    # normalize by fraction of genes detected in cell
    if normalize:
        obs = (obs > lim) / adata.obs['n_genes'].T[None, :] 
    else:
        obs = (obs > lim) 
    # average per phenotype
    average_obs = obs.groupby(level=0).mean()
    return average_obs.T

def df_avg_cluster_signature_mean_expression(adata, gene_dict, clustering, layer=None, use_raw=False):
    df_signature = df_signature_mean_expression(adata, gene_dict, layer=layer, use_raw=use_raw)
    df_signature[clustering] = adata.obs[clustering]
    return df_signature.groupby(clustering).mean().T

def df_avg_cluster_signature_mean_detection(adata, gene_dict, clustering, layer=None, use_raw=False, limit='mean', normalize=False):
    df_signature = df_signature_mean_detection(adata, gene_dict, layer=layer, use_raw=use_raw, limit=limit, normalize=normalize)
    df_signature[clustering] = adata.obs[clustering]
    return df_signature.groupby(clustering).mean().T


def corr_mean_bulk(adata, clustering, bulkprofiles=['novershtern'], annot=True, out='./', plot=True,\
    row_cluster=False, col_cluster=False, layer='normalized', use_raw=False, avg=True):
    clusters = adata.obs[clustering].cat.categories
    cluster_colors = adata.uns[clustering+'_colors']

    all_corr = []
    all_celltypes = []
    for bulkprofile in bulkprofiles: 
        bulkfile = BULKPROFILES[bulkprofile]
        # 0th column are gene symbols, drop columns filled with nan
        df_bulk = pd.read_csv(bulkfile, index_col=0).dropna(axis=1, how='all') 
        df_bulk.index = df_bulk.index.values
        celltypes = df_bulk.columns
        all_celltypes += list(celltypes)

        if np.all(df_bulk.index.str.contains('ENSG')):
            var_col = 'gene_ids'
        else:
            var_col = 'gene_names'
        df_cluster = df_cluster_mean_expression(adata, df_bulk.index, clustering, layer=layer, use_raw=use_raw, var_col=var_col)
        # convert from CategoricalIndex to Index type before merging dfs
        df_bulk.columns = df_bulk.columns.astype('str')
        df_cluster.columns = df_cluster.columns.astype('str')
        # merge dfs horizontally using intersection of genes
        df = pd.concat([df_cluster, df_bulk], axis=1, sort=False, join='inner')
        # calculate correlation matrix
        corr = df.corr() 
        diag = corr.loc[celltypes, clusters]
        # take mean across biological replicates
        if avg:
            #diag['celltype'] = [name.split('-')[1] for name in diag.index.values]
            diag['celltype'] = ['-'.join(name.split('-')[1:]) for name in diag.index.values]
            diag = diag.groupby('celltype').mean()

        all_corr.append(diag)
            
    combined_corr = pd.concat(all_corr, axis=0)

    return combined_corr

def rank_df(df):
    df = df.copy()
    for col in df.columns:
        df[col] = df[col].rank(na_option='keep', ascending=False)
    return df

def rank_corr_bulk(adata, bulkprofiles, clustering, method, layer='normalized', out='./', plot_metric=True, plot_rank=True, **kwargs):
    save_name = '_'.join(bulkprofiles + [method])
    if method == 'corr_mean_bulk':
        df = corr_mean_bulk(adata, clustering, bulkprofiles=bulkprofiles, layer=layer)
        save_name += '_'+layer

    df_rank = rank_df(df)

    col_colors = adata.uns[clustering+'_colors']
    if plot_metric:
        # center = 0?
        plot_df_heatmap(df, cmap='coolwarm', center=None,  out=out, col_colors=col_colors, \
            save_name=save_name, annot_kws={'fontsize':5}, **kwargs) 
        plot_df_heatmap(df, cmap='coolwarm', center=None,  out=out, col_colors=col_colors, \
            save_name=save_name+'_clustered', annot_kws={'fontsize':5}, row_cluster=True, col_cluster=True) 

    if plot_rank:
        plot_df_heatmap(df_rank, cmap='viridis_r', annot=True, out=out, col_colors=col_colors, \
            save_name='rank_'+save_name, **kwargs)
        plot_df_heatmap(df_rank, cmap='viridis_r', annot=True, out=out, col_colors=col_colors, \
            save_name='rank_'+save_name+'_clustered', row_cluster=True, col_cluster=True)
    return df_rank, save_name

def rank_gene_dict(adata, name, gene_dict, clustering, method, layer='normalized', 
    de_method=None, de_filtered=False, out='./', plot_metric=True, plot_rank=True, 
    top_n_markers=None, adj_pval_threshold=None, **kwargs):
    """
    Args:
        adata (Anndata)
        name (str): name of gene_dict
        gene_dict (str: list of str) dictionary mapping phenotype to genes
        clustering (str): key in adata.obs storing clustering labels
        method (str): ranking method 
            "avg_exp"
            "avg_det"
            "de_overlap_count"
            "de_overlap_count_normalized"
            "de_overlap_coef"
            "de_jaccard"
        de_method (str): method_layer identifier
            {'scvi', 't-test', 'wilcoxon'}_{'raw', 'normalized', 'corrected'}
    """
    save_name = name +'_' + method
    if method == 'avg_exp':
        df = df_avg_cluster_signature_mean_expression(adata, gene_dict, clustering, layer=layer)
        save_name += '_' + layer
        
    if method == 'avg_det':
        df = df_avg_cluster_signature_mean_detection(adata, gene_dict, clustering, layer=layer)
        save_name += '_' + layer

    de_key = 'rank_genes_groups_{}_{}_vrest{}'.format(clustering, de_method, '_filtered'*de_filtered)

    if de_filtered: 
        revert_filtered_de(adata, de_key) # convert "nan" to np.nan

    if method == 'de_overlap_count':
        df = sc.tl.marker_gene_overlap(adata, gene_dict, key=de_key, method='overlap_count', top_n_markers=top_n_markers, adj_pval_threshold=adj_pval_threshold)
        save_name += '_' + de_method +'_filtered'*de_filtered
    if method == 'de_overlap_count_normalized':
        df = sc.tl.marker_gene_overlap(adata, gene_dict, key=de_key, method='overlap_count', normalize='reference', top_n_markers=top_n_markers, adj_pval_threshold=adj_pval_threshold)
        save_name += '_' + de_method +'_filtered'*de_filtered
    if method == 'de_overlap_coef':
        df = sc.tl.marker_gene_overlap(adata, gene_dict, key=de_key, method='overlap_coef', top_n_markers=top_n_markers, adj_pval_threshold=adj_pval_threshold)
        save_name += '_' + de_method +'_filtered'*de_filtered
    if method == 'de_jaccard':
        df = sc.tl.marker_gene_overlap(adata, gene_dict, key=de_key, method='jaccard', top_n_markers=top_n_markers, adj_pval_threshold=adj_pval_threshold)
        save_name += '_' + de_method +'_filtered'*de_filtered
    df_rank = rank_df(df)
    
    col_colors = adata.uns[clustering+'_colors']
    if plot_metric:
        plot_df_heatmap(df, cmap='rocket', out=out, col_colors=col_colors, \
            save_name=save_name, **kwargs)

    if plot_rank:
        plot_df_heatmap(df_rank, cmap='viridis_r', annot=True, out=out, col_colors=col_colors, \
            save_name='rank_'+save_name, **kwargs)
    
    if de_filtered:
        convert_filtered_de(adata, de_key) # change back to str type for saving

    return df_rank, save_name

def plot_df_heatmap(df, out='./', col_colors=None, row_cluster=False, col_cluster=False, \
    annot=False, cmap='rocket', center=None, save_name=None, labelsize=None, figsize='adjust', annot_kws=None):
    if figsize is None:
        figsize = rcParams['figure.figsize']
    elif figsize == 'adjust':
        x = rcParams['figure.figsize'][0]
        y = x * df.shape[0] / df.shape[1] 
        # x =/ (1-0.01*row_cluster)
        # y =/ (1-0.01*col_cluster)
        figsize = (x, y)
    elif not ((type(figsize)==list) or (type(figsize)==tuple)):
        raise ValueError('Enter None, "adjust", or tuple for figsize')

    g = sns.clustermap(df, cmap=cmap, center=center, annot=annot, figsize=figsize, \
        col_cluster=col_cluster, row_cluster=row_cluster, col_colors=col_colors, annot_kws=annot_kws)
    g.ax_heatmap.tick_params(axis='both', which='major', labelsize=labelsize, labelrotation=0)
    if save_name is not None:
        g.savefig('{}{}.png'.format(out, save_name))
    else:
        return g

def plot_corr(corr, X, Y, out='./', col_colors=None, row_cluster=False, col_cluster=False, \
    annot=True, cmap='coolwarm', center=0, save_name=None, labelsize=None, figsize='adjust', \
    annot_kws={'fontsize':5}):
    df = corr.loc[Y, X]
    plot_df_heatmap(df, out=out, col_colors=col_colors, row_cluster=row_cluster, col_cluster=col_cluster, \
    annot=annot, cmap=cmap, center=center, save_name=save_name, labelsize=labelsize, figsize=figsize, annot_kws=annot_kws)
    return 
def plot_phenotype_markerset(adata, key, markersets, out='./', mode='heatmap', 
    layers=['corrected', 'normalized']):
    temp = sc.settings.figdir 
    sc.settings.figdir  = out
    combined_dict = {}
    for markerset in markersets:
        signature_dict = get_signature_dict(markerset, adata=adata)
        combined_dict.update(signature_dict)
    sc.tl.dendrogram(adata, groupby=key)
    order = adata.uns['dendrogram_{}'.format(key)]['dendrogram_info']['ivl']
    ordered_dict = collections.OrderedDict()
    for x in order:
        if x in combined_dict:
            ordered_dict[x] = combined_dict[x]
    functions = {
        'heatmap':sc.pl.heatmap, 
        'matrixplot':sc.pl.matrixplot, 
        'dotplot':sc.pl.dotplot}
    function = functions[mode]
    temp_cmap = rcParams['image.cmap'] 
    rcParams['image.cmap'] = 'magma'
    markersetname = '_'.join(markersets)
    for layer in layers:
        if len(ordered_dict)>0:
            function(adata, ordered_dict, groupby=key, 
            use_raw=False, dendrogram=True, layer=layer, 
            save=f"_{key}_{markersetname}_{layer}")

            function(adata, ordered_dict, groupby=key, 
            use_raw=False, dendrogram=True, layer=layer, 
            save=f"_{key}_{markersetname}_{layer}_stdscale",
            standard_scale='var')

        function(adata, combined_dict, groupby=key, 
        use_raw=False, dendrogram=True, layer=layer, 
        save=f"_{key}_{markersetname}_all_{layer}")

        function(adata, combined_dict, groupby=key, 
        use_raw=False, dendrogram=True, layer=layer, 
        save=f"_{key}_{markersetname}_all_{layer}_stdscale",
        standard_scale='var')
    sc.settings.figdir  = temp   
    rcParams['image.cmap'] = temp_cmap
    return

def de_overlap(adata, key, signature_dict):
    # convert structured numpy array into DataFrame
    de = pd.DataFrame(adata.uns[key]['names'])
    groups = []
    phenotypes = []
    genes = []
    for group in de.columns:
        for phenotype in signature_dict:
            overlap = set(signature_dict[phenotype]).intersection(de[group].values)
            groups += [group]*len(overlap)
            phenotypes += [phenotype]*len(overlap)
            genes += list(overlap)
    df = pd.DataFrame({'group':groups, 'phenotype':phenotypes, 'gene':genes})
    return df

def save_de_overlap(adata, key, markersets, out='./'):
    combined_dict = {}
    for markerset in markersets:
        signature_dict = get_signature_dict(markerset, adata=adata)
        combined_dict.update(signature_dict) 
    df = de_overlap(adata, key, combined_dict)
    mkdir(out)
    df.to_csv(os.path.join(out, f'de_overlap_{key}_{markerset}.csv'), index=False)
    # df.to_csv('{}de_overlap_{}_{}.csv'.format(out, key, markerset), index=False)
    return 

def plot_reps_signature_dict(adata, markerset, use_raw=False, layer=None, 
    out='./', reps=['umap', 'diffmap'], figsize=(4,4)):
    mkdir(out)
    signature_dict = get_signature_dict(markerset, adata=adata)
    for celltype, genes in signature_dict.items():
        plot_reps_markers(adata, genes, markerset+'_'+celltype.strip(), \
            outdir=out, reps=reps, use_raw=use_raw, layer=layer, figsize=figsize)
    return

class clusterAnnotation():
    def __init__(self, adata, outdir, clustering, out='annotation', 
        bulkprofiles=BULKPROFILES, 
        markersets=MARKERSETS, 
        pop2markersetchoices=POP2MARKERSETCHOICES, 
        pop2markersets=POP2MARKERSETS, 
        pop2bulkprofiles=POP2BULKPROFILES, 
        pop2phenotype=POP2PHENOTYPE,
        reps=None):
        """
        Automatic cluster annotation 
        1) Avg cell marker detection rate (fraction of total genes detected)
        2) Correlate cluster centroid with bulk profiles
            - use only genes with coeff. of variation >20% in bulk dataset?
        3) compare DE genes to know marker genes
        Args:
            h5ad (str): path to '.h5ad' file
            out (str): name of subfolder for output (default: annotation)
            clustering (key): key name of clustering stored in adata.obs
        """
        
        # self.h5ad = h5ad
        # self.out = '{}/{}/{}/'.format(os.path.dirname(self.h5ad), out, clustering)
        self.out = os.path.join(outdir, clustering)
        self.adata = adata
        self.clustering = clustering
        self.bulkprofiles = bulkprofiles
        self.markersets = markersets
        self.pop2markersetchoices=pop2markersetchoices
        self.pop2markersets = pop2markersets
        self.pop2phenotype = pop2phenotype
        self.pop2bulkprofiles = pop2bulkprofiles
        if reps is None:
             self.reps = ['pca',
                'umap_pcs', 'umap_latent', 'umap_latent_regressed', 
                'tsne_pcs', 'tsne_latent', 'tsne_latent_regressed', 
                'diffmap_pcs', 'diffmap_latent', 'diffmap_latent_regressed']
        else:
            self.reps = reps
        # make output folder
        mkdir(self.out)
        return
    
    def get_signature_dict(self, markerset):
        df_markers = pd.read_excel(self.markersets[markerset]).astype(str)
        df_markers['symbol'] = df_markers['symbol'].apply(lambda x:x.strip())
        df_markers = df_markers.loc[df_markers['symbol'].isin(self.adata.var_names.values)]
        signature_dict = gene_dict_from_df(df_markers)
        return signature_dict

    def plot_reps_signature_dict(self, signature_dict, markerset, use_raw=False, 
        dpi=300, layer=None, figsize=(2,2)):

        old_dpi = rcParams["savefig.dpi"] 
        rcParams["savefig.dpi"] = dpi
        out = os.path.join(self.out, 'expression')
        mkdir(out)
        for celltype, genes in signature_dict.items():
            plot_reps_markers(self.adata, genes, markerset+'_'+celltype.strip(), \
                outdir=out, reps=self.reps, use_raw=use_raw, layer=layer, figsize=figsize)
        rcParams["savefig.dpi"] = old_dpi
        return

    def set_highest_ranked_phenotype(self, df_rank, name='', keepcluster=False):
        cluster2phenotype = {}
        for col in df_rank.columns:
            # if equally ranked, set phenotype as NA
            if len(df_rank[col].unique()) ==  1:
                cluster2phenotype[col] = 'NA'
            else:
                cluster2phenotype[col] = str(df_rank[col].idxmin())
        
        # make new clustering
        if len(name)>0:
            name = '_' + name
        if keepcluster:
            key = 'phenotype_'+self.clustering+name
            self.adata.obs[key] = self.adata.obs[self.clustering].apply(\
                lambda x:'{}:{}'.format(x, cluster2phenotype[x]))
            
        else:
            key = 'phenotype'+name
            self.adata.obs[key] = self.adata.obs[self.clustering].apply(\
                lambda x:cluster2phenotype[x])
        
        out = os.path.join(self.out,'phenotype')
        mkdir(out)
        plot_reps(self.adata, key, save_name=key, outdir=out)
        plot_reps(self.adata, key, save_name=key+'_ondata', outdir=out, \
            legend_loc='on data', legend_fontweight='normal', legend_fontsize=10)

        return cluster2phenotype

    def annotate_bulkprofiles(self, bulkprofiles):
        methods = ['corr_mean_bulk']
        # name = '_'.join(bulkprofiles)

        save_names = []
        cluster2phenotypes = []
        for method in methods:
            for layer in ['normalized', 'corrected']:
                df_rank, save_name = rank_corr_bulk(self.adata, bulkprofiles, \
                    self.clustering, method=method, layer=layer, out=self.out)
                cluster2phenotype = self.set_highest_ranked_phenotype(df_rank, \
                    name=save_name)
                self.set_highest_ranked_phenotype(df_rank, keepcluster=True, \
                        name=save_name)
                save_names.append(save_name)
                cluster2phenotypes.append(cluster2phenotype)
        return save_names, cluster2phenotypes

    def annotate_markersets(self, markersets, plot_reps=True, 
        plot_reps_dpi=300, figsize=(2,2)):
        methods_kwargs = [
            ['avg_exp', dict()], 
            ['avg_det', dict()],
            ['de_overlap_count',{'de_method':'t-test_normalized', 'de_filtered':True, 'adj_pval_threshold':5e-3}],
            ['de_overlap_count_normalized',{'de_method':'t-test_normalized', 'de_filtered':True, 'adj_pval_threshold':5e-3}],
            ['de_overlap_coef',{'de_method':'t-test_normalized', 'de_filtered':True, 'adj_pval_threshold':5e-3}],
            ['de_jaccard',{'de_method':'t-test_normalized', 'de_filtered':True, 'adj_pval_threshold':5e-3}]
            ]
        # ['de_overlap_count',{'de_method':'t-test_normalized', 'de_filtered':False, 'top_n_markers':500}]
        markersetname = '_'.join(markersets)
        combined_dict = {}
        
        for markerset in markersets:
            signature_dict = self.get_signature_dict(markerset)
            combined_dict.update(signature_dict) # note this would overwrite identical keys
        if plot_reps:
            for layer in ['corrected', 'corrected_regressed', 'normalized']:
                self.plot_reps_signature_dict(combined_dict, markersetname, 
                    layer=layer, dpi=plot_reps_dpi, figsize=figsize)
            
        
        save_names = []
        cluster2phenotypes = []
        for method, kwargs in methods_kwargs:
            for layer in ['normalized', 'corrected']:
                df_rank, save_name = rank_gene_dict(self.adata, markersetname, combined_dict, \
                    self.clustering, method=method, out=self.out, layer=layer, **kwargs)
                cluster2phenotype = self.set_highest_ranked_phenotype(df_rank, \
                    name=save_name)
                self.set_highest_ranked_phenotype(df_rank, keepcluster=True, \
                        name=save_name)

                save_names.append(save_name)
                cluster2phenotypes.append(cluster2phenotype)
        
        temp = sc.settings.figdir 
        sc.settings.figdir  = self.out
        # plot heatmap/matrixplot/dotplot of genes in the markersets
        genes = [g for gl in combined_dict.values() for g in gl]
        for layer in ['corrected', 'raw', 'normalized']:
            if issparse(self.adata.layers[layer]):
                mat = self.adata[:,genes].layers[layer].toarray()
                vmin = None
                vmax = np.percentile(mat, 98)
            else:
                mat = self.adata.layers[layer]
                vmin = None
                vmax = np.percentile(mat, 98)
            
            for function in [sc.pl.heatmap, sc.pl.matrixplot, sc.pl.dotplot]:
                kwargs = {}
                if function == sc.pl.heatmap:
                    kwargs.update({'vmin':vmin, 'vmax':vmax, "cmap":"magma"})
                if function == sc.pl.matrixplot:
                    kwargs.update({"cmap":"magma"})
                function(self.adata, combined_dict, groupby=self.clustering, 
                    use_raw=False, dendrogram=True, layer=layer, 
                    save=f"_{markersetname}_{layer}", **kwargs)
                kwargs = {k:v for k,v in kwargs.items() if k not in ['vmin', 'vmax']}
                function(self.adata, combined_dict, groupby=self.clustering, 
                    use_raw=False, dendrogram=True, layer=layer, 
                    save=f"_{markersetname}_{layer}_stdscale", 
                    standard_scale='var', **kwargs)
            sc.settings.figdir  = temp   

            # sc.pl.heatmap(self.adata, combined_dict, groupby=self.clustering, \
            # use_raw=False, dendrogram=True, layer=layer, save='_{}_{}'.format(markersetname, layer), \
            # vmin=vmin, vmax=vmax, cmap='magma')

            # sc.pl.heatmap(self.adata, combined_dict, groupby=self.clustering, \
            # use_raw=False, dendrogram=True, layer=layer, save='_{}_{}_stdscale'.format(markersetname, layer), \
            # standard_scale='var', cmap='magma')

            # sc.pl.matrixplot(self.adata, combined_dict, groupby=self.clustering, \
            # use_raw=False, dendrogram=True, layer=layer, save='_{}_{}_stdscale'.format(markersetname, layer), \
            # standard_scale='var', cmap='magma')

            # sc.pl.matrixplot(self.adata, combined_dict, groupby=self.clustering, \
            # use_raw=False, dendrogram=True, layer=layer, save='_{}_{}'.format(markersetname, layer), cmap='magma')

            # sc.pl.dotplot(self.adata, combined_dict, groupby=self.clustering, \
            # use_raw=False, dendrogram=True, layer=layer, save='_{}_{}_stdscale'.format(markersetname, layer), \
            # standard_scale='var')

            # sc.pl.dotplot(self.adata, combined_dict, groupby=self.clustering, \
            # use_raw=False, dendrogram=True, layer=layer, save='_{}_{}'.format(markersetname, layer))
        return save_names, cluster2phenotypes

    def try_annotation(self, population, plot_reps=True, try_all=True, 
        figsize=(2,2), plot_reps_dpi=300):
        clusters = self.adata.obs[self.clustering].cat.categories
        df = pd.DataFrame(index=clusters)
        identifier = []
        phenotype = []
        # try each markerset individually
        if try_all:
            markersets = self.pop2markersetchoices[population]
            for markerset in markersets:
                save_names, cluster2phenotypes = self.annotate_markersets(
                    [markerset], plot_reps=plot_reps, figsize=figsize, 
                    plot_reps_dpi=plot_reps_dpi)
                for save_name, cluster2phenotype in zip(save_names, cluster2phenotypes):
                    df[save_name] = [cluster2phenotype[i] for i in clusters]
        
        # try the selected combination
        markersets = self.pop2markersets[population]
        save_names, cluster2phenotypes = self.annotate_markersets(
            markersets, plot_reps=plot_reps, figsize=figsize, 
            plot_reps_dpi=plot_reps_dpi)
        for save_name, cluster2phenotype in zip(save_names, cluster2phenotypes):
            df[save_name] = [cluster2phenotype[i] for i in clusters]

        # try bulk profiles individually
        bulkprofiles = self.pop2bulkprofiles[population]
        for bulkprofile in bulkprofiles:
            save_names, cluster2phenotypes = self.annotate_bulkprofiles([bulkprofile])
            for save_name, cluster2phenotype in zip(save_names, cluster2phenotypes):
                df[save_name] = [cluster2phenotype[i] for i in clusters]

        df.to_csv(os.path.join(self.out, 'annotation.csv'))
        return

    def set_annotation(self, population, pop2phenotype, markersets=None):
        phenotype = pop2phenotype[population]
        self.adata.obs['Phenotype'] = self.adata.obs[phenotype]
        # reset phenotype colors to avoid slicing error
        if 'Phenotype_colors' in self.adata.uns:
            del self.adata.uns['Phenotype_colors']

        # plot the set phenotype
        out = os.path.join(self.out, 'phenotype')
        mkdir(out)

        plot_reps(self.adata, 'Phenotype', save_name='Phenotype'+'_ondata', 
            outdir=out, reps=self.reps,
            legend_loc='on data', legend_fontweight='normal', legend_fontsize=10)
        plot_reps(self.adata, 'Phenotype', save_name='Phenotype', 
            outdir=out, reps=self.reps)
        
        # plot markerset heatmap
        if markersets is None:
            markersets = self.pop2markersets[population]

        plot_phenotype_markerset(self.adata, 'Phenotype', markersets, out=self.out, mode='heatmap')
        plot_phenotype_markerset(self.adata, 'Phenotype', markersets, out=self.out, mode='matrixplot')
        plot_phenotype_markerset(self.adata, 'Phenotype', markersets, out=self.out, mode='dotplot')
        return
