import os
import pandas as pd
import scanpy as sc
import mygene

import numpy as np
from scimmunity.utils import mkdir, clean_up_str, gsea

def de_layer(adata, label_name, 
    layer='normalized', n_genes=100, method='t-test', 
    filter=True, query=False, outdir='./',
    groups='all', reference='rest', 
    min_fold_change=2, min_in_group_fraction=0.25, max_out_group_fraction=0.5, 
    enrich=True):
    """
    Perform differential expression analysis based on group. 
    Args:
        adata
        label_name (str): key storing labels in adata.obs 
        layer (str): {"raw", "normalized", "corrected"}
        method (str): {'logreg', 't-test', 'wilcoxon', 't-test_overestim_var'}
        n_genes (int or str): Top number of genes to record. 'all' or integer
    """
    
    if n_genes=='all':
        n_genes = adata.shape[1]
    
    X = adata.X
    adata.X = adata.layers[layer] # set X to chosen layer

    comparison = f"v{reference}"
    key = f"rank_genes_groups_{label_name}_{method}_{layer}_{comparison}"
    sc.tl.rank_genes_groups(
                adata,
                label_name,
                method=method,
                use_raw=False,
                key_added=key,
                n_genes=n_genes,
                reference=reference,
                groups=groups
            )
    adata.X = X # restore original X
    save_de(adata, label_name, method, layer, comparison, out=outdir)
    plot_de(adata, label_name, method, layer, comparison, out=outdir)

    if filter:
        filter_de(adata, key, label_name, layer=layer, 
            min_fold_change=min_fold_change, 
            min_in_group_fraction=min_in_group_fraction, 
            max_out_group_fraction=max_out_group_fraction)
        save_de(adata, label_name, method, layer, comparison, out=outdir, 
            filtered=True, query=False, enrich=enrich)
        plot_de(adata, label_name, method, layer, comparison, out=outdir, 
            filtered=True)
    return 

def get_df_ranked_genes(adata, rank_genes_groups, 
    show_scores=True,
    keys=['scores', 'pvals', 'pvals_adj', 'logfoldchanges']):
    
    result = adata.uns[rank_genes_groups]
    groups = result['names'].dtype.names
    df_ranked_genes = pd.DataFrame()
    for group in groups:
        df_ranked_genes[group] = adata.uns[rank_genes_groups]['names'][group]
        for key in keys:
            if key in adata.uns[rank_genes_groups]:
                df_ranked_genes[group+'_'+key] = adata.uns[rank_genes_groups][key][group]
    return df_ranked_genes

def rank_genes_groups_df(adata, group, key, 
    pval_cutoff=5e-3, 
    log2fc_min=1, log2fc_max=None,
    gene_symbols='gene_ids'):
    """
    :func:`scanpy.tl.rank_genes_groups` results in the form of a
    :class:`~pandas.DataFrame`.
    Args:
        adata: Object to get results from.
    group: Which group to return results from.
    key: Key differential expression groups were stored under.
    pval_cutoff: Minimum adjusted pval to return.
    log2fc_min: Minumum logfc to return.
    log2fc_max: Maximum logfc to return.
    gene_symbols: Column name in `.var` DataFrame that stores gene symbols. Specifying
        this will add that column to the returned dataframe.
    """
    d = pd.DataFrame()
    for k in ['scores', 'names', 'logfoldchanges', 'pvals', 'pvals_adj']:
        if k in adata.uns[key]:
            d[k] = adata.uns[key][k][group]
    if pval_cutoff is not None:
        d = d[d["pvals_adj"] < pval_cutoff]
    if log2fc_min is not None:
        d = d[d["logfoldchanges"] > log2fc_min]
    if log2fc_max is not None:
        d = d[d["logfoldchanges"] < log2fc_max]
    if gene_symbols is not None:
        d = d.join(adata.var[gene_symbols], on="names")
    
    return d

def annotate_de(df, 
    gene_id_col='gene_ids', 
    keys=['scores', 'pvals', 'pvals_adj', 'logfoldchanges']):
    """
    Annotate genes in dataframe by querying for gene name, summary, and related pathways
    """
    top_genes = df[gene_id_col]
    mg = mygene.MyGeneInfo()
    fields = ['symbol', 'name', 'summary', 'go.BP.term','pathway.kegg.name']
    df_annotation = mg.querymany(top_genes, scopes='ensembl.gene', 
        species='human', fields=fields, dotfield=True, as_dataframe=True)
    df_annotation = df_annotation.loc[:, fields]
    cols = []
    for key in keys:
        # col = str(cluster)+'_'+key
        if key in df.columns:
            cols.append('DE_'+key)
            map_key = df[[gene_id_col, key]]
            map_key = map_key.set_index(gene_id_col)
            df_annotation['DE_'+key] = df_annotation.index.map(map_key[key]).values
    return df_annotation.loc[:,fields+cols]

def save_de(adata, label_name, method, layer, comparison, 
    out='./', filtered=False, 
    query=True, pval_cutoff=5e-3, log2fc_min=1, enrich=True):
    """
    Save rank genes groups results as csv per group. 
    Annotate and enrich up- and down-regulated genes per group filtered based on criteria.
    
    
    """
    filt = "_filtered"*filtered
    method_name = f"{method}_{layer}_{comparison}" + filt
    key = f"rank_genes_groups_{label_name}_{method_name}"
    # get overview of ranked genes
    df_ranked_genes = get_df_ranked_genes(adata, key)
    outdir = os.path.join(out, label_name, method_name)
    mkdir(outdir)
    df_ranked_genes.to_csv(os.path.join(outdir, f'{method_name}.csv'))

    # get up- and down-regulated per group based on criteria
    for i in adata.uns[key]['names'].dtype.names:
        df_up = rank_genes_groups_df(adata, group=i, key=key, 
            pval_cutoff=pval_cutoff, log2fc_min=log2fc_min, log2fc_max=None)
        df_down = rank_genes_groups_df(adata, group=i, key=key, 
            pval_cutoff=pval_cutoff, log2fc_max=-log2fc_min, log2fc_min=None)
        for df, direction in zip([df_up, df_down], ['up', 'down']):
            df = df.loc[(df['names']!='')&(~df['names'].isnull()),:]
            name = f"{method}_{layer}_{clean_up_str(i)}_{comparison+filt}"
            path = os.path.join(outdir, f"{name}_{direction}.csv")
            # gene name and description annotation
            if (len(df) > 0) and query:
                df_annotation = annotate_de(df)
                df_annotation.to_csv(path)
            else:
                df.to_csv(path)
            # Gene set enrichment
            if (len(df) > 0) and enrich:
                name = f"{clean_up_str(i)}_{direction}"
                gsea(list(df.names), description=name, out=os.path.join(outdir, name))
    return 

def plot_de(adata, label_name, method, layer, comparison, 
    n_genes=20, plot_layers=['corrected', 'normalized'], filtered=False, out='./'):
    temp = sc.settings.figdir 
    
    filt = "_filtered"*filtered
    method_name = f"{method}_{layer}_{comparison}" + filt
    key = f"rank_genes_groups_{label_name}_{method_name}"
    sc.settings.figdir = os.path.join(out, label_name)
    revert_filtered_de(adata, key)
    sc.pl.rank_genes_groups(adata, key=key, sharey=True, n_genes=n_genes, save='_'+method_name)
    for layer in plot_layers:
        sc.pl.rank_genes_groups_heatmap(adata, groupby=label_name, key=key, 
            save=f"_{method_name}_{layer}_stdscale", 
            cmap='RdBu_r', use_raw=False, 
            layer=layer, standard_scale='var', n_genes=n_genes)
    convert_filtered_de(adata, key)
    sc.settings.figdir = temp   
    return 

# convert between nan and string to save and load dataset properly 
def convert_filtered_de(adata, key):
    """nan -> str"""
    df = pd.DataFrame(adata.uns[key]['names'])
    adata.uns[key]['names'] = df.fillna('').to_records(index=False, column_dtypes='<U50')
    return

def revert_filtered_de(adata, key):
    """str -> nan"""
    df = pd.DataFrame(adata.uns[key]['names'])
    adata.uns[key]['names'] = df.replace('', np.nan).to_records(index=False)
    return    

def filter_de(adata, key, label_name, layer='normalized', 
    min_fold_change=2, min_in_group_fraction=0.25, max_out_group_fraction=0.5):
    filter_rank_genes_groups(adata, layer=layer, key=key, 
        key_added=(key+'_filtered'), groupby=label_name, 
        use_raw=False, log=True, min_fold_change=min_fold_change, 
        min_in_group_fraction=min_in_group_fraction, 
        max_out_group_fraction=max_out_group_fraction)
    
    names = pd.DataFrame(adata.uns[key+'_filtered']['names'])
    names = names.fillna('').to_records(index=False, column_dtypes='<U50')
    adata.uns[key+'_filtered']['names'] = names
    return

# modified version of sc.tl.filter_rank_genes_groups. Fix fold change calculations
def filter_rank_genes_groups(adata, key=None, groupby=None, layer='normalized', 
    use_raw=True, log=True, key_added='rank_genes_groups_filtered', 
    min_fold_change=2, min_in_group_fraction=0.25, max_out_group_fraction=0.5):

    if key is None:
        key = 'rank_genes_groups'

    if groupby is None:
        groupby = str(adata.uns[key]['params']['groupby'])

    # convert structured numpy array into DataFrame
    gene_names = pd.DataFrame(adata.uns[key]['names'])

    fraction_in_cluster_matrix = pd.DataFrame(np.zeros(gene_names.shape), 
        columns=gene_names.columns, index=gene_names.index)
    fold_change_matrix = pd.DataFrame(np.zeros(gene_names.shape), 
        columns=gene_names.columns, index=gene_names.index)
    fraction_out_cluster_matrix = pd.DataFrame(np.zeros(gene_names.shape), 
        columns=gene_names.columns, index=gene_names.index)
    
    from scanpy.plotting._anndata import _prepare_dataframe
    for cluster in gene_names.columns:
        # iterate per column
        var_names = gene_names[cluster].values

        # add column to adata as __is_in_cluster__. This facilitates to measure fold change
        # of each gene with respect to all other clusters
        adata.obs['__is_in_cluster__'] = pd.Categorical(adata.obs[groupby] == cluster)

        # obs_tidy has rows=groupby, columns=var_names
        categories, obs_tidy = _prepare_dataframe(adata, var_names, 
            groupby='__is_in_cluster__', layer=layer, use_raw=use_raw)

        # for if category defined by groupby (if any) compute for each var_name
        # 1. the mean value over the category
        # 2. the fraction of cells in the category having a value > 0

        # 1. compute mean value
        mean_obs = obs_tidy.groupby(level=0).mean()

        # 2. compute fraction of cells having value >0
        # transform obs_tidy into boolean matrix
        obs_bool = obs_tidy.astype(bool)

        # compute the sum per group which in the boolean matrix this is the number
        # of values >0, and divide the result by the total number of values in the group
        # (given by `count()`)
        fraction_obs = obs_bool.groupby(level=0).sum() / obs_bool.groupby(level=0).count()

        # Because the dataframe groupby is based on the '__is_in_cluster__' column,
        # in this context, [True] means __is_in_cluster__.
        # Also, in this context, fraction_obs.loc[True].values is the row of values
        # that is assigned *as column* to fraction_in_cluster_matrix to follow the
        # structure of the gene_names dataFrame
        fraction_in_cluster_matrix.loc[:, cluster] = fraction_obs.loc[True].values
        fraction_out_cluster_matrix.loc[:, cluster] = fraction_obs.loc[False].values

        # compute fold change.
        if log:
            fold_change_matrix.loc[:, cluster] = ((np.expm1(mean_obs.loc[True]) + 1e-9)/ (np.expm1(mean_obs.loc[False])+ 1e-9)).values
        else:
            fold_change_matrix.loc[:, cluster] = (mean_obs.loc[True] / mean_obs.loc[False]).values

    # remove temporary columns
    adata.obs.drop(columns='__is_in_cluster__')
    # filter original_matrix
    gene_names = gene_names[(fraction_in_cluster_matrix > min_in_group_fraction) &
                            (fraction_out_cluster_matrix < max_out_group_fraction) &
                            (fold_change_matrix > min_fold_change)]
    # create new structured array using 'key_added'.
    adata.uns[key_added] = adata.uns[key].copy()
    adata.uns[key_added]['names'] = gene_names.to_records(index=False)
    return 