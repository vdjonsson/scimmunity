import os
import re
import time
import numpy as np
import gseapy

def mkdir(path):
    if not os.path.isdir(path):
        os.system('mkdir -p '+path)
    return

def reorder_obs(adata, key, order, colors):
    new_order = []
    new_colors = []
    categories = adata.obs[key].unique() 
    for cat, color in zip(order, colors):
        if cat in categories:
            new_order.append(cat)
            new_colors.append(color)

    adata.obs[key] = adata.obs[key].astype('category').cat.reorder_categories(new_order, ordered=True)
    adata.uns[key+'_colors'] = new_colors
    return 

def explode_str(df, col, sep):
    """
    Explode dataframe by doing str split on a column
    """
    s = df[col]
    i = np.arange(len(s)).repeat(s.str.count(sep) + 1)
    return df.iloc[i].assign(**{col: sep.join(s).split(sep)})

def explode_strs(df, cols, sep):
    """
    Explode dataframe by doing str split on multiple columns
    """
    s = df[cols[0]]
    i = np.arange(len(s)).repeat(s.str.count(sep) + 1)
    return df.iloc[i].assign(**{col: sep.join(df[col]).split(sep) for col in cols})

def clean_up_str(string, replace='_'):
    return re.sub('[^A-Za-z0-9]+', replace, string)

def gsea(genes, description='', out='./', sleeptime=1, sleep=False,
    gsets=['GO_Biological_Process_2018', 'KEGG_2019_Human', 'WikiPathways_2019_Human']):
    """
    genes (list of str): gene symbols
    description (str): name for enrichment report
    sleeptime (int): length of wait time between each query 
        (overloading server causes connection to be cut)
    """
    if sleep:
        for gset in gsets:
            time.sleep(sleeptime)
            gseapy.enrichr(gene_list=genes, description=description, 
                gene_sets=gset, outdir=out)
    else:
        gseapy.enrichr(gene_list=genes, description=description, gene_sets=gsets, 
            outdir=out)
    return 