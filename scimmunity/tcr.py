import os
import collections

import numpy as np
import pandas as pd

from scimmunity.palette import detection as detection_colors
from scimmunity.palette import expanded as expanded_colors
from scimmunity.utils import reorder_obs, explode_str, explode_strs, clean_up_str

def join(x):
    return ';'.join(x.sort_values())

def load_tcr(adata, sample_names, alignments):
    bc2clone = collections.defaultdict(str)
    clone2cdr3 = collections.defaultdict(str)
    clone2tra = collections.defaultdict(str)
    clone2trb = collections.defaultdict(str)
    clone2trav = collections.defaultdict(str)
    clone2traj = collections.defaultdict(str)
    clone2trbv = collections.defaultdict(str)
    clone2trbj = collections.defaultdict(str)

    # load TCRs for each sample
    for s, alignment in zip(sample_names, alignments):
        # get contig annotation with mapping of cell barcode to clonotype 
        contig_file = 'filtered_contig_annotations.csv'
        contig = pd.read_csv(os.path.join(alignment, 'outs', contig_file))
        for i, row in contig.iterrows():
            clone = row.raw_clonotype_id
            if clone !='None':
                bc2clone[f"{s}:{row.barcode}"] = f"{s}:{clone}"
        
        # get amino acid for chain
        clonotype_file = 'clonotypes.csv'
        clonotype = pd.read_csv(os.path.join(alignment, 'outs', clonotype_file))
        # keep track of sample name
        clonotype['clonotype_id'] = str(s) + ':' + clonotype['clonotype_id']
        clone2cdr3.update(clonotype.set_index('clonotype_id')['cdr3s_aa'].to_dict())

        # get consensus annotation with cdr3 and vj gene for each clonotype
        consensus_file = 'consensus_annotations.csv'
        consensus = pd.read_csv(os.path.join(alignment, 'outs', consensus_file))
        # keep track of sample name
        consensus['clonotype_id'] = str(s) + ':' + consensus['clonotype_id']
        
        # TRA chain
        tra_gb = consensus[consensus['chain']=='TRA'].groupby(['clonotype_id'])
        clone2tra.update(tra_gb['cdr3'].apply(join).to_dict())
        clone2trav.update(tra_gb['v_gene'].apply(join).to_dict())
        clone2traj.update(tra_gb['j_gene'].apply(join).to_dict())
        # TRB chain
        trb_gb = consensus[consensus['chain']=='TRB'].groupby(['clonotype_id'])
        clone2trb.update(trb_gb['cdr3'].apply(join).to_dict())
        clone2trbv.update(trb_gb['v_gene'].apply(join).to_dict())
        clone2trbj.update(trb_gb['j_gene'].apply(join).to_dict())

    # map TCRs onto adata
    adata.obs['clonotype'] = adata.obs_names.map(bc2clone)
    adata.obs['cdr3s_aa'] = adata.obs['clonotype'].map(clone2cdr3)
    adata.obs['TRA'] =  adata.obs['clonotype'].map(clone2tra)
    adata.obs['TRAV'] =  adata.obs['clonotype'].map(clone2trav)
    adata.obs['TRAJ'] =  adata.obs['clonotype'].map(clone2traj)
    adata.obs['TRB'] =  adata.obs['clonotype'].map(clone2trb)
    adata.obs['TRBV'] =  adata.obs['clonotype'].map(clone2trbv)
    adata.obs['TRBJ'] =  adata.obs['clonotype'].map(clone2traj)

    # find cells with multiple TRA/TRB's
    adata.obs['multiTRA'] = adata.obs['TRA'].str.contains(';') 
    adata.obs['multiTRB'] = adata.obs['TRB'].str.contains(';') 
    adata.obs['multiTCR'] = adata.obs['multiTRA'] | adata.obs['multiTRB'] 
    return adata

def tcr_detection(adata):
    adata.obs['TCR detection'] = adata.obs['clonotype'].apply(
        lambda x: 'TCR' if x != '' else 'No TCR')
    reorder_obs(adata, 'TCR detection', ['TCR', 'No TCR'], detection_colors)
    return 

def count_tcr(adata, feature, groupby=[], 
    normalize=False, ascending=False, explode=True,
    save=True, out='./'):
    """
    Get TCR counts or frequency.
    Args:
        adata (Anndata object or adata.obs (pandas dataframe))
        feature (str): TCR observational column 
            {'TRA', 'TRB', 'clonotype', 'cdr3s_aa'}
        groupby (list): List of variables to group by (optional)
        normalize (bool): Get normalized frequency per TCR (per group)
        ascending (bool): Sort values per feature (per group)
        explode (True): If True, explode feature by performing str split with ;
        save (bool): Save output as csv if True.
        out (str): Output directory
        
    Return:
        counts (dataframe): counts of unique values in descending order 
            if groupby is provided, counts sorted within group
    """
    if normalize:
        kind = 'frequency'
    else:
        kind = 'count'
    
    df = adata.obs.loc[adata.obs[feature]!='',  :]
    if explode:
        df = explode_str(df, feature, ';')
    if len(groupby) == 0:
        counts = df[feature].value_counts(normalize=normalize, ascending=ascending).to_frame()
    else:
        counts = df.groupby([feature]+groupby[:-1])[groupby[-1]].value_counts(normalize=normalize, ascending=ascending)
        counts = counts.to_frame().rename(columns={groupby[-1]:kind})
    if save:
        name = ''
        for var in groupby:
            name += '_'+clean_up_str(var)
        counts.to_csv(out+feature+name+'_{}.csv'.format(kind))
    return counts


def tcr_size(adata, feature, log=False, normalize=False, inplace=True):
    ''''
    Calcualte feature size/frequency calculated per merged adata
    If feature has multiple sequences, choose the max size.
    Note that log(0) gives -Inf
    Args:
        adata (Anndata object or adata.obs (pandas dataframe))
        feature (str): TCR observational column 
            {'TRA', 'TRB', 'clonotype', 'cdr3s_aa'}
        log (bool): Take log of clone size if True.
        normalize (bool): Get normalized frequency per TCR if True.
        inplace (bool): Update adata.obs 

    '''
    tcrs = adata.obs.loc[:, feature].to_frame(name=feature)
    counts = count_tcr(adata, feature, save=False, normalize=normalize) 
    
    if (feature == 'TRA') or (feature == 'TRB'):    
        tcrs = explode_str(tcrs, feature, ';')
    col = feature
    if log:
        col = 'log ' + col
    if normalize:
        col += ' frequency'
    else:
        col += ' size'

    tcrs[col] = tcrs[feature].map(counts[feature]).fillna(value=0.0)
    size = tcrs.groupby([feature], level=0).max()
    if log:
        size[col] = np.log(size[col])
    if inplace:
        adata.obs[col] = size[col]
    else:
        return size[col]

def tcr_size_per_sample(adata, feature, key='sample_name', 
    log=False, normalize=False, inplace=True):
    ''''
    Calculate feature size/frequency per sample in adata
    If feature has multiple sequences, choose the max size
    Note that log(0) gives -Inf
    Args:
        adata (Anndata object or adata.obs (pandas dataframe))
        feature (str): TCR observational column 
            {'TRA', 'TRB', 'clonotype', 'cdr3s_aa'}
        log (bool): Take log of clone size if True.
        normalize (bool): Get normalized frequency per TCR if True.
        inplace (bool): Update adata.obs 
    '''
    sizes = []
    # calculate TCR size per sample
    for sample in adata.obs[key].unique():
        subset = adata[adata.obs[key]==sample]
        sizes.append(tcr_size(subset, feature, 
            log=log, normalize=normalize, inplace=False))
    # reorder based on original adata
    size = pd.concat(sizes)[adata.obs_names] 
    col = size.name
    if inplace:
        adata.obs[col] = size
    else:
        return size

def expanded_tcr(adata, feature, thresh=10, normalize=False):
    ''''
    Args:
        feature (str): {'TRA', 'TRB', 'clonotype', 'cdr3s_aa'}
    Return:
        Updates adata with Clonal Feature (thresh+)
    '''
    count_col = feature
    if normalize:
        count_col += ' frequency'
        clonal_col = 'Expanded {} (f≥{})'.format(feature, thresh)
    else:
        count_col += ' size'
        clonal_col = 'Expanded {} ({}+)'.format(feature, thresh)
    if count_col not in adata.obs.columns:
        tcr_size_per_sample(adata, feature, normalize=normalize)
    
    greatereq = '≥{}'.format(thresh)
    lesser = '<{}'.format(thresh)
    if normalize:
        greatereq = 'f'+greatereq
        lesser = 'f'+lesser
    adata.obs[clonal_col] = adata.obs[count_col].apply( \
        lambda x: greatereq if x >= thresh else lesser)

    reorder_obs(adata, clonal_col, [lesser, greatereq], expanded_colors)
    return clonal_col
