import os
import numpy as np
import scanpy as sc
import pandas as pd

from scimmunity.utils import mkdir
from scimmunity.plotting import set_plotting_params

class scAnalysis():
    '''
    Combined analysis using scVI for batch correction.
    Class for organizing analysis input and outputs
    Args:
        name 
        samplesheet
        gtf
        scdir 


    '''
    def __init__(
        self,
        name,
        samplesheet,
        gtf,
        scdir,
        sample_names=[],
        sample_name_col='sample_name',
        # library_ids=[],
        # library_id_col='library_id',
        whole='Whole',
        gex_col='gex',
        vdj_col='vdj',
        metadata_cols=[],
        dpi=300,
        n_epochs=50, 
        use_batches=True, 
        use_cuda=False, 
        n_latent=30, 
        train_size=1.0
        ):

        self.samplesheet = pd.read_csv(samplesheet)

        if not sample_names:
            self.sample_names = self.samplesheet[sample_name_col].tolist()
        else:
            self.sample_names = sample_names
            inds = self.samplesheet[sample_name_col].isin(sample_names)
            self.samplesheet = self.samplesheet[inds].reset_index(drop=True)
        
        self.gtf = gtf
        self.whole = whole
        self.alignments = self.samplesheet[gex_col].tolist()

        # import metadata
        if not metadata_cols:
            metadata_cols = list(self.samplesheet.columns)
            metadata_cols.remove(gex_col)
            metadata_cols.remove(vdj_col)
        self.metadata_cols = metadata_cols

        # scvi arguments
        self.scvi_kwargs = {'n_epochs':n_epochs, 'use_batches':use_batches, 
            'use_cuda':use_cuda, 'n_latent':n_latent, 'train_size':train_size}

        # set analysis name
        self.name = name

        # set output paths
        self.scdir = scdir
        self.outdir = os.path.join(self.scdir, self.name)
        self.filtered = os.path.join(self.outdir, whole, 'filtered.h5ad')
        self.corrected = os.path.join(self.outdir, whole, 'corrected.h5ad')
        self.pkl =  os.path.join(self.outdir, 'scvi.model.pkl')
        self.no_batch_pkl = os.path.join(self.outdir, 'no_batch_scvi.model.pkl')
        
        print('Analysis saved at ' + self.outdir)
        mkdir(self.outdir)

        # set working directory for cache files
        os.chdir(self.outdir)

        set_plotting_params(dpi=dpi)
        
        return 

### Processing data: load raw data, filter, and batch correct ###

    def load_mtx(self, biotypes=["lincRNA", "antisense"], removeMIR=True, 
        filtered=True):
        """
        biotypes (list of str): gene_biotypes to remove in gene count matrix 
        {lincRNA, antisense}
        removeMIR (bool): if True remove genes with MIR prefix
        filtered (bool): use cellranger determined cellular barcodes
        """
        from scimmunity.gene import get_gid2biotype
        gid2biotype = get_gid2biotype(self.gtf)
        adatas = []
        for alignment, sample_name in zip(self.alignments, self.sample_names):
            if filtered:
                mtx_dir = alignment + '/outs/filtered_feature_bc_matrix/'
            else:
                mtx_dir = alignment + '/outs/raw_feature_bc_matrix/'
            adata = sc.read_10x_mtx(mtx_dir, var_names='gene_symbols', cache=True)
            genes_info = pd.read_csv(os.path.join(mtx_dir, 'features.tsv.gz'), \
                sep='\t', header=None)
            adata.obs_names = [sample_name + ":" + bc for bc in adata.obs_names]
            adata.var['gene_names'] =  genes_info.values[:, 1].astype(np.str).ravel()
            adata.var['gene_ids'] =  genes_info.values[:, 0].astype(np.str).ravel()
            adata.var['gene_biotypes'] = adata.var['gene_ids'].map(gid2biotype)

            # remove genes of certain biotypes
            adata._inplace_subset_var(~adata.var['gene_biotypes'].isin(biotypes))
            
            # remove microRNA genes (impossible to be captured by 10x)
            if removeMIR:
                mir_genes = [name for name in adata.var_names if name.startswith('MIR')]
                adata._inplace_subset_var(~adata.var['gene_names'].isin(mir_genes))
            adatas.append(adata)
        return adatas

    def add_metadata_to_adata(self, adata):
        for col in self.metadata_cols:
            mapping = self.samplesheet[col].to_dict()
            adata.obs[col] = adata.obs['batch_indices'].map(mapping)
        return adata
        
    def write_filtered(self, 
        biotypes=['lincRNA', 'antisense'], removeMIR=True, 
        subsample=False, n_obs=2000, 
        min_counts=1000, min_genes=300, min_cells=0, max_mito=20):  

        outdir = os.path.join(self.outdir, "filter")
        mkdir(outdir)
        adatas = self.load_mtx(biotypes=biotypes, removeMIR=removeMIR)
        from scimmunity.filtering import filter_and_merge

        # preprocess, filter, and merge
        adata = filter_and_merge(adatas, self.sample_names, outdir, 
            subsample=subsample, n_obs=n_obs, 
            min_counts=min_counts, min_genes=min_genes, 
            min_cells=min_cells, max_mito=max_mito)

        adata = self.add_metadata_to_adata(adata)

        # write adata
        adata.write(self.filtered)
        return

    def batch_correct(self):
        # batch correct with scvi
        from scimmunity.batch_correct import scviCorrect
        correct = scviCorrect(self.outdir, self.filtered, self.corrected, 
            out='batchcorrect', **self.scvi_kwargs)
        correct.run_scvi()
        return 
    
    def write_scvi_scaled(self):
        # batch correct with scvi
        from scimmunity.batch_correct import scviCorrect
        correct = scviCorrect(self.outdir, self.filtered, self.corrected, 
            out='batchcorrect', **self.scvi_kwargs)
        correct.write_scaled()
        return 

    def process_data(self,
        biotypes=['lincRNA', 'antisense'], removeMIR=True, 
        subsample=False, n_obs=2000, 
        min_counts=1000, min_genes=300, min_cells=0, max_mito=20):

        self.write_filtered(
            biotypes=biotypes, removeMIR=removeMIR, 
            subsample=subsample, n_obs=n_obs, 
            min_counts=min_counts, min_genes=min_genes, 
            min_cells=min_cells, max_mito=max_mito)
        self.batch_correct()
        self.write_scvi_scaled()
        return

#### Initialize dimension reduction object ####

    def reduction(self, **kwargs):
        from scimmunity.reduction import scReduction
        reduction = scReduction(self.outdir, **kwargs)
        return reduction

    # def reduction(self, pca_s=1.0, min_n_pcs=5, n_neighbors=20,  n_jobs=None, scaled=None,\
    #     subset_name=None, subset_cond=dict(), run_reduction=False, regress=False, \
    #     regress_vars={'latent':['percent_mito'],  'corrected':['percent_mito']}, \
    #     plot_clustering_metrics=False, default_rep='pcs', plot_umap=False):
    #     if scaled is None:
    #         scaled = self.scaled
    #     reduced = scVIReduction(self.outdir, self.filtered, scaled, \
    #         pca_s=pca_s, n_neighbors=n_neighbors, n_jobs=n_jobs, \
    #         subset_name=subset_name, subset_cond=subset_cond, plot_umap=plot_umap)
    #     if run_reduction:
    #         reduced.run_all(regress=regress, regress_vars=regress_vars, 
    #             min_n_pcs=min_n_pcs, default_rep=default_rep)
    #         reduced.save()
    #     if plot_clustering_metrics:
    #         reduced.plot_clustering_metrics()
    #     return reduced

#### Quality control scoring ####

    def run_qc(self, reduction):
        from scimmunity.qc import cellcycle, housekeeping, heatshock
        cellcycle(reduction.adata)
        housekeeping(reduction.adata)
        heatshock(reduction.adata)
        return
    
    def score_doublet(self, reduction, sample_key='sample_name', thresh_list=[]):
        from scimmunity.qc import score_doublet_per_sample
        out = reduction.out.replace('reduction', 'doublet')
        mkdir(out)
        score_doublet_per_sample(reduction.adata, sample_key, thresh_list=thresh_list, out=out)
        return 
    
    def call_doublet(self, reduction, sample_key='sample_name', thresh_list=[]):
        from scimmunity.qc import call_doublet_per_sample
        out = reduction.out.replace('reduction', 'doublet')
        mkdir(out)
        call_doublet_per_sample(reduction.adata, sample_key, thresh_list=thresh_list, out=out)
        return 

#### T cell receptor analysis ####

    def tcr(self, reduction, key='sample_name', count_thresh=10, freq_thresh=0.10):
        mappings = tcr.tcr_mapping(self.sample_names, self.vdj_sample_ids, self.vdj_aligndirs, self.irbs)
        tcr.map_adata_tcr(reduction.adata, *mappings)
        tcr.tcr_detection(reduction.adata)
        
        for feature in ['TRB', 'clonotype']:
            tcr.tcr_size_per_sample(reduction.adata, feature, key=key, log=False, normalize=False)
            tcr.tcr_size_per_sample(reduction.adata, feature, key=key, log=False, normalize=True)
            tcr.tcr_size_per_sample(reduction.adata, feature, key=key, log=True, normalize=False)

            key = tcr.expanded_tcr(reduction.adata, feature, thresh=count_thresh, normalize=False)
            key = tcr.expanded_tcr(reduction.adata, feature, thresh=freq_thresh, normalize=True)

        return 