import os
import numpy as np
import scanpy as sc
import pandas as pd

from scimmunity.utils import mkdir, clean_up_str
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
            inds = self.samplesheet[sample_name_col].isin(sample_names)
            self.samplesheet = self.samplesheet[inds].reset_index(drop=True)
            self.sample_names = self.samplesheet[sample_name_col].tolist()
        self.gtf = gtf
        self.whole = whole
        self.alignments = self.samplesheet[gex_col].tolist()
        self.vdj_alignments = self.samplesheet[vdj_col].tolist()

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
            adata.obs[col] = adata.obs['batch_indices'].astype(int).map(mapping)
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
    
    def map_obs(self, reduction, mapping, old_key, new_key, palette=None):
        reduction.adata.obs[new_key] = reduction.adata.obs[old_key].apply(
            lambda x:mapping[x]).astype('category')
        if palette:
            colors = [palette[x] for x in reduction.adata.obs[new_key].cat.categories]
            reduction.adata.uns[new_key+'_colors'] = colors
        return 

    def plot_metadata(self, reduction):
        # plot metadata reps
        for obs in self.metadata_cols:
            reduction.plot_obs(obs)
        return 

#### Initialize dimension reduction object ####

    def reduction(self, **kwargs):
        from scimmunity.reduction import scReduction
        reduction = scReduction(self.outdir, **kwargs)
        return reduction

    def run_reduction(self, reduction):
        reduction.run_all()
        reduction.save()
        reduction.plot_clustering_metrics()
        return reduction

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
        score_doublet_per_sample(reduction.adata, sample_key, 
            thresh_list=thresh_list, out=out)
        return 
    
    def call_doublet(self, reduction, sample_key='sample_name', thresh_list=[]):
        from scimmunity.qc import call_doublet_per_sample
        out = reduction.out.replace('reduction', 'doublet')
        mkdir(out)
        call_doublet_per_sample(reduction.adata, sample_key, 
            thresh_list=thresh_list, out=out)
        return 

    def plot_qc(self, reduction):
        metrics = ['phase', 'housekeeping', 'heatshock_score',
            'percent_mito', 'n_counts']
        for metric in metrics:
            reduction.plot_obs(metric)
        return

#### T cell receptor analysis ####

    def add_tcr(self, reduction, 
        key='sample_name', count_thresh=10, freq_thresh=0.10):

        import scimmunity.tcr as tcr

        tcr.load_tcr(reduction.adata, self.sample_names, self.vdj_alignments)
        tcr.tcr_detection(reduction.adata)
        
        for feature in ['TRB', 'clonotype']:
            tcr.tcr_size_per_sample(reduction.adata, feature, 
                key=key, log=False, normalize=False)
            tcr.tcr_size_per_sample(reduction.adata, feature, 
                key=key, log=False, normalize=True)
            tcr.tcr_size_per_sample(reduction.adata, feature, 
                key=key, log=True, normalize=False)

            key = tcr.expanded_tcr(reduction.adata, feature, 
                thresh=count_thresh, normalize=False)
            key = tcr.expanded_tcr(reduction.adata, feature, 
                thresh=freq_thresh, normalize=True)
        return 
    
    def plot_tcr(self, reduction, count_thresh=10, freq_thresh=0.10, 
        count_vmin=-20, freq_vmin=-0.02, log_vmin=-2):
        reduction.plot_bool('TCR detection', ['TCR'], folder='tcr')

        for feature in ['TRB', 'clonotype']:
            reduction.plot_obs('{} size'.format(feature), folder='tcr', 
                color_map='Greens', vmin=count_vmin)
            reduction.plot_obs('{} frequency'.format(feature), folder='tcr', 
                color_map='Greens', vmin=freq_vmin)
            reduction.plot_obs('log {} size'.format(feature), folder='tcr', 
                color_map='Greens', vmin=log_vmin)
            reduction.plot_obs('Expanded {} ({}+)'.format(feature, count_thresh), \
                save_name='expanded_{}_count'.format(feature), folder='tcr')
            reduction.plot_obs('Expanded {} (f≥{})'.format(feature, freq_thresh), \
                save_name='expanded_{}_frequency'.format(feature), folder='tcr')
        return

### Differential Expression ###
    def de(self, reduction, clustering, 
        methods=['t-test', 'wilcoxon'],
        **kwargs):
        """
        Perform differential expression analysis
        Args:
            reduction: scimmunity.reduction.scReduction
            clustering (str or List of str): Keys for annotation of observations
            n_genes: Number of genes to store statistics for in the adata 
            methods: List of methods to use for differential expression 
                {'logreg', 't-test', 'wilcoxon', 't-test_overestim_var'}
        Returns:

        """
        from scimmunity.de import de_layer
        outdir = reduction.out.replace('reduction', 'de')

        if type(clustering) == str:
            clusterings = [clustering]
        elif type(clustering) == list:
            clusterings = clustering

        for clustering in clusterings:
            for method in methods:
                de_layer(reduction.adata, clustering, 
                    method=method, 
                    outdir=outdir, 
                    **kwargs)
        return 

### Annotation ###
    def annotate_pop(self, reduction, clustering, population, 
        plot_reps=False, try_all=True, reps=None,
        plot_reps_dpi=300, figsize=(3,3), **kwargs):
        from scimmunity.annotation import clusterAnnotation
        outdir = reduction.out.replace('reduction', 'annotation')
        annotation = clusterAnnotation(reduction.adata, outdir, clustering, 
            reps=reps, **kwargs)
        annotation.try_annotation(population, plot_reps=plot_reps, 
            try_all=try_all, figsize=figsize, plot_reps_dpi=plot_reps_dpi)
        annotation.set_annotation(population, annotation.pop2phenotype)
        return annotation.adata

    def set_annotation(self, reduction, clustering, population, pop2phenotype, 
        markersets=None):
        from scimmunity.annotation import clusterAnnotation
        outdir = reduction.out.replace('reduction', 'annotation')
        annotation = clusterAnnotation(reduction.adata, outdir, clustering)
        annotation.set_annotation(population, pop2phenotype, markersets=markersets)
        return annotation.adata

    def run_annotation(self, reduction, clustering, population,
        plot_reps=True, try_all=True, reps=None, 
        plot_reps_dpi=300, figsize=(3,3), **kwargs):
        reduction.adata = self.annotate_pop(reduction, clustering, population, 
            plot_reps=plot_reps, try_all=try_all, 
            reps=reps, plot_reps_dpi=plot_reps_dpi, figsize=figsize, **kwargs)
        reduction.plot_obs('Phenotype')
        return 

    def annotate_comp(self, reduction, rep, prefix='', dims=[0,1], offset=1, 
        layer=None, thresh=0.8, markersets=[], 
        gsets=['GO_Biological_Process_2018', 'KEGG_2019_Human', 'WikiPathways_2019_Human']):
        from scvi_analysis.component import corr_rep_gene_dict, corr_rep_gene
        out = os.path.join(reduction.out.replace('reduction', 'annotation'), rep)
        mkdir(out)
        corr_rep_gene(reduction.adata, rep, prefix=prefix, dims=dims, 
            offset=offset, layer=layer, thresh=thresh, gsets=gsets, out=out)
        for markerset in markersets:
            corr_rep_gene_dict(reduction.adata, markerset, rep, prefix=prefix, 
                dims=dims, offset=offset, layer=layer, out=out)
        return 

### Frequency Plot ###

    def plot_frequency(self, reduction, x, y, 
        xrot=0, yrot=45, xorder=None, yorder=None, sort_x=False, sort_y=False, 
        explode=[], swap_axes=True, dropna=False, **kwargs):
        import scimmunity.frequency as freq
        subfolder = f"{clean_up_str(x)}_{clean_up_str(y)}"
        out = os.path.join(reduction.out.replace('reduction', 'frequency'), 
            subfolder)
        mkdir(out)
        df = reduction.adata.obs.copy()
        if len(explode) > 0:
            for cols in explode:
                if type(cols)==str:
                    cols = [cols]
                df = tcr.explode_strs(df, cols, ';')
        if sort_x:
            props = df.groupby(x)[y].count().sort_values(ascending=False)
            sort_order = list(props.index)
            print(sort_order)
            if xorder is not None:
                xorder = [x for x in sort_order if x in xorder]
            else:
                xorder = sort_order
        if sort_y:
            props = df.groupby(y)[x].count().sort_values(ascending=False)
            sort_order = props.index
            if yorder is not None:
                yorder = [y for y in sort_order if y in yorder]
            else:
                yorder = sort_order
        freq.plots(reduction.adata, df, \
            out, x=x, y=y, xrot=xrot, yrot=yrot, xorder=xorder, yorder=yorder, swap_axes=swap_axes, dropna=dropna, **kwargs)
        freq.save_df(x, y, df, 'Frequency',out, dropna=dropna)
        freq.save_df(x, y, df, 'Count',out, dropna=dropna)
        return xorder, yorder

### Infer CNV ###
    def infercnv(self, reduction, ref_groups, annotation_key='Phenotype', 
        sample_key='sample_name', 
        gene_order_dir=None, use_name=True, write=True, 
        cores=4, mem=32, partition='all', time=24):
        import infercnv.pipeline as cnv

        if gene_order_dir is None:
            gene_order_dir = cnv.GENE_ORDER_DIR

        out = reduction.out.replace('reduction', 'infercnv')
        mkdir(out)
        cnv.run_all_samples(reduction.adata, annotation_key, sample_key, ref_groups, \
            self.reference, out=out, gene_order_dir=gene_order_dir, use_name=use_name, write=write, \
            cores=cores, mem=mem, partition=partition, time=time)
        return

### 