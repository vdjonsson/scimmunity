import os
import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
import anndata

from kneed import KneeLocator

from scimmunity.plotting import plot_reps, plot_bool
from scimmunity.clustering import plot_variance_ratio, plot_silhouette_coeff
from scimmunity.utils import mkdir
from scimmunity.annotation import plot_phenotype_markerset

class scReduction():

    def __init__(self, outdir, 
        parent_name='Whole', parent_h5ad='corrected.h5ad',
        subset_name='Whole', subset_h5ad='corrected.h5ad', subset_cond=dict(), 
        subfolder='reduction',
        pca_s=1.0, min_n_pcs=5, n_neighbors=20, n_jobs=None, 
        verify_barcodes=False, 
        neighbors_reps=['pcs', 'latent', 'latent_regressed'],
        default_rep='pcs',
        reductions=['pca', 'umap', 'tsne', 'diffmap'], 
        res_list=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2],
        regress=False, 
        regress_vars={
            'latent':['percent_mito'], 
            'normalized':['n_counts', 'percent_mito', 'S_score', 'G2M_score'],
            'corrected':['percent_mito']
            }
        ):
    
        """
        pcs_s (float): sensitivity for knee detection in determing number of PCs
        n_neighbors (int): number of neighbors for constructing neighborhood graph
        n_jobs (int): number of jobs for regressing out variable
        subset_name (str): name of subset
        subset_cond (dict of str to list): dictionary mapping obs column to list of values to include
        (ex. {'louvain':['0', '1', '2']}) 
        neighbors_reps (list):  Representations used for neighborhood graphs
        default_rep (list):  Default representation for neighborhood graph
        reductions (list): List of dimension reduction to perform
        res_list (list): List of clustering resolution to perform
        """

        self.outdir = outdir
        self.parent = os.path.join(self.outdir, parent_name, parent_h5ad)
        self.subset = os.path.join(self.outdir, subset_name, subset_h5ad)

        self.subset_name = subset_name
        self.subset_cond = subset_cond
        self.subfolder = subfolder

        # set output folders
        self.out = os.path.join(self.outdir, subset_name, subfolder)
        self.prefix = subset_name

        # make output folder
        mkdir(self.out)

        sc.settings.figdir = self.out

        if not os.path.isfile(self.subset):
            # create new subset if none exists
            self.adata = self.create_subset()
        else:
            # load existing subset adata
            self.adata = sc.read(self.subset)
            if verify_barcodes:
            # verify that the subset conditions give the same barcodes
                self.verify_barcodes()

        # Parameters for dimension reduction     
        self.pca_s = pca_s
        self.min_n_pcs = min_n_pcs
        self.n_neighbors = n_neighbors
        self.n_jobs = n_jobs
        self.regress = regress
        self.regress_vars = regress_vars
        self.default_rep = default_rep
        
        # define representations used for constructing neighborhood graph
        self.neighbors_reps = neighbors_reps
        self.neighbors_kwargs_list = [{'use_rep': 'X_'+rep} \
            for rep in self.neighbors_reps]
        self.neighbors_list = [] # store name of neighborhood graphs
        self.all_reps = reductions + \
            [f'{x}_{y}'for x in reductions if x!='pca' for y in neighbors_reps]

        # Clustering resolutions
        self.res_list = res_list

        return
    def update_params(self,**kwargs):
        self.__dict__.update(**kwargs)
        return
    def get_subset_inds(self, adata_parent):
        """
        Generate boolean mask based on subset conditions applied to parent adata.
        """
        subset_inds = np.ones(len(adata_parent), dtype=bool)
        for condition, values in self.subset_cond.items():
            subset_inds *= adata_parent.obs[condition].isin(values)
        return subset_inds

    def create_subset(self):
        adata_parent = sc.read(self.parent)
        subset_inds = self.get_subset_inds(adata_parent)
        adata_parent.obs[self.subset_name] = subset_inds
        if 'X_umap' in adata_parent.obsm:
            sc.pl.umap(adata_parent, color=self.subset_name, 
                save='_'+self.subset_name)
        return adata_parent[subset_inds]

    def verify_barcodes(self):
        """
        Verify that the given subset conditions applied to the parent dataset
        gives the same cell barcodes as the ones in the loaded subset adata.
        """
        adata_parent = sc.read(self.parent)
        subset_inds = self.get_subset_inds(adata_parent)
        barcodes = adata_parent[subset_inds].obs_names.values
        if set(barcodes)!=set(self.adata.obs_names.values):
            raise ValueError('Subset differs from existing subset.')
        return

    def save(self):
        self.adata.write(self.subset)
        return

    ### Regress out variables from matrix or lower dimension representations ###
    
    def regress_rep(self, rep, keys, n_jobs=None):
        X = self.adata.obsm['X_'+rep]
        adata = anndata.AnnData(X)
        for key in keys:
            adata.obs[key] = self.adata.obs[key].values
        sc.pp.regress_out(adata, keys, n_jobs=n_jobs, copy=False)
        self.adata.obsm['X_{}_regressed'.format(rep)] = adata.X
        return 

    def regress_layer(self, layer, keys, n_jobs=None):
        X = self.adata.layers[layer]
        adata = anndata.AnnData(X)
        for key in keys:
            adata.obs[key] = self.adata.obs[key].values
        sc.pp.regress_out(adata, keys, n_jobs=n_jobs, copy=False)
        self.adata.layers['{}_regressed'.format(layer)] = adata.X
        return 

    def run_regress(self):
        for key in self.regress_vars:
            if 'X_' + key in self.adata.obsm:
                self.regress_rep(key, self.regress_vars[key], n_jobs=self.n_jobs)
            elif key in self.adata.layers:
                self.regress_layer(key, self.regress_vars[key], n_jobs=self.n_jobs)
        return
    ### Dimension reduction ###

    def run_pca(self, n_comps=50):
        mkdir(os.path.join(self.out,'pca'))
        sc.settings.figdir = os.path.join(self.out,'pca')
        sc.tl.pca(self.adata, n_comps=n_comps, svd_solver='auto', use_highly_variable=False)
        # arpack can give zero variance explained, so we use auto solver
        sc.settings.figdir = self.out
        sc.pl.pca_variance_ratio(self.adata, log=False, save=True)
        self.set_n_pcs(min_n_pcs=self.min_n_pcs) 
    
    def set_n_pcs(self, min_n_pcs=5):
        # knee detection
        y = np.array(self.adata.uns['pca']['variance_ratio'])
        x = np.arange(len(y))
        kneedle = KneeLocator(x, 1-y, S=self.pca_s, curve='concave', direction='increasing')
        self.n_pcs = max(kneedle.knee+1, min_n_pcs) # change to 1-based

        # plot 
        fig, ax = plt.subplots()
        ax.plot(x+1, y, '-') # change to 1-based
        ax.axvline(x = self.n_pcs, color='red')
        ax.set_xlabel('PC') 
        ax.set_ylabel('PCA variance ratio')
        ax.set_title('n_pcs={}, S={}'.format(self.n_pcs, self.pca_s))
        fig.savefig(os.path.join(self.out, 'pca/pca_variance_ratio_cutoff.png'))
        kneedle.plot_knee()
        plt.savefig(os.path.join(self.out, 'pca/pca_kneedle.png'))

        # add rep with top pcs
        self.adata.obsm['X_pcs'] = self.adata.obsm['X_pca'][:, :self.n_pcs]
        return 
    
    def run_neighbors(self, method='umap'):
        '''
        Args:
            method (str): {'umap', 'gauss'} method for computing connectivities
                use gauss for diffmap 
        '''
        for rep in self.neighbors_reps:
            sc.pp.neighbors(self.adata, n_neighbors=self.n_neighbors, 
                method=method, use_rep='X_'+rep)
            key_added = 'neighbors_{}_{}'.format(method, rep)
            self.adata.uns[key_added] = self.adata.uns['neighbors']
        return

    def run_diffmap(self):
        # Note: use the Gauss kernel method for connectivities
        for rep in self.neighbors_reps:
            # set the precalculated neighbors 
            self.adata.uns['neighbors'] = self.adata.uns['neighbors_gauss_{}'.format(rep)]
            sc.tl.diffmap(self.adata, n_comps=15)
            # get rid of first component
            self.adata.obsm['X_diffmap_'+rep] =  self.adata.obsm['X_diffmap'][:, 1:] 
            self.adata.uns['diffmap_evals_'+rep] =  self.adata.uns['diffmap_evals'][1:]
            if rep == self.default_rep:
                default_diffmap = self.adata.obsm['X_diffmap']
                default_evals = self.adata.uns['diffmap_evals']
        
        # set default diffmap (keep 1st component because of scanpy plotting behaviors)
        self.adata.obsm['X_diffmap'] = default_diffmap
        self.adata.uns['diffmap_evals'] = default_evals 
        return

    def run_tsne(self):
        for rep in self.neighbors_reps:
            sc.tl.tsne(self.adata, use_rep='X_'+rep)
            self.adata.obsm['X_tsne_'+rep] =  self.adata.obsm['X_tsne']
        # set default tsne
        self.adata.obsm['X_tsne'] = self.adata.obsm['X_tsne_'+self.default_rep]
        return
    
    def run_umap(self):
        # use 'umap' method
        for rep in self.neighbors_reps:
            # set the precalculated neighbors 
            self.adata.uns['neighbors'] = self.adata.uns['neighbors_umap_{}'.format(rep)]
            sc.tl.umap(self.adata)
            self.adata.obsm['X_umap_'+rep] =  self.adata.obsm['X_umap'] 
        # set default umap
        self.adata.obsm['X_umap'] = self.adata.obsm['X_umap_'+self.default_rep]
        return

    def run_clustering(self):
        # use 'umap' method
        for rep in self.neighbors_reps:
            # set the precalculated neighbors 
            self.adata.uns['neighbors'] = self.adata.uns['neighbors_umap_{}'.format(rep)]
            #  loop through clustering resolutions
            for res in self.res_list:
                name = self.prefix + 'LouvainRes{}_{}'.format(res, rep)
                sc.tl.louvain(self.adata, resolution=res, key_added=name)
                name = self.prefix + 'LeidenRes{}_{}'.format(res, rep)
                sc.tl.leiden(self.adata, resolution=res, key_added=name)
        return 
    
    def run_all(self):
        self.adata.X = self.adata.layers['normalized']
        if self.regress:
            for key in self.regress_vars:
                if 'X_' + key in self.adata.obsm:
                    self.regress_rep(key, self.regress_vars[key], n_jobs=self.n_jobs)
                if key in self.adata.layers:
                    self.regress_layer(key, self.regress_vars[key], n_jobs=self.n_jobs)
        if 'normalized_regressed' in self.adata.layers:
            self.adata.X = self.adata.layers['normalized_regressed']
        self.run_pca()
        self.run_neighbors(method='gauss')
        self.run_neighbors(method='umap')
        self.run_diffmap()
        self.run_tsne()
        self.run_umap()
        self.run_clustering()
        self.plot_clustering()
        return 

    def run_and_save(self):
        self.run_all()
        self.save()
        self.plot_clustering_metrics()
        return 
    ### Plotting ####
    
    def plot_clustering(self, subfolder=None):
        if not hasattr(self, 'n_pcs'):
            self.set_n_pcs()
        if subfolder is None:
            outdir = self.out
        else:
            outdir = self.outdir + subfolder + '/'
        for neighbors_kwargs in self.neighbors_kwargs_list:
            if 'n_pcs' in neighbors_kwargs:
                rep = 'pcs'
            else:
                rep = list(neighbors_kwargs.values())[0].replace('X_', '')
            
            louvain_names = [self.prefix+'LouvainRes{}_{}'.format(res, rep) for res in self.res_list]
            plot_reps(self.adata, louvain_names, outdir=outdir, save_name=('louvain_'+rep), reps=self.all_reps)
            leiden_names = [self.prefix+'LeidenRes{}_{}'.format(res, rep) for res in self.res_list]
            plot_reps(self.adata, leiden_names, outdir=outdir, save_name=('leiden_'+rep), reps=self.all_reps)
        return
    
    def plot_clustering_metrics(self, reps=['latent_regressed', 'latent', 'pcs']):
        for rep in reps:
            out = os.path.join(self.out, 'clustering', rep)
            mkdir(out)
            plot_variance_ratio(self.adata, self.res_list, X=rep, 
                prefix=self.prefix, rep=rep, out=out)
            plot_silhouette_coeff(self.adata, self.res_list, X=rep, 
                prefix=self.prefix, rep=rep, out=out)
        return

    def plot_obs(self, obs, reps=None, folder='reduction', **kwargs):
        out = os.path.join(self.outdir, self.subset_name, folder)
        if reps is None:
            reps=self.all_reps
        plot_reps(self.adata, obs, outdir=out, reps=reps, **kwargs)
        return
    
    def plot_bool(self, obs, groups, reps=None, folder='reduction', **kwargs):
        if reps is None:
            reps=self.all_reps
        for rep in reps:
            out = os.path.join(self.outdir, self.subset_name, folder, rep)
            plot_bool(self.adata, obs, rep, groups, out=out, **kwargs)        
        return 
    
    def plot_signature_dict(self, groupby, markersets, mode='heatmap', 
        layers=['corrected_regressed', 'normalized_regressed']):
        out = self.out.replace('reduction', 'heatmap')
        mkdir(out)
        plot_phenotype_markerset(self.adata, groupby, markersets, out=out, 
            mode=mode, layers=layers)
        return 