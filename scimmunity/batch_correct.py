import matplotlib.pyplot as plt

import os
import numpy as np
import scanpy as sc

from scvi.dataset.anndataset import AnnDatasetFromAnnData
from scvi.inference import UnsupervisedTrainer
from scvi.models import VAE
import torch

from scimmunity.utils import mkdir

class scviCorrect():
    def __init__(self, outdir, filtered, scaled, out='batchcorrect', \
        n_epochs=50, use_batches=True, use_cuda=False, n_latent=30, train_size=1.0):
        '''
        Args:
            outdir (str): path to analysis output directory
            filtered (str): path to filtered raw adata input 
            scaled (str): path to scaled adata output
            out (str, optional): output subfolder name
        '''
        self.outdir = outdir
        self.out = os.path.join(outdir, out)
        self.filtered = filtered
        self.scaled = scaled
        # load raw filtered dataset
        self.gene_dataset = self.load_filtered()
        

        # scvi model variables
        self.n_epochs = n_epochs
        self.use_batches = use_batches
        self.use_cuda = use_cuda
        self.n_latent = n_latent
        self.train_size = train_size

        self.vae = self.get_vae()
        self.trainer = self.get_trainer(self.vae, self.train_size)

        # make output folder
        mkdir(self.out)
        return

    def load_filtered(self, subsample=False, n_obs=1000): 
        adata = sc.read(self.filtered)
        if subsample: 
            sc.pp.subsample(adata, n_obs=n_obs)
        dataset = AnnDatasetFromAnnData(adata)
        # reset original gene names because scvi capitalizes gene_names
        dataset.gene_names = np.asarray(adata.var.index.values, dtype="<U64")
        return dataset

    def get_vae(self):
        n_batch = self.gene_dataset.n_batches * self.use_batches
        if self.use_batches:
            vae = VAE(self.gene_dataset.nb_genes, n_batch=n_batch, 
            dispersion='gene-batch', n_layers=2, n_hidden=128, n_latent=self.n_latent)
        else:
            vae = VAE(self.gene_dataset.nb_genes, n_batch=n_batch,
                dispersion='gene', n_layers=2, n_hidden=128, n_latent=self.n_latent)
        return vae

    def get_trainer(self, vae, train_size):
        batch_size = 128
        while self.gene_dataset.nb_cells % batch_size == 1:
            batch_size += 1 # adjust batch size such that no batch has only one cell
        trainer = UnsupervisedTrainer(vae, self.gene_dataset, 
            train_size=train_size, use_cuda=self.use_cuda, 
            frequency=1, data_loader_kwargs={'batch_size':batch_size})
        if self.train_size==1.0:
            trainer._posteriors['test_set'].to_monitor = []
            trainer.metrics_to_monitor = {}
        return trainer

    def run_scvi(self):

        lr=1e-3 # learning rate
        self.trainer.train(n_epochs=self.n_epochs, lr=lr)

        # plot training history (ELBO)
        self.plot_training(self.trainer)
        # save model
        save_name = os.path.join(self.outdir, 
            'no_batch_'*(not self.use_batches)+'scvi.model.pkl')
        torch.save(self.trainer.model.state_dict(), save_name) 
        return
    
    def plot_training(self, trainer):
        fig, ax = plt.subplots()
        for metric in trainer.history:
            values = trainer.history[metric]
            x = np.linspace(0,self.n_epochs,(len(values)))
            ax.plot(x, values, label=metric)
        ax.set_ylabel('metric')
        ax.set_xlabel('epoch')
        ax.legend()
        plt.tight_layout()
        path = os.path.join(self.out, 
            'no_batch_' * (not self.use_batches)+'training_history.png')
        fig.savefig(path, dpi=300)
        plt.close()
        return 

    def load_posterior(self, new_trainer=True):
        # load model
        if new_trainer:
            vae = self.get_vae()
            trainer = self.get_trainer(vae, self.train_size)
            save_name = os.path.join(self.outdir, 
                'no_batch_'*(not self.use_batches)+'scvi.model.pkl')
            trainer.model.load_state_dict(torch.load(save_name)) 
            trainer.model.eval()
        else:
            vae = self.vae
            trainer = self.trainer
        full = trainer.create_posterior(trainer.model, self.gene_dataset, 
            indices=np.arange(len(self.gene_dataset)))
        return full 

    def load_normalized(self):
        adata = sc.read(self.filtered)
        adata.layers['raw'] = adata.X
        # normalize the non-batch corrected data 
        sc.pp.normalize_per_cell(adata, counts_per_cell_after=1e4)
        
        sc.pp.log1p(adata)
        adata.layers['normalized'] = adata.X
        # scale each gene
        adata.layers['scaled'] = sc.pp.scale(adata, copy=True).X

        return adata

    def write_scaled(self, new_trainer=True):
        # load model
        if new_trainer:
            vae = self.get_vae()
            trainer = self.get_trainer(vae, self.train_size)
            save_name = os.path.join(self.outdir, 
                'no_batch_'*(not self.use_batches)+'scvi.model.pkl')
            trainer.model.load_state_dict(torch.load(save_name)) 
            trainer.model.eval()
        else:
            vae = self.vae
            trainer = self.trainer

        full = trainer.create_posterior(trainer.model, self.gene_dataset, 
            indices=np.arange(len(self.gene_dataset)))
        latent, batch_indices, labels = full.sequential().get_latent()
        batch_indices = batch_indices.ravel()

        adata = self.load_normalized()

        # add latent representation
        adata.obsm["X_latent"] = latent

        # add the imputed, scaled, and harmonized values to adata layers
        adata.layers['harmonized'] = full.sequential().get_harmonized_scale(0)
        adata.layers['imputed'] = full.sequential().imputation()
        adata.layers['corrected'] = full.sequential().get_sample_scale()
        # save the log normalized scaled gene expression in adata.raw
        adata.raw = adata
        # set scaled_values (batch-corrected) as X
        adata.X = adata.layers['corrected']
        # save batch corrected adata
        adata.write(self.scaled)
        return 
