""""
Configuration file for single cell analysis
Default values can be overwritten
"""
import pkg_resources

# ANNOTATION GENE DATASETS
# ========================
MARKERSETFILES = {
    'peer_simple_immune': 'azizi_peer_2018_simple_immune.xlsx', 
    'quake_simple_gbm': 'quake_2017_simple_gbm.xlsx',
    'gbm_antigen': 'gbm_antigen.xlsx',
    'verhaak_subtypes': 'wang_verhaak_2017_glioma_subtype.xlsx',
    'neftel_gbm': 'neftel_gbm.xlsx',
    'cd4cd8' : 'forman_tcell_CD4CD8.xlsx',
    'nmeth_myeloid': 'nmeth_myeloid.xlsx',
    'forman_myeloid': 'forman_myeloid.xlsx',
}

MARKERSETS = {k:pkg_resources.resource_filename('scimmunity', 
    f"data/markersets/{MARKERSETFILES[k]}") for k in MARKERSETFILES}

POP2MARKERSETCHOICES= {
    'PBMC':['peer_simple_immune'], 
    'whole_gbm':['peer_simple_immune', 'verhaak_subtypes', 'neftel_gbm', 'quake_simple_gbm'], 
    'Tcell':['yost_tcell', 'tirosh_tcell', 'forman_tcell_v2', 'forman_tcell_v3', 'cd4cd8', 'peer_tcell', 'peer_metabolic', 'darya_tcell'], 
    'CD4':['yost_cd4', 'forman_tcell_v3', 'forman_tcell_v4'], 
    'CD8':['yost_cd8', 'tirosh_tcell'], 
    'Malignant':['neftel_gbm', 'verhaak_subtypes', 'gbm_antigen'], 
    'Monocytes':['murray_macrophage', 'nmeth_myeloid', 'forman_myeloid', 'peer_myeloid', 'haage_microglia', 'peer_metabolic'], 
}

POP2MARKERSETS = {
    'PBMC':['nlj_simple_immune'], 
    'whole_gbm':['peer_simple_immune', 'quake_simple_gbm'], 
    'Tcell':['cd4cd8'], 
    'CD4':['forman_tcell_v4', 'yost_cd8'], 
    'CD8':['yost_cd8'], 
    'Malignant':['neftel_gbm'], 
    'Monocytes':['forman_myeloid'],
}

POP2PHENOTYPE = {
    'PBMC':'phenotype_peer_simple_immune_avg_det_normalized', 
    'whole_gbm':'phenotype_peer_simple_immune_quake_simple_gbm_avg_det_normalized', \
    'Tcell':'phenotype_cd4cd8_avg_det_normalized', 
    'CD4':'phenotype_forman_tcell_v4_avg_det_normalized',
    'CD8':'phenotype_yost_cd8_avg_det_normalized', 
    'Malignant':'phenotype_neftel_gbm_avg_exp_normalized', 
    'Monocytes':'phenotype_forman_myeloid_avg_det_normalized',
}


BULKFILES = {
    'novershtern': 'novershtern/GSE24759_data_cellpop.csv', 
    'calderon' : 'calderon/GSE118165_RNA_gene_abundance.csv'
}

BULKPROFILES = {k:pkg_resources.resource_filename('scimmunity', 
    f"data/markersets/{BULKFILES[k]}") for k in BULKFILES}

POP2BULKPROFILES = {
    'PBMC':['novershtern','calderon'], 
    'whole_gbm':[], 
    'Tcell':['novershtern','calderon'], 
    'CD4':['novershtern','calderon'], 
    'CD8':['novershtern','calderon'], 
    'Malignant':[], 
    'Monocytes':['novershtern','calderon'],
}