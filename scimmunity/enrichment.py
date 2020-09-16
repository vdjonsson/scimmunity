import time
import gseapy

def gsea(genes, description='', out='./', sleeptime=10, \
    gsets=['GO_Biological_Process_2018', 'KEGG_2019_Human', 'WikiPathways_2019_Human']):
    """
    genes (list of str): gene symbols
    description (str): name for enrichment report
    sleeptime (int): length of wait time between each query 
        (overloading server causes connection to be cut)
    """
    for gset in gsets:
        time.sleep(sleeptime)
        gseapy.enrichr(gene_list=genes, description=description, gene_sets=gset, outdir=out)
        
    # gseapy.enrichr(gene_list=genes, description=description, gene_sets=gsets, outdir=out)
    return 