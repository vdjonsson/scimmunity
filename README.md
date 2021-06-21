# scimmunity

scimmunity is a python tool for analyzing single-cell RNAseq data specifically for immune and cancer cells.  It includes preprocessing, dimension reduction, clustering, annotation, visualization, differential expression testing, TCR analysis, and additional integration with existing single cell packages.

# Paper
https://drive.google.com/drive/u/0/folders/1xiL3dBATmDDPIiG2WN-4VZixuxT4eTUH

### Installation

To install the scimmunity, clone the repository and pip install in development mode. 
```
git clone https://github.com/vdjonsson/scimmunity.git
cd scimmunity
pip install --user -e .
```

### Usage
#### Prepare Sample File
Prepare a samplesheet csv file with the following columns for each sample:
- sample_name:
- gex: path to cellranger count (gene) alignment
- vdj (optional): path to cellranger vdj alignment

Additional metadata columns (treatment, patient id, etc.) can be added to annotate each sample.



