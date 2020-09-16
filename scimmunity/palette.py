import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib.colors

def split_paired(paired):
    return [*paired[1::2],*paired[0::2]]

# all color brewer qualitative colors
set1 = sns.color_palette('Set1', n_colors=9)
pastel1 = sns.color_palette('Pastel1', n_colors=9)

set2 = sns.color_palette('Set2', n_colors=8)
pastel2 = sns.color_palette('Pastel2', n_colors=8)
dark2 = sns.color_palette('Dark2', n_colors=8)

set3 = sns.color_palette('Set3', n_colors=12)

paired = sns.color_palette('Paired', n_colors=12)
paired_desat = sns.color_palette('Paired', n_colors=12, desat=0.5)

colorblind = sns.color_palette('colorblind', n_colors=6)
colorblind_desat = sns.color_palette('colorblind', n_colors=6, desat=0.5)

# clustering
cluster_palette1 = set2 + colorblind_desat
cluster_palette2 = colorblind + colorblind_desat
cluster_palette3 = set2 + dark2 
cluster_palette = cluster_palette3

# pre- post- treatment 
blues_2 = sns.color_palette('Blues', n_colors=2)

# cell cycle (phase) 
# order: G1, S, G2/M
orrd_3 = sns.color_palette('OrRd', n_colors=3)

# gene expression colormap
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("GrYlOrRd", ["lightgray"]+sns.color_palette('YlOrRd', 3))
plt.register_cmap(cmap=cmap)
gene_cmap = "GrYlOrRd"
gene_heatmap = 'RdBu_r'

# compartment
csf = sns.color_palette('Purples', n_colors=3)[-1]
blood = sns.color_palette('Reds', n_colors=3)[-1]
cart = sns.color_palette('Greens', n_colors=3)[-1]
cart_it = sns.color_palette('Greens', n_colors=5)[-1]
cart_til = sns.color_palette('PuBuGn', n_colors=3)[-1]

# antigens
antigens = set2[2:5]

# map feature to palette
feature2color = {'treatment': blues_2, 'cellcycle':orrd_3, \
    'cluster_palette':cluster_palette3, 'csf':csf, 'blood':blood, 'cart':cart, \
    'cart_it':cart_it, 'cart_til':cart_til}
