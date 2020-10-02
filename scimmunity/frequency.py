import os

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc

def map_obs(self, mapping, old_key, new_key):
    self.adata.obs[new_key] = self.adata.obs[old_key].apply(lambda x:mapping[x])
    return

def save_df(x, y, df, kind, out, dropna=False):
    if kind=='Frequency':
        props = df.groupby(x)[y].value_counts(dropna=dropna, normalize=True).unstack()
    elif kind=='Count':
        props = df.groupby(x)[y].value_counts(dropna=dropna, normalize=False).unstack()
    
    props.to_csv('{}/{}_{}_{}.csv'.format(out, x,y,kind))
    return

def plot_bar(adata, x, y, df, kind, out, ax=None, stacked=True, colors=None, \
    rotation=0, xorder=None, yorder=None, cmap=None, 
    dropna=False, plotna=False, legend=True, **kwargs):

    if kind=='Frequency':
        props = df.groupby(x)[y].value_counts(dropna=dropna, normalize=True).unstack()
    elif kind=='Count':
        props = df.groupby(x)[y].value_counts(dropna=dropna, normalize=False).unstack()

    if xorder is not None:
        props = props.loc[xorder]
    if yorder is not None:
        props = props.loc[:, yorder]
    if not plotna:
        props = props.loc[:, props.columns.notnull()]

    # stacked barplot
    with sns.axes_style('white'):
        if cmap is not None:
            colors = [cmap[col] for col in props.columns]
        elif y+'_colors' in adata.uns:
            palette = adata.uns[y+'_colors']
            categories = adata.obs[y].cat.categories
            cmap = {cat:color for cat, color in zip(categories, palette)}
            colors = [cmap[cat] for cat in props.columns]
        if stacked:
            ax = props.plot(kind='bar', stacked='True', color=colors, ax=ax, **kwargs)
        else:
            ax = props.plot(kind='bar', stacked=None, color=colors, ax=ax, **kwargs)
        if legend:
            ax.legend(bbox_to_anchor=(1.1,0.5))
        else:
            ax.get_legend().remove()

        sns.despine()
        ax.set_ylabel(kind)
        ax.set_xlabel(x)
        ax.tick_params(axis='x', which='major', labelrotation=rotation)
        fig = ax.get_figure()
        fig.savefig('{}/{}_{}_{}{}.png'.format(out, x,y,kind, '_stacked'*stacked), \
            bbox_inches='tight', transparent=True)
        plt.close()
    return 

def plot_bar_tidy(x, y, df, kind, out, ax=None, rotation=0, order=None, palette=None, dropna=False):
    if order is None:
        if kind=='Frequency':
            props = df.groupby(x)[y].value_counts(dropna=dropna, normalize=True).unstack()
        elif kind=='Count':
            props = df.groupby(x)[y].value_counts(dropna=dropna, normalize=False).unstack()
    else:
        if kind=='Frequency':
            props = df.groupby(x)[y].value_counts(dropna=dropna, normalize=True).unstack().loc[order]
        elif kind=='Count':
            props = df.groupby(x)[y].value_counts(dropna=dropna, normalize=False).unstack().loc[order]
    props_tidy = props.reset_index().melt(id_vars=x)

    
    with sns.axes_style('white'):
        ax = sns.barplot(data=props_tidy, hue=x, x=y, y='value', ax=ax, palette=palette)
        sns.despine()
        plt.ylabel(kind)
        plt.xlabel(y)
        plt.xticks(rotation=rotation)
        fig = ax.get_figure()
        fig.savefig('{}/{}_{}_{}_tidy.png'.format(out, x,y,kind), bbox_inches='tight', transparent=True)
        plt.close()
    return

def plots(adata, df, out, x='treatment', y='phenotype', xrot=0, yrot=45, 
    xorder=None, yorder=None, cmaps=[None, None], figsize=None, 
    swap_axes=False,  dropna=False, legend=True, **kwargs):
    cmap_x, cmap_y = cmaps
    args = {'rotation':xrot, 'xorder':xorder, 'yorder':yorder, 'cmap':cmap_y, 
        'figsize':figsize, 'dropna':dropna, 'legend':legend}
    plot_bar(adata, x, y, df, 'Frequency', out, **args, **kwargs)
    plot_bar(adata, x, y, df, 'Count', out, **args, **kwargs)
    plot_bar(adata, x, y, df, 'Frequency', out, stacked=False, **args, **kwargs)
    plot_bar(adata, x, y, df, 'Count', out, stacked=False, **args, **kwargs)

    if cmap_x is not None:
        palette = cmap_x
    elif x+"_colors" in adata.uns:
        palette = {cat:color for cat,color in zip(adata.obs[x].cat.categories, adata.uns[x+'_colors'])}
    else:
        palette = None
    
    if swap_axes:
        plt.figure(figsize=figsize)
        plot_bar_tidy(x, y, df, 'Frequency', out, rotation=yrot, order=xorder, palette=palette)
        plt.figure(figsize=figsize)
        plot_bar_tidy(x, y, df, 'Count', out, rotation=yrot, order=xorder, palette=palette)

        plot_bar(adata, y, x, df, 'Count', out, rotation=yrot, xorder=yorder, yorder=xorder, cmap=cmap_x, figsize=figsize, **kwargs)
        plot_bar(adata, y, x, df, 'Count', out, stacked=False, rotation=yrot, xorder=yorder, yorder=xorder, cmap=cmap_x, figsize=figsize, **kwargs)
        plot_bar(adata, y, x, df, 'Frequency', out, rotation=yrot, xorder=yorder, yorder=xorder, cmap=cmap_x, figsize=figsize, **kwargs) 
    return
