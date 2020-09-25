import os
import re
import numpy as np
def mkdir(path):
    if not os.path.isdir(path):
        os.system('mkdir -p '+path)
    return

def reorder_obs(adata, key, order, colors):
    new_order = []
    new_colors = []
    categories = adata.obs[key].unique() 
    for cat, color in zip(order, colors):
        if cat in categories:
            new_order.append(cat)
            new_colors.append(color)

    adata.obs[key] = adata.obs[key].astype('category').cat.reorder_categories(new_order, ordered=True)
    adata.uns[key+'_colors'] = new_colors
    return 

def explode_str(df, col, sep):
    """
    Explode dataframe by doing str split on a column
    """
    s = df[col]
    i = np.arange(len(s)).repeat(s.str.count(sep) + 1)
    return df.iloc[i].assign(**{col: sep.join(s).split(sep)})

def explode_strs(df, cols, sep):
    """
    Explode dataframe by doing str split on multiple columns
    """
    s = df[cols[0]]
    i = np.arange(len(s)).repeat(s.str.count(sep) + 1)
    return df.iloc[i].assign(**{col: sep.join(df[col]).split(sep) for col in cols})

def clean_up_str(string, replace='_'):
    return re.sub('[^A-Za-z0-9]+', replace, string)
