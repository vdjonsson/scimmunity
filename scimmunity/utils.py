import os

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