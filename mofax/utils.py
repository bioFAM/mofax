import pandas as pd

#######################
## Utility functions ##
#######################


def umap(data, n_neighbors=10, spread=1, random_state=None, **kwargs):
    """
    Run UMAP on a provided matrix or data frame

    Parameters
    ----------
    data
        Numpy array or Pandas DataFrame with data to run UMAP on (samples in rows)
    n_neighbors : optional
        n_neighbors parameter for UMAP
    spread : optional
        spread parameter for UMAP
    """
    import umap

    embedding = umap.UMAP(
        n_neighbors=n_neighbors, spread=spread, random_state=random_state, **kwargs
    ).fit_transform(data)

    if isinstance(data, pd.DataFrame):
        embedding = pd.DataFrame(embedding)
        embedding.columns = ["UMAP1", "UMAP2"]
        embedding.index = data.index

    return embedding


def padjust_fdr(xs):
    """
    Adjust p-values using the BH procedure
    """
    from scipy.stats import rankdata

    ranked_p_values = rankdata(xs)
    fdr = xs * len(xs) / ranked_p_values
    fdr[fdr > 1] = 1
    return fdr


def padjust_fdr_2d(mx):
    """
    Adjust p-values in a matrix using the BH procedure
    """
    from scipy.stats import rankdata

    ranked_p_values = rankdata(mx).reshape((-1, mx.shape[1]))
    fdr = mx * mx.shape[0] * mx.shape[1] / ranked_p_values
    fdr[fdr > 1] = 1
    return fdr
