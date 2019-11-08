from .core import mofa_model
from .core import get_r2

from typing import Union, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("ticks")
sns.set_palette("Set2")

def plot_weights(model: mofa_model, factor="Factor1", n_features: int = 10, **kwargs):
    """
    Plot loadings for a specific factor

    Parameters
    ----------
    model : mofa_model
        Factor model
    factor : optional
        Factor to use (default is Factor1)
    n_features : optional
        Number of features to label with most positive and most negative loadings
    """
    w = model.get_weights(factors=factor, df=True).sort_values(factor)
    w["rank"] = np.arange(w.shape[0])
    plot = sns.scatterplot(x="rank", y=factor, data=w, **kwargs)
    sns.despine(offset=10, trim=True)
    for line in list(range(-n_features, 0)) + list(range(n_features)):
        plot.text(w.iloc[line]["rank"], w.iloc[line][factor], w.index[line], horizontalalignment='left', size='medium', color='black', weight='regular')


def plot_weights_heatmap(model: mofa_model, factors: Union[int, List[int]] = None,
                         n_features: int = None, w_threshold: float = None, 
                         features_col: pd.DataFrame = None, cmap = None, **kwargs):
    """
    Plot loadings for top features in a heatmap

    Parameters
    ----------
    model : mofa_model
        Factor model
    factors : optional
        Factors to use (all factors in the model by default)
    n_features : optional
        Number of features for each factor by their absolute value (10 by default)
    w_threshold : optional
        Absolute loading threshold for a feature to plot (no threshold by default)
    features_col : optional
        Pandas data frame with index by feature name with the first column 
        containing the colour for every feature
    cmap : optional
        Color map (blue-to-red divergent palette with by default)
    """

    # Set defaults
    n_features_default = 10
    if factors is None:
        factors = list(range(model.nfactors))
    if cmap is None:
        cmap = sns.diverging_palette(240, 10, n=9, as_cmap=True)

    # Fetch weights for the relevant factors
    w = model.get_weights(factors=factors, df=True).rename_axis("feature").reset_index()
    wm = w.melt(id_vars="feature", var_name="factor", value_name="value")
    wm = wm.assign(value_abs = lambda x: x.value.abs())
    wm["factor"] = wm["factor"].astype('category')

    if n_features is None and w_threshold is not None:
        features = wm[wm.value_abs >= w_threshold].feature.unique()
    else:
        if n_features is None:
            n_features = n_features_default
        # Get a subset of features
        wm = wm.sort_values(['factor','value_abs'], ascending=False).groupby('factor')
        if w_threshold is None:
            features = wm.head(n_features).feature.unique()
        else:
            features = wm[wm.value_abs >= w_threshold].head(n_features).feature.unique()

    w = w[w.feature.isin(features)].set_index("feature").T

    col_colors = list(features_col.loc[features,:].iloc[:,0]) if features_col is not None else None

    cg = sns.clustermap(w, cmap=cmap, col_colors = col_colors, **kwargs)
    sns.despine(offset=10, trim=True)
    plt.setp(cg.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)

    return cg


def plot_factors(model: mofa_model, x="Factor1", y="Factor2", hist=False, **kwargs):
    """
    Plot factor values for two factors

    Parameters
    ----------
    model : mofa_model
        Factor model
    x : optional
        Factor to plot along X axis (Factor1 by default)
    y : optional
        Factor to plot along Y axis (Factor2 by default)
    hist : optional
        Boolean value if to add marginal histograms to the scatterplot (jointplot)
    """
    z = model.get_factors(factors = [x, y], df=True)
    sns_plot = sns.jointplot if hist else sns.scatterplot
    sns_plot(x=x, y=y, data=z, **kwargs)
    sns.despine(offset=10, trim=True)

def plot_factor(model: mofa_model, factors: Union[int, List[int]], x="factor", y="value", hue="group", violin=False, **kwargs):
    """
    Plot factor values as stripplots (jitter plots)

    Parameters
    ----------
    model : mofa_model
        Factor model
    factors : optional
        Index of a factor (or indices of factors) to use (all factors by default)
    x : optional
        Variable to plot along X axis (factor identity by default)
    y : optional
        Variable to plot along Y axis (factor value by default)
    hue : optional
        Variable to split & colour dots by (cell group by default)
    violin : optional
        Boolean value if to add violin plots
    """
    z = model.get_factors(factors=factors, df=True)
    z = z.rename_axis("cell").reset_index()
    # Make the table long for plotting
    z = z.melt(id_vars="cell", var_name="factor", value_name="value")
    # Add group information for cells
    z = z.set_index("cell").join(model.get_cells().set_index("cell")).reset_index()

    if violin:
        ax = sns.violinplot(x=x, y=y, hue=hue, data=z, inner=None, color=".9")
    ax = sns.stripplot(x=x, y=y, hue=hue, data=z, dodge=True, **kwargs)
    sns.despine(offset=10, trim=True)

    return ax


def plot_r2(model: mofa_model, factors: Union[int, List[int], str, List[str]] = None, **kwargs):
    """
    Plot R2 values for the model (draft)

    Parameters
    ----------
    model : mofa_model
        Factor model
    """
    r2_df = get_r2(model, factors=factors)
    g = sns.heatmap(r2_df.sort_values("R2").loc[:,["R2"]], **kwargs)

    return g
