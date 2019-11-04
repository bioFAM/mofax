from .core import mofa_model
from .core import get_r2

from typing import Union, List
import numpy as np
import seaborn as sns

sns.set_style("ticks")
sns.set_palette("Set2")

def plot_weights(model: mofa_model, factor="Factor1", nfeatures: int = 10, **kwargs):
    """
    Plot loadings for a specific factor

    Parameters
    ----------
    model : mofa_model
        Factor model
    factor : optional
        Factor to use (default is Factor1)
    nfeatures : optional
        Number of features to label with most positive and most negative loadings
    """
    w = model.get_weights(factors=factor, df=True).sort_values(factor)
    w["rank"] = np.arange(w.shape[0])
    plot = sns.scatterplot(x="rank", y=factor, data=w, **kwargs)
    sns.despine(offset=10, trim=True)
    for line in list(range(-nfeatures, 0)) + list(range(nfeatures)):
        plot.text(w.iloc[line]["rank"], w.iloc[line][factor], w.index[line], horizontalalignment='left', size='medium', color='black', weight='regular')

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
    factors
        Index of a factor (or indices of factors) to use
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


def plot_r2(model: mofa_model, **kwargs):
    """
    Plot R2 values for the model (draft)

    Parameters
    ----------
    model : mofa_model
        Factor model
    """
    r2_df = get_r2(model)
    sns.heatmap(r2_df.sort_values("R2").loc[:,["R2"]], **kwargs)
