from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import ListedColormap
import seaborn as sns

from .core import mofa_model


def plot_data_overview(
    model: mofa_model,
    colors: Optional[dict[str, str]] = None,
    show_dimensions: bool = True,
):
    """
    Plot data overview

    Parameters
    ----------
    model : mofa_model
        MOFA model
    colors : dict[str, str], optional
        Colors for each view
    show_dimensions : bool, optional
        Show dimensions in the plot

    Returns
    -------
    fig, ax
        Matplotlib figure and axis
    """
    if colors is not None:
        assert set(colors.keys()) == set(
            model.views
        ), "Colors must be provided for all views"
    else:
        palette = sns.color_palette("husl", len(model.views)).as_hex()
        colors = {view: palette[i] for i, view in enumerate(model.views)}

    alpha_values = np.linspace(1.0, 0.5, len(model.groups))
    alphas = {group: alpha_values[i] for i, group in enumerate(model.groups)}

    max_dim = 1
    shapes = {}
    for view in model.views:
        shapes[view] = {}
        for group in model.groups:
            shape = model.data[view][group].shape
            shapes[view][group] = shape
            max_dim = max(max_dim, shape[0])
            max_dim = max(max_dim, shape[1])

    views = {view: x for x, view in enumerate(model.views)}
    groups = {group: y for y, group in enumerate(model.groups)}

    fig, ax = plt.subplots()
    eps = 0.1
    min_dim = 0.1

    cumulative_x, cumulative_y = 0, 0
    total_height = 0
    heights = [0 for _ in groups]
    for view in shapes.keys():
        view_width = 0
        for group in shapes[view].keys():
            shape = shapes[view][group]
            x_offset = views[view]
            y_offset = groups[group]
            width = max(shape[1] * 1.0 / max_dim, min_dim)
            height = max(shape[0] * 1.0 / max_dim, min_dim)
            ax.add_patch(
                Rectangle(
                    (cumulative_x + eps * x_offset, -cumulative_y - eps * y_offset),
                    width,
                    -height,
                    color=colors[view],
                    alpha=alphas[group],
                )
            )
            cumulative_y += height
            view_width = max(view_width, width)

            heights[y_offset] = max(heights[y_offset], height)

            # Add a group label
            if x_offset == 0:
                ax.text(
                    -eps / 2,
                    -cumulative_y - eps * y_offset + height / 2,
                    group,
                    horizontalalignment="right",
                    verticalalignment="center",
                    rotation=90,
                )

            # Add group dimensions
            if show_dimensions and x_offset == len(model.views) - 1:
                ax.text(
                    cumulative_x + view_width + eps * x_offset + eps / 2,
                    -cumulative_y - eps * y_offset + heights[y_offset] / 2,
                    shape[0],
                    horizontalalignment="left",
                    verticalalignment="center",
                    rotation=90,
                )

        # Add a view label
        ax.text(
            cumulative_x + eps * x_offset + view_width / 2,
            eps / 2,
            view,
            horizontalalignment="center",
            verticalalignment="bottom",
        )

        # Add view dimensions
        if show_dimensions:
            ax.text(
                cumulative_x + eps * x_offset + view_width / 2,
                -cumulative_y - eps * len(model.groups) + eps / 2,
                shape[1],
                horizontalalignment="center",
                verticalalignment="top",
            )

        cumulative_x += view_width
        cumulative_y = 0

    ax.set_xlim([0, cumulative_x + eps * len(model.views)])
    ax.set_ylim([-(sum(heights) + eps * len(model.groups)), 0])

    plt.axis("off")
    return fig, ax
