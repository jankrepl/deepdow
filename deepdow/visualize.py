"""Collection of utilities for visualization."""

from itertools import cycle

from matplotlib import cm
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt


def portfolio_evolution(weights, always_visible=None, n_displayed_assets=None, n_seconds=3, figsize=(10, 10),
                        colors=None, autopct='%1.1f%%'):
    """Visualize portfolio evolution over time with pie charts.

    Parameters
    ----------
    weights : pd.DataFrame
        The index is a ``pd.DateTimeIndex`` and the columns are different assets.

    always_visible : None or list
        List of assets to always include no matter how big the weights are. Passing None is identical to passing
        an emtpy list - no forcing of any asset.

    n_displayed_assets : int or None
        Number of assets to show. All the remaining assets will be grouped to "others". The selected assets
        are determined via the average weight over all timestamps and additionally via the `always_visible`
        list. If None then all assets are displayed.

    n_seconds : float
        Length of the animation in seconds.

    figsize : tuple
        Size of the figure.

    colors : dict or matplotlib.colors.ListedColormap or None
        If ``dict`` then one can provide a color for each asset present in the columns. Missing assets are assigned
        random colors. If ``matplotlib.colors.Colormap`` then usign a matplotlib colormap. If None then using default
        coloring.

    autopct : str or None
        Formatting of numerical values inside of wedges.

    Returns
    -------
    ani : FuncAnimation
        Animated piechart over the time dimension.

    """
    if 'others' in weights.columns:
        raise ValueError('Cannot use an asset named others since it is user internally.')

    n_timesteps, n_assets = weights.shape
    n_displayed_assets = n_displayed_assets or n_assets

    if not n_displayed_assets <= weights.shape[1]:
        raise ValueError('Invalid number of assets.')

    fps = n_timesteps / n_seconds
    interval = (1 / fps) * 1000

    always_visible = always_visible or []

    if n_displayed_assets <= len(always_visible):
        raise ValueError('Too many always visible assets.')

    top_assets = weights.sum(0).sort_values(ascending=False).index[:n_displayed_assets].to_list()

    for a in reversed(always_visible):
        if a not in top_assets:
            top_assets.pop()
            top_assets = [a] + top_assets

    remaining_assets = [a for a in weights.columns if a not in top_assets]

    new_weights = weights[top_assets].copy()
    new_weights['others'] = weights[remaining_assets].sum(1)

    # create animation
    fig, ax = plt.subplots(figsize=figsize)
    plt.axis('off')

    labels = new_weights.columns

    if colors is None:
        colors_ = None

    elif isinstance(colors, dict):
        colors_ = [colors.get(l, 'black') for l in labels]

    elif isinstance(colors, cm.colors.ListedColormap):
        colors_ = cycle(colors.colors)

    def update(i):
        """Update function."""
        ax.clear()  # pragma: no cover
        ax.axis('equal')  # pragma: no cover
        values = new_weights.iloc[i].values  # pragma: no cover
        ax.pie(values, labels=labels, colors=colors_, autopct=autopct)  # pragma: no cover
        ax.set_title(new_weights.iloc[i].name)  # pragma: no cover

    ani = FuncAnimation(fig,
                        update,
                        frames=n_timesteps,
                        interval=interval)

    return ani
