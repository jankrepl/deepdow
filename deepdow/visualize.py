"""Collection of utilities for visualization."""

from itertools import cycle

from matplotlib import cm
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

from .benchmarks import Benchmark
from .data import RigidDataLoader
from .losses import Loss, portfolio_cumulative_returns


def generate_metrics_table(
    benchmarks, dataloader, metrics, device=None, dtype=None
):
    """Generate metrics table for all benchmarks.

    Parameters
    ----------
    benchmarks : dict
        Dictionary where keys are benchmark names and values are instances of `Benchmark` (possible
        also `torch.nn.Network`).

    dataloader : deepdow.data.RigidDataLoader
        Dataloader that we will fully iterate over.

    metrics : dict
        Keys are metric names and values are instances of `deepdow.loss.Loss` representing. They
        all have the logic the lower the better.

    device : torch.device or None
        Device to be used. If not specified defaults to `torch.device('cpu')`.

    dtype : torch.dtype or None
        Dtype to be used. If not specified defaults to `torch.float`.

    Returns
    -------
    metrics_table : pd.DataFrame
        Table with the following columns - 'metric', 'timestamp', 'benchmark' and 'value'.

    """
    # checks
    if not all(isinstance(bm, Benchmark) for bm in benchmarks.values()):
        raise TypeError(
            "The values of benchmarks need to be of type Benchmark"
        )

    if not isinstance(dataloader, RigidDataLoader):
        raise TypeError("The type of dataloader needs to be RigidDataLoader")

    if not all(isinstance(metric, Loss) for metric in metrics.values()):
        raise TypeError("The values of metrics need to be of type Loss")

    device = device or torch.device("cpu")
    dtype = dtype or torch.float

    for bm in benchmarks.values():
        if isinstance(bm, torch.nn.Module):
            bm.eval()

    all_entries = []

    for batch_ix, (X_batch, y_batch, timestamps, _) in enumerate(dataloader):
        # Get batch
        X_batch, y_batch = X_batch.to(device).to(dtype), y_batch.to(device).to(
            dtype
        )
        for bm_name, bm in benchmarks.items():
            weights = bm(X_batch)
            for metric_name, metric in metrics.items():
                metric_per_s = metric(weights, y_batch).detach().cpu().numpy()
                all_entries.append(
                    pd.DataFrame(
                        {
                            "timestamp": timestamps,
                            "benchmark": bm_name,
                            "metric": metric_name,
                            "value": metric_per_s,
                        }
                    )
                )

    metrics_table = pd.concat(all_entries)

    return metrics_table


def generate_cumrets(
    benchmarks,
    dataloader,
    device=None,
    dtype=None,
    returns_channel=0,
    input_type="log",
    output_type="log",
):
    """Generate cumulative returns over the horizon for all benchmarks.

    Parameters
    ----------
    benchmarks : dict
        Dictionary where keys are benchmark names and values are instances of `Benchmark` (possible
        also `torch.nn.Network`).

    dataloader : deepdow.data.RigidDataLoader
        Dataloader that we will fully iterate over.

    device : torch.device or None
        Device to be used. If not specified defaults to `torch.device('cpu')`.

    dtype : torch.dtype or None
        Dtype to be used. If not specified defaults to `torch.float`.

    returns_channel : int
        What channel in `y` represents the returns.

    input_type : str, {'log', 'simple'}
        What type of returns are we dealing with in `y`.

    output_type : str, {'log', 'simple'}
        What type of returns are we dealing with in the output.

    Returns
    -------
    cumrets_dict : dict
       Keys are benchmark names and values are ``pd.DataFrame`` with index equal to timestamps,
       columns horizon timesteps and values cumulative returns.
    """
    # checks
    if not all(isinstance(bm, Benchmark) for bm in benchmarks.values()):
        raise TypeError(
            "The values of benchmarks need to be of type Benchmark"
        )

    if not isinstance(dataloader, RigidDataLoader):
        raise TypeError("The type of dataloader needs to be RigidDataLoader")

    device = device or torch.device("cpu")
    dtype = dtype or torch.float

    all_entries = {}
    for bm_name, bm in benchmarks.items():
        all_entries[bm_name] = []
        if isinstance(bm, torch.nn.Module):
            bm.eval()

    for batch_ix, (X_batch, y_batch, timestamps, _) in enumerate(dataloader):
        # Get batch
        X_batch, y_batch = X_batch.to(device).to(dtype), y_batch.to(device).to(
            dtype
        )
        for bm_name, bm in benchmarks.items():
            weights = bm(X_batch)
            cumrets = portfolio_cumulative_returns(
                weights,
                y_batch[:, returns_channel, ...],
                input_type=input_type,
                output_type=output_type,
            )

            all_entries[bm_name].append(
                pd.DataFrame(cumrets.detach().cpu().numpy(), index=timestamps)
            )

    cumrets_dict = {
        bm_name: pd.concat(entries).sort_index()
        for bm_name, entries in all_entries.items()
    }

    return cumrets_dict


def plot_metrics(metrics_table):
    """Plot performance of all benchmarks for all metrics.

    Parameters
    ----------
    metrics_table : pd.DataFrame
        Table with the following columns - 'metric', 'timestamp', 'benchmark' and 'value'.

    Returns
    -------
    return_ax : 'matplotlib.axes._subplots.AxesSubplot
        Axes with number of subaxes equal to number of metrics.

    """
    all_metrics = metrics_table["metric"].unique()
    n_metrics = len(all_metrics)

    _, axs = plt.subplots(n_metrics)

    for i, metric_name in enumerate(all_metrics):
        df = pd.pivot_table(
            metrics_table[metrics_table["metric"] == metric_name],
            values="value",
            columns="benchmark",
            index="timestamp",
        ).sort_index()
        df.plot(ax=axs[i])
        axs[i].set_title(metric_name)

    plt.tight_layout()

    return axs


def generate_weights_table(network, dataloader, device=None, dtype=None):
    """Generate a pd.DataFrame with predicted weights over all indices.

    Parameters
    ----------
    network : deepdow.benchmarks.Benchmark
        Any benchmark that is performing portfolio optimization via the `__call__` magic method.

    dataloader : deepdow.data.RigidDataLoader
        Dataloader that we will fully iterate over.

    device : torch.device or None
        Device to be used. If not specified defaults to `torch.device('cpu')`.

    dtype : torch.dtype or None
        Dtype to be used. If not specified defaults to `torch.float`.

    Returns
    -------
    weights_table : pd.DataFrame
        Index represents the timestep and column are different assets. The values are allocations.
    """
    if not isinstance(network, Benchmark):
        raise TypeError("The network needs to be an instance of a Benchmark")

    if not isinstance(dataloader, RigidDataLoader):
        raise TypeError(
            "The network needs to be an instance of a RigidDataloader"
        )

    device = device or torch.device("cpu")
    dtype = dtype or torch.float

    if isinstance(network, torch.nn.Module):
        network.to(device=device, dtype=dtype)
        network.eval()

    all_batches = []
    all_timestamps = []

    for X_batch, _, timestamps, _ in dataloader:
        X_batch = X_batch.to(device=device, dtype=dtype)
        weights_batch = network(X_batch).cpu().detach().numpy()

        all_batches.append(weights_batch)
        all_timestamps.extend(timestamps)

    weights = np.concatenate(all_batches, axis=0)
    asset_names = [
        dataloader.dataset.asset_names[asset_ix]
        for asset_ix in dataloader.asset_ixs
    ]

    weights_table = pd.DataFrame(
        weights, index=all_timestamps, columns=asset_names
    )

    return weights_table.sort_index()


def plot_weight_anim(
    weights,
    always_visible=None,
    n_displayed_assets=None,
    n_seconds=3,
    figsize=(10, 10),
    colors=None,
    autopct="%1.1f%%",
):
    """Visualize portfolio evolution over time with pie charts.

    Parameters
    ----------
    weights : pd.DataFrame
        The index is a represents the timestamps and the columns are asset names. Values are
        weights.

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
    if "others" in weights.columns:
        raise ValueError(
            "Cannot use an asset named others since it is user internally."
        )

    n_timesteps, n_assets = weights.shape
    n_displayed_assets = n_displayed_assets or n_assets

    if not n_displayed_assets <= weights.shape[1]:
        raise ValueError("Invalid number of assets.")

    fps = n_timesteps / n_seconds
    interval = (1 / fps) * 1000

    always_visible = always_visible or []

    if n_displayed_assets <= len(always_visible):
        raise ValueError("Too many always visible assets.")

    top_assets = (
        weights.sum(0)
        .sort_values(ascending=False)
        .index[:n_displayed_assets]
        .to_list()
    )

    for a in reversed(always_visible):
        if a not in top_assets:
            top_assets.pop()
            top_assets = [a] + top_assets

    remaining_assets = [a for a in weights.columns if a not in top_assets]

    new_weights = weights[top_assets].copy()
    new_weights["others"] = weights[remaining_assets].sum(1)

    # create animation
    fig, ax = plt.subplots(figsize=figsize)
    plt.axis("off")

    labels = new_weights.columns

    if colors is None:
        colors_ = None

    elif isinstance(colors, dict):
        colors_ = [colors.get(label, "black") for label in labels]

    elif isinstance(colors, cm.colors.ListedColormap):
        colors_ = cycle(colors.colors)

    def update(i):
        """Update function."""
        ax.clear()  # pragma: no cover
        ax.axis("equal")  # pragma: no cover
        values = new_weights.iloc[i].values  # pragma: no cover
        ax.pie(
            values, labels=labels, colors=colors_, autopct=autopct
        )  # pragma: no cover
        ax.set_title(new_weights.iloc[i].name)  # pragma: no cover

    ani = FuncAnimation(fig, update, frames=n_timesteps, interval=interval)

    return ani


def plot_weight_heatmap(
    weights,
    add_sum_column=False,
    cmap="YlGnBu",
    ax=None,
    always_visible=None,
    asset_skips=1,
    time_skips=1,
    time_format="%d-%m-%Y",
    vmin=0,
    vmax=1,
):
    """Create a heatmap out of the weights.

    Parameters
    ----------
    weights : pd.DataFrame
        The index is a represents the timestamps and the columns are asset names. Values are
        weights.

    add_sum_column : bool
        If True, appending last colum representing the sum of all assets.

    cmap : str
        Matplotlib cmap.

    always_visible : None or list
        List of assets that are always annotated. Passing None is identical to passing
        an emtpy list - no forcing of any asset. Overrides the `asset_skips=None`.

    asset_skips : int or None
        Displaying every `asset_skips` asset names. If None then asset names not shown.

    time_skips : int or None
        Displaying every `time_skips` time steps. If None then time steps not shown.

    time_format : None or str
        If None, then no special formatting applied. Otherwise a string that determines the
        formatting of the ``datetime``.

    vmin, vmax : float
        Min resp. max of the colorbar.

    Returns
    -------
    return_ax : 'matplotlib.axes._subplots.AxesSubplot
        Axes with a heatmap plot.

    """
    displayed_table = weights
    always_visible = always_visible or []

    if add_sum_column:
        if "sum" in displayed_table.columns:
            raise ValueError(
                "The weights dataframe already contains the sum column."
            )

        displayed_table = displayed_table.copy()
        displayed_table["sum"] = displayed_table.sum(axis=1)
        always_visible.append("sum")

    xlab = [
        str(c)
        if ((asset_skips and i % asset_skips == 0) or c in always_visible)
        else ""
        for i, c in enumerate(weights.columns)
    ]

    def formatter(x):
        """Format row index."""
        if time_format is not None:
            return x.strftime(time_format)
        else:
            return x

    ylab = [
        formatter(ix) if (time_skips and i % time_skips == 0) else ""
        for i, ix in enumerate(weights.index)
    ]

    return_ax = sns.heatmap(
        displayed_table,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        ax=ax,
        xticklabels=xlab,
        yticklabels=ylab,
    )

    return_ax.xaxis.set_ticks_position("top")
    return_ax.tick_params(axis="x", rotation=75, length=0)
    return_ax.tick_params(axis="y", rotation=0, length=0)

    return return_ax
