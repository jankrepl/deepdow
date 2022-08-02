"""Collection of utilities and helpers."""
import os
import pathlib

import numpy as np
import pandas as pd


class ChangeWorkingDirectory:
    """Context manager that changes current working directory.

    Parameters
    ----------
    directory : str or pathlib.Path or None
        The new working directory. If None then staying in the current one.

    Attributes
    ----------
    _previous : pathlib.Path
        The original working directory we want to return to after exiting the context manager.

    """

    def __init__(self, directory):
        self.directory = (
            pathlib.Path(directory)
            if directory is not None
            else pathlib.Path.cwd()
        )
        if not self.directory.is_dir():
            raise NotADirectoryError(
                "{} is not a directory".format(str(self.directory))
            )

        self._previous = pathlib.Path.cwd()

    def __enter__(self):
        """Change directory."""
        os.chdir(str(self.directory))

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Go bach to the original directory."""
        os.chdir(str(self._previous))


class PandasChecks:
    """General checks for pandas objects."""

    @staticmethod
    def check_no_gaps(index):
        """Check if a time index has no gaps.

        Parameters
        ----------
        index : pd.DatetimeIndex
            Time index to be checked for gaps.

        Raises
        ------
        TypeError
            If inconvenient type.

        IndexError
            If there is a gap.

        """
        if not isinstance(index, pd.DatetimeIndex):
            raise TypeError("Unsupported type: {}".format(type(index)))

        correct_index = pd.date_range(
            index[0], periods=len(index), freq=index.freq
        )

        if not correct_index.equals(index):
            raise IndexError("Index has gaps.")

    @staticmethod
    def check_valid_entries(table):
        """Check if input table has no nan or +-inf entries.

        Parameters
        ----------
        table : pd.Series or pd.DataFrame
            Input table.

        Raises
        ------
        TypeError
            Inappropriate type of `table`.

        ValueError
            At least one entry invalid.

        """
        if not isinstance(table, (pd.Series, pd.DataFrame)):
            raise TypeError("Unsupported type: {}".format(type(table)))

        if not np.all(np.isfinite(table.values)):
            raise ValueError("There is an invalid entry")

    @staticmethod
    def check_indices_agree(*frames):
        """Check if inputs are pd.Series or pd.DataFrame with same indices / columns.

        Parameters
        ----------
        frames : list
            Elements are either `pd.Series` or `pd.DataFrame`.

        Raises
        ------
        TypeError
            If elements are not `pd.Series` or `pd.DataFrame`.

        IndexError
            If indices/colums do not agree.

        """
        if not all([isinstance(x, (pd.Series, pd.DataFrame)) for x in frames]):
            raise TypeError("Some elements are not pd.Series or pd.DataFrame")

        reference_index = frames[0].index

        for i, f in enumerate(frames):
            if not f.index.equals(reference_index):
                raise IndexError(
                    "The {} entry has wrong index: {}".format(i, f.index)
                )

            if isinstance(f, pd.DataFrame) and not f.columns.equals(
                reference_index
            ):
                raise IndexError(
                    "The {} entry has wrong columns: {}".format(i, f.columns)
                )


def prices_to_returns(prices, use_log=True):
    """Convert prices to returns.

    Parameters
    ----------
    prices : pd.DataFrame
        Rows represent different time points and the columns represent different assets. Note that the columns
        can also be a ``pd.MultiIndex``.

    use_log : bool
        If True, then logarithmic returns are used (natural logarithm). If False, then simple returns.

    Returns
    -------
    returns : pd.DataFrame
        Returns per asset per period. The first period is deleted.

    """
    # checks

    if use_log:
        values = np.log(prices.values) - np.log(prices.shift(1).values)
    else:
        values = (prices.values - prices.shift(1).values) / prices.shift(
            1
        ).values

    return pd.DataFrame(
        values[1:, :], index=prices.index[1:], columns=prices.columns
    )


def returns_to_Xy(returns, lookback=10, horizon=10, gap=0):
    """Create a deep learning dataset (in memory).

    Parameters
    ----------
    returns : pd.DataFrame
        Returns where columns represent assets and rows timestamps. The last row
        is the most recent.

    lookback : int
        Number of timesteps to include in the features.

    horizon : int
        Number of timesteps to inclued in the label.

    gap : int
        Integer representing the number of time periods one cannot act after observing the features.

    Returns
    -------
    X : np.ndarray
        Array of shape `(N, 1, lookback, n_assets)`. Generated out of the entire dataset.

    timestamps : pd.DateTimeIndex
        Index corresponding to the feature matrix `X`.

    y : np.ndarray
        Array of shape `(N, 1, horizon, n_assets)`. Generated out of the entire dataset.

    """
    n_timesteps = len(returns.index)

    if lookback >= n_timesteps - horizon - gap + 1:
        raise ValueError("Not enough timesteps to extract X and y.")

    X_list = []
    timestamps_list = []
    y_list = []

    for i in range(lookback, n_timesteps - horizon - gap + 1):
        X_list.append(returns.iloc[i - lookback : i, :].values)
        timestamps_list.append(returns.index[i - 1])
        y_list.append(returns.iloc[i + gap : i + gap + horizon, :].values)

    X = np.array(X_list)
    timestamps = pd.DatetimeIndex(timestamps_list, freq=returns.index.freq)
    y = np.array(y_list)

    return X[:, np.newaxis, :, :], timestamps, y[:, np.newaxis, :, :]


def raw_to_Xy(
    raw_data,
    lookback=10,
    horizon=10,
    gap=0,
    freq="B",
    included_assets=None,
    included_indicators=None,
    use_log=True,
):
    """Convert raw data to features.

    Parameters
    ----------
    raw_data : pd.DataFrame
        Rows represents different timestamps stored in index. Note that there can be gaps. Columns are pd.MultiIndex
        with the zero level being assets and the first level indicator.

    lookback : int
        Number of timesteps to include in the features.

    horizon : int
        Number of timesteps to included in the label.

    gap : int
        Integer representing the number of time periods one cannot act after observing the features.

    freq : str
        Periodicity of the data.

    included_assets : None or list
        Assets to be included. If None then all available.

    included_indicators : None or list
        Indicators to be included. If None then all available.

    use_log : bool
        If True, then logarithmic returns are used (natural logarithm). If False, then simple returns.

    Returns
    -------
    X : np.ndarray
        Feature array of shape `(n_samples, n_indicators, lookback, n_assets)`.

    timestamps : pd.DateTimeIndex
        Per row timestamp of shape length `n_samples`.

    y : np.ndarray
        Targets arra of shape `(n_samples, n_indicators, horizon, n_assets)`.

    asset_names : list
        Names of assets.

    indicators : list
        List of indicators.
    """
    if freq is None:
        raise ValueError("Frequency freq needs to be specified.")

    asset_names = (
        included_assets
        if included_assets is not None
        else raw_data.columns.levels[0].to_list()
    )
    indicators = (
        included_indicators
        if included_indicators is not None
        else raw_data.columns.levels[1].to_list()
    )

    index = pd.date_range(
        start=raw_data.index[0], end=raw_data.index[-1], freq=freq
    )

    new = pd.DataFrame(raw_data, index=index).ffill().bfill()

    to_exclude = []
    for a in asset_names:
        is_valid = np.all(np.isfinite(new[a])) and np.all(new[a] > 0)
        if not is_valid:
            to_exclude.append(a)

    asset_names = sorted(list(set(asset_names) - set(to_exclude)))

    absolute = new.iloc[:, new.columns.get_level_values(0).isin(asset_names)][
        asset_names
    ]  # sort
    absolute = absolute.iloc[
        :, absolute.columns.get_level_values(1).isin(indicators)
    ]

    returns = prices_to_returns(absolute, use_log=use_log)

    X_list = []
    y_list = []
    for ind in indicators:
        X, timestamps, y = returns_to_Xy(
            returns.xs(ind, axis=1, level=1),
            lookback=lookback,
            horizon=horizon,
            gap=gap,
        )
        X_list.append(X)
        y_list.append(y)

    X = np.concatenate(X_list, axis=1)
    y = np.concatenate(y_list, axis=1)

    return X, timestamps, y, asset_names, indicators
