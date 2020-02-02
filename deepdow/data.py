"""Collection of functions related to data."""
import numpy as np
import pandas as pd

from deepdow.utils import PandasChecks


def returns_to_Xy(returns, lookback=10, horizon=10):
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


    Returns
    -------
    X : np.ndarray
        Array of shape `(N, 1, lookback, n_assets)`. Generated out of the entire dataset.

    timestamps : pd.DateTimeIndex
        Index corresponding to the feature matrix `X`.

    y : np.ndarray
        Array of shape `(N, horizon, n_assets)`. Generated out of the entire dataset.

    """
    # check
    PandasChecks.check_no_gaps(returns.index)
    PandasChecks.check_valid_entries(returns)

    n_timesteps = len(returns.index)

    X_list = []
    timestamps_list = []
    y_list = []

    for i in range(lookback, n_timesteps - horizon + 1):
        X_list.append(returns.iloc[i - lookback: i, :].values)
        timestamps_list.append(returns.index[i])
        y_list.append(returns.iloc[i: i + horizon, :].values)

    X = np.array(X_list)
    timestamps = pd.DatetimeIndex(timestamps_list, freq=returns.index.freq)
    y = np.array(y_list)

    return X[:, np.newaxis, :, :], timestamps, y
