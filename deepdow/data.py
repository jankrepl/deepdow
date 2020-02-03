"""Collection of functions related to data."""
import numpy as np
import pandas as pd
import torch

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


class InRAMDataset(torch.utils.data.Dataset):
    """Dataset that lives entirely in RAM.

    Parameters
    ----------
    X : np.ndarray
        Full features dataset of shape `(n_assets, 1, lookback, n_assets)`.

    y : np.ndarray
        Full targets dataset of shape `(n_assets, horizon, n_assets)`.

    device : torch.device or None
        Device to which the samples will be assigned. If None then `torch.device('cpu')`.

    """

    def __init__(self, X, y, device=None):
        """Construct."""
        # checks
        if len(X) != len(y):
            raise ValueError('X and y need to have the same number of samples.')

        if X.shape[-1] != y.shape[-1]:
            raise ValueError('X and y need to have the same number of assets.')

        self.X = X
        self.y = y
        self.device = device or torch.device('cpu')

    def __len__(self):
        """Compute length."""
        return len(self.X)

    def __getitem__(self, ix):
        """Get item."""
        X_sample = torch.from_numpy(self.X[ix]).to(self.device)
        y_sample = torch.from_numpy(self.y[ix]).to(self.device)

        return X_sample, y_sample
