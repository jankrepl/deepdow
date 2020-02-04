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


def collate_uniform(batch, n_assets_range=(5, 10), lookback_range=(1, 20), horizon_range=(3, 15), random_state=None):
    """Create batch of samples.

    Randomly (from uniform distribution) selects assets, lookback and horizon.

    Parameters
    ----------
    batch : list
        List of tuples representing `(X_sample, y_sample)`. Note that the sample dimension is not present and all
        the other dimensions are full (as determined by the dataset).

    n_assets_range : tuple
        Minimum and maximum (only left included) number of assets that are randomly subselected.

    lookback_range : tuple
        Minimum and maximum (only left included) of the lookback that is randomly selected.

    horizon_range : tuple
        Minimum and maximum (only left included) of the horizon that is randomly selected.

    random_state : int or None
        Random state.

    Returns
    -------
    X_batch : torch.Tensor
        Features batch of shape `(batch_size, 1, sampled_lookback, n_sampled_assets)`.

    y_batch : torch.Tensor
        Targets batch of shape `(batch_size, sampled_horizon, n_sampled_assets)`.

    """
    # checks
    if not n_assets_range[1] > n_assets_range[0] >= 1:
        raise ValueError('Incorrect number of assets range.')

    if not lookback_range[1] > lookback_range[0] >= 1:
        raise ValueError('Incorrect lookback range.')

    if not horizon_range[1] > horizon_range[0] >= 1:
        raise ValueError('Incorrect horizon range.')

    if random_state is not None:
        torch.manual_seed(random_state)

    lookback_max, n_assets_max = batch[0][0].shape[1:]
    horizon_max = batch[0][1].shape[0]

    # sample assets
    n_assets = torch.randint(low=n_assets_range[0], high=min(n_assets_max + 1, n_assets_range[1]), size=(1,))[0]
    asset_idx = torch.multinomial(torch.from_numpy(np.array(range(n_assets_max), dtype='float')), n_assets.item())

    # sample lookback
    lookback = torch.randint(low=lookback_range[0], high=min(lookback_max + 1, lookback_range[1]), size=(1,))[0]

    # sample horizon
    horizon = torch.randint(low=horizon_range[0], high=min(horizon_max + 1, horizon_range[1]), size=(1,))[0]

    X_batch = torch.stack([b[0][:, -lookback:, asset_idx] for b in batch], dim=0)
    y_batch = torch.stack([b[1][:horizon, asset_idx] for b in batch], dim=0)

    return X_batch, y_batch
