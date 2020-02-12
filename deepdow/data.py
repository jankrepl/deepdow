"""Collection of functions related to data."""
from functools import partial

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
        Full features dataset of shape `(n_samples, 1, lookback, n_assets)`.

    y : np.ndarray
        Full targets dataset of shape `(n_samples, horizon, n_assets)`.

    timestamps : None or array-like
        If not None then of shape `(n_samples,)` representing a timestamp for each sample.

    asset_names : None or array-like
        If not None then of shape `(n_assets, )` representing the names of assets.
    """

    def __init__(self, X, y, timestamps=None, asset_names=None):
        """Construct."""
        # checks
        if len(X) != len(y):
            raise ValueError('X and y need to have the same number of samples.')

        if X.shape[-1] != y.shape[-1]:
            raise ValueError('X and y need to have the same number of assets.')

        self.X = X
        self.y = y
        self.timestamps = list(range(len(X))) if timestamps is None else timestamps
        self.asset_names = ['a_{}'.format(i) for i in range(X.shape[-1])] if asset_names is None else asset_names

        # utility
        self.lookback, self.n_assets = X.shape[2:]
        self.horizon = y.shape[1]

    def __len__(self):
        """Compute length."""
        return len(self.X)

    def __getitem__(self, ix):
        """Get item."""
        X_sample = torch.from_numpy(self.X[ix]).to(torch.double)
        y_sample = torch.from_numpy(self.y[ix]).to(torch.double)
        timestamps_sample = self.timestamps[ix]

        return X_sample, y_sample, timestamps_sample, self.asset_names


def collate_uniform(batch, n_assets_range=(5, 10), lookback_range=(1, 20), horizon_range=(3, 15), asset_ixs=None,
                    random_state=None):
    """Create batch of samples.

    Randomly (from uniform distribution) selects assets, lookback and horizon. If `assets` are specified then assets
    kept constant.

    Parameters
    ----------
    batch : list
        List of tuples representing `(X_sample, y_sample)`. Note that the sample dimension is not present and all
        the other dimensions are full (as determined by the dataset).

    n_assets_range : tuple
        Minimum and maximum (only left included) number of assets that are randomly subselected. Ignored if `asset_ixs`
        specified.

    lookback_range : tuple
        Minimum and maximum (only left included) of the lookback that is randomly selected.

    horizon_range : tuple
        Minimum and maximum (only left included) of the horizon that is randomly selected.

    asset_ixs : None or list
        If None, then `n_assets` sampled randomly. If ``list`` then it represents the indices of desired assets - no
        randomness and `n_assets_range` is not used.

    random_state : int or None
        Random state.

    Returns
    -------
    X_batch : torch.Tensor
        Features batch of shape `(batch_size, 1, sampled_lookback, n_sampled_assets)`.

    y_batch : torch.Tensor
        Targets batch of shape `(batch_size, sampled_horizon, n_sampled_assets)`.

    timestamps_batch : list
        List of timestamps (per sample).

    asset_names_batch : list
        List of asset names in the batch (same for each sample).
    """
    # checks
    if asset_ixs is None and not n_assets_range[1] > n_assets_range[0] >= 1:
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
    if asset_ixs is None:
        n_assets = torch.randint(low=n_assets_range[0], high=min(n_assets_max + 1, n_assets_range[1]), size=(1,))[0]
        asset_ixs = torch.multinomial(torch.from_numpy(np.array(range(n_assets_max), dtype='float')), n_assets.item())
    else:
        pass

    # sample lookback
    lookback = torch.randint(low=lookback_range[0], high=min(lookback_max + 1, lookback_range[1]), size=(1,))[0]

    # sample horizon
    horizon = torch.randint(low=horizon_range[0], high=min(horizon_max + 1, horizon_range[1]), size=(1,))[0]

    X_batch = torch.stack([b[0][:, -lookback:, asset_ixs] for b in batch], dim=0)
    y_batch = torch.stack([b[1][:horizon, asset_ixs] for b in batch], dim=0)
    timestamps_batch = [b[2] for b in batch]
    asset_names_batch = batch[0][3][asset_ixs]  # same for the entire batch

    return X_batch, y_batch, timestamps_batch, asset_names_batch


class FlexibleDataLoader(torch.utils.data.DataLoader):
    """Flexible data loader.

    Flexible data loader is well suited for training because one can train the network on different lookbacks, horizons
    and assets. However, it is not well suited for validation.

    Parameters
    ----------
    dataset : InRAMDataset
        Dataset containing the actual data.

    indices : list or None
        List of indices to consider from the provided `dataset` which is inherently ordered. If None then considering
        all the samples.

    n_assets_range : tuple
        Minimum and maximum (only left included) number of assets that are randomly subselected. Ignored if `asset_ixs`
        specified.

    lookback_range : tuple
        Minimum and maximum (only left included) of the lookback that is randomly selected.

    horizon_range : tuple
        Minimum and maximum (only left included) of the horizon that is randomly selected.

    asset_ixs : None or list
        If None, then `n_assets` sampled randomly. If ``list`` then it represents the indices of desired assets - no
        randomness and `n_assets_range` is not used.

    """

    def __init__(self, dataset, indices=None, n_assets_range=(5, 10), lookback_range=(3, 20), horizon_range=(3, 15),
                 asset_ixs=None, **kwargs):
        # checks
        if not (2 <= n_assets_range[0] <= n_assets_range[1] <= dataset.n_assets):
            raise ValueError('Invalid n_assets_range.')

        if not (2 <= lookback_range[0] <= lookback_range[1] <= dataset.lookback):
            raise ValueError('Invalid lookback_range.')

        if not (1 <= horizon_range[0] <= horizon_range[1] <= dataset.horizon):
            raise ValueError('Invalid horizon_range.')

        if indices is not None and not (0 <= min(indices) <= max(indices) <= len(dataset) - 1):
            raise ValueError('The indices our outside of the range of the dataset.')

        self.dataset = dataset
        self.indices = indices if indices is not None else list(range(len(dataset)))
        self.n_assets_range = n_assets_range
        self.lookback_range = lookback_range
        self.horizon_range = horizon_range
        self.asset_ixs = asset_ixs

        super(FlexibleDataLoader, self).__init__(dataset,
                                                 collate_fn=partial(collate_uniform,
                                                                    n_assets_range=n_assets_range,
                                                                    lookback_range=lookback_range,
                                                                    horizon_range=horizon_range,
                                                                    asset_ixs=asset_ixs),
                                                 sampler=torch.utils.data.SubsetRandomSampler(indices),
                                                 batch_sampler=None,
                                                 shuffle=False,
                                                 drop_last=False,
                                                 **kwargs)

    @property
    def mlflow_params(self):
        """Generate dictionary of relevant parameters."""
        return {'lookback_range': self.lookback_range,
                'horizon_range': self.horizon_range,
                'n_assets_range': self.n_assets_range,
                'asset_ixs': self.asset_ixs,
                'batch_size': self.batch_size}


class RigidDataLoader(torch.utils.data.DataLoader):
    """Rigid data loader.

    Rigid data loader is well suited for validation purposes since all horizon, lookback and assets are frozen.
    However, it is not good for training since it enforces the user to choose a single setup.

    Parameters
    ----------
    dataset : torch.utils.data.Dataset
        Instance of our dataset. See ``InRAMDataset`` for more details.

    asset_ixs : list
        Represents indices of considered assets.

    indices : list or None
        List of indices to consider from the provided `dataset` which is inherently ordered. If None then considering
        all the samples.

    lookback : int
        How many time steps do we look back.

    horizon : int
        How many time steps we look forward.

    """

    def __init__(self, dataset, asset_ixs, indices=None, lookback=5, horizon=5, **kwargs):

        if not (2 <= lookback <= dataset.lookback):
            raise ValueError('Invalid lookback_range.')

        if not (1 <= horizon <= dataset.horizon):
            raise ValueError('Invalid horizon_range.')

        if indices is not None and not (0 <= min(indices) <= max(indices) <= len(dataset) - 1):
            raise ValueError('The indices our outside of the range of the dataset.')

        self.dataset = dataset
        self.indices = indices if indices is not None else list(range(len(dataset)))
        self.lookback = lookback
        self.horizon = horizon
        self.asset_ixs = asset_ixs

        super(RigidDataLoader, self).__init__(dataset,
                                              collate_fn=partial(collate_uniform,
                                                                 n_assets_range=None,
                                                                 lookback_range=(lookback, lookback + 1),
                                                                 horizon_range=(horizon, horizon + 1),
                                                                 asset_ixs=asset_ixs),
                                              sampler=torch.utils.data.SubsetRandomSampler(indices),
                                              batch_sampler=None,
                                              shuffle=False,
                                              drop_last=False,
                                              **kwargs)

    @property
    def mlflow_params(self):
        """Generate dictionary of relevant parameters."""
        return {'lookback': self.lookback,
                'horizon': self.horizon,
                'asset_ixs': self.asset_ixs,
                'batch_size': self.batch_size}

# class ValidationDataLoader(torch.utils.data.DataLoader):
#     """Validation data loader.
#
#     The idea is that when validating one needs to fix `lookback`, `horizon` and `n_assets` (or specify them). This
#     way the metrics will be comparable.
#
#     Parameters
#     ----------
#     dataset : torch.utils.data.Dataset
#         Instance of our dataset. See ``InRAMDataset`` for more details.
#
#     indices : list
#         List of indices of the `dataset` to be used. The idea is that one passes the indices of the validation set
#         here.
#
#     lookback : int
#         How many time steps do we look back.
#
#     horizon : int
#         How many time steps we look forward.
#
#     assets : None or list
#         If None, then assets are randomly sampled at each iteration. If ``list`` then indices of considered assets.
#         In this cases `n_iters` needs to be 1 (there is no randomness anymore).
#
#     n_iters : int
#         Number of times to sample assets (`n_assets` of them). If user specifies the actual assets via `assets` then
#         needs to be 1.
#
#     kwargs : dict
#         Additional parameters to be passed into the parent constructor - `torch.utils.data.DataLoader`.
#
#     """
#
#     def __init__(self, dataset, indices, lookbacks=5, horizon=5, assets=None, **kwargs):
#         # checks
#         if n_assets > dataset.n_assets:
#             raise ValueError('Cannot select more assets than in the original dataset - {}.'.format(dataset.n_assets))
#
#         if max(lookbacks) > dataset.lookback:
#             raise ValueError(
#                 'Cannot select a bigger lookback than in the original dataset - {}.'.format(dataset.lookback))
#
#         if horizon > dataset.horizon:
#             raise ValueError(
#                 'Cannot select a bigger horizon than in the original dataset - {}.'.format(dataset.horizon))
#
#         if not (0 <= min(indices) <= max(indices) <= len(dataset) - 1):
#             raise ValueError('The indices our outside of the range of the dataset.')
#
#         self.dataset = dataset
#         self.indices = indices
#         self.n_assets = n_assets
#         self.lookback = lookback
#         self.horizon = horizon
#
#         if assets is not None:
#             if n_iters != 1:
#                 raise ValueError('When assets are provided there can only be one iteration.')
#
#             raise NotImplementedError()
#
#         self.assets = assets
#         self.n_iters = n_iters
#
#         super(ValidationDataLoader, self).__init__(dataset,
#                                                    collate_fn=partial(collate_uniform,
#                                                                       n_assets_range=(n_assets, n_assets + 1),
#                                                                       lookback_range=(lookback, lookback + 1),
#                                                                       horizon_range=(horizon, horizon + 1),
#                                                                       assets=assets),
#                                                    sampler=torch.utils.data.SubsetRandomSampler(indices),
#                                                    batch_sampler=None,
#                                                    shuffle=False,
#                                                    drop_last=False,
#                                                    **kwargs)
