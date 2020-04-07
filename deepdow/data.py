"""Collection of functions related to data."""
from functools import partial

import numpy as np
import torch


def scale_features(X, approach='standard'):
    """Scale feature matrix.

    Parameters
    ----------
    X : torch.Tensor
        Tensor of shape (n_samples, n_channels, lookback, n_assets). Unscaled

    approach : str, {'standard', 'percent'}
        How to scale features.

    Returns
    -------
    X_scaled : torch.tensor
        Tensor of shape (n_samples, n_channels, lookback, n_assets). Scaled.
    """
    n_samples, n_channels, lookback, n_assets = X.shape

    if approach == 'standard':
        means = X.mean(dim=[2, 3])  # for each sample and each channel a mean is computed (over lookback and assets)
        stds = X.std(dim=[2, 3]) + 1e-6  # for each sample and each channel a std is computed (over lookback and assets)

        means_rep = means.view(n_samples, n_channels, 1, 1).repeat(1, 1, lookback, n_assets)
        stds_rep = stds.view(n_samples, n_channels, 1, 1).repeat(1, 1, lookback, n_assets)

        X_scaled = (X - means_rep) / stds_rep

    elif approach == 'percent':
        X_scaled = X * 100

    else:
        raise ValueError('Invalid scaling approach {}'.format(approach))

    return X_scaled


class InRAMDataset(torch.utils.data.Dataset):
    """Dataset that lives entirely in RAM.

    Parameters
    ----------
    X : np.ndarray
        Full features dataset of shape `(n_samples, n_input_channels, lookback, n_assets)`.

    y : np.ndarray
        Full targets dataset of shape `(n_samples, n_input_channels, horizon, n_assets)`.

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

        if X.shape[1] != y.shape[1]:
            raise ValueError('X and y need to have the same number of input channels.')

        if X.shape[-1] != y.shape[-1]:
            raise ValueError('X and y need to have the same number of assets.')

        self.X = X
        self.y = y
        self.timestamps = list(range(len(X))) if timestamps is None else timestamps
        self.asset_names = ['a_{}'.format(i) for i in range(X.shape[-1])] if asset_names is None else asset_names

        # utility
        self.n_channels, self.lookback, self.n_assets = X.shape[1:]
        self.horizon = y.shape[2]

    def __len__(self):
        """Compute length."""
        return len(self.X)

    def __getitem__(self, ix):
        """Get item."""
        X_sample = torch.from_numpy(self.X[ix])
        y_sample = torch.from_numpy(self.y[ix])
        timestamps_sample = self.timestamps[ix]

        return X_sample, y_sample, timestamps_sample, self.asset_names


def collate_uniform(batch, n_assets_range=(5, 10), lookback_range=(1, 20), horizon_range=(3, 15), asset_ixs=None,
                    random_state=None, scaler=None):
    """Create batch of samples.

    Randomly (from uniform distribution) selects assets, lookback and horizon. If `assets` are specified then assets
    kept constant.

    Parameters
    ----------
    batch : list
        List of tuples representing `(X_sample, y_sample, timestamp_sample, asset_names)`. Note that the sample
        dimension is not present and all the other dimensions are full (as determined by the dataset).

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

    scaler : None or {'standard', 'percent'}
        If None then no scaling applied. If string then a specific scaling theme. Only applied to X_batch.

    Returns
    -------
    X_batch : torch.Tensor
        Features batch of shape `(batch_size, n_input_channels, sampled_lookback, n_sampled_assets)`.

    y_batch : torch.Tensor
        Targets batch of shape `(batch_size, n_input_channels, sampled_horizon, n_sampled_assets)`.

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
    horizon_max = batch[0][1].shape[1]

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
    if scaler is not None:
        X_batch = scale_features(X_batch, approach=scaler)

    y_batch = torch.stack([b[1][:, :horizon, asset_ixs] for b in batch], dim=0)
    timestamps_batch = [b[2] for b in batch]
    asset_names_batch = [batch[0][3][ix] for ix in asset_ixs]

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

    scaler : None or {'standard', 'percent'}
        If None then no scaling applied. If string then a specific scaling theme. Only applied to X_batch.

    """

    def __init__(self, dataset, indices=None, n_assets_range=(5, 10), lookback_range=(3, 20), horizon_range=(3, 15),
                 asset_ixs=None, scaler=None, **kwargs):
        # checks
        if not (2 <= n_assets_range[0] <= n_assets_range[1] <= dataset.n_assets + 1):
            raise ValueError('Invalid n_assets_range.')

        if not (2 <= lookback_range[0] <= lookback_range[1] <= dataset.lookback + 1):
            raise ValueError('Invalid lookback_range.')

        if not (1 <= horizon_range[0] <= horizon_range[1] <= dataset.horizon + 1):
            raise ValueError('Invalid horizon_range.')

        if indices is not None and not (0 <= min(indices) <= max(indices) <= len(dataset) - 1):
            raise ValueError('The indices our outside of the range of the dataset.')

        self.dataset = dataset
        self.indices = indices if indices is not None else list(range(len(dataset)))
        self.n_assets_range = n_assets_range
        self.lookback_range = lookback_range
        self.horizon_range = horizon_range
        self.asset_ixs = asset_ixs
        self.scaler = scaler

        super().__init__(dataset,
                         collate_fn=partial(collate_uniform,
                                            n_assets_range=n_assets_range,
                                            lookback_range=lookback_range,
                                            horizon_range=horizon_range,
                                            asset_ixs=asset_ixs,
                                            scaler=scaler),
                         sampler=torch.utils.data.SubsetRandomSampler(indices),
                         batch_sampler=None,
                         shuffle=False,
                         drop_last=False,
                         **kwargs)

    @property
    def hparams(self):
        """Generate dictionary of relevant parameters."""
        return {
            'lookback_range': self.lookback_range,
            'horizon_range': self.horizon_range,
            'batch_size': self.batch_size}


class RigidDataLoader(torch.utils.data.DataLoader):
    """Rigid data loader.

    Rigid data loader is well suited for validation purposes since all horizon, lookback and assets are frozen.
    However, it is not good for training since it enforces the user to choose a single setup.

    Parameters
    ----------
    dataset : torch.utils.data.Dataset
        Instance of our dataset. See ``InRAMDataset`` for more details.

    asset_ixs : list or None
        Represents indices of considered assets (not asset names). If None then considering all assets.

    indices : list or None
        List of indices to consider (not timestamps) from the provided `dataset` which is inherently ordered. If None
        then consider all the samples.

    lookback : int or None
        How many time steps do we look back. If None then taking the maximum lookback from `dataset`.

    horizon : int or None
        How many time steps we look forward. If None then taking the maximum horizon from `dataset`.

    scaler : None or {'standard', 'percent'}
        If None then no scaling applied. If string then a specific scaling theme. Only applied to X_batch.
    """

    def __init__(self, dataset, asset_ixs=None, indices=None, lookback=None, horizon=None, scaler=None, **kwargs):

        if asset_ixs is not None and not (0 <= min(asset_ixs) <= max(asset_ixs) <= dataset.n_assets - 1):
            raise ValueError('Invalid asset_ixs.')

        if lookback is not None and not (2 <= lookback <= dataset.lookback):
            raise ValueError('Invalid lookback_range.')

        if horizon is not None and not (1 <= horizon <= dataset.horizon):
            raise ValueError('Invalid horizon_range.')

        if indices is not None and not (0 <= min(indices) <= max(indices) <= len(dataset) - 1):
            raise ValueError('The indices our outside of the range of the dataset.')

        self.dataset = dataset
        self.indices = indices if indices is not None else list(range(len(dataset)))
        self.lookback = lookback if lookback is not None else dataset.lookback
        self.horizon = horizon if horizon is not None else dataset.horizon
        self.asset_ixs = asset_ixs if asset_ixs is not None else list(range(len(dataset.asset_names)))
        self.scaler = scaler

        super().__init__(self.dataset,
                         collate_fn=partial(collate_uniform,
                                            n_assets_range=None,
                                            lookback_range=(self.lookback, self.lookback + 1),
                                            horizon_range=(self.horizon, self.horizon + 1),
                                            asset_ixs=self.asset_ixs,
                                            scaler=self.scaler),
                         sampler=torch.utils.data.SubsetRandomSampler(self.indices),
                         batch_sampler=None,
                         shuffle=False,
                         drop_last=False,
                         **kwargs)

    @property
    def hparams(self):
        """Generate dictionary of relevant parameters."""
        return {'lookback': self.lookback,
                'horizon': self.horizon,
                'batch_size': self.batch_size}
