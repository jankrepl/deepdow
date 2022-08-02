"""Collection of callable functions that augment deepdow tensors."""

import numpy as np
import torch


def prepare_standard_scaler(X, overlap=False, indices=None):
    """Compute mean and standard deviation for each channel.

    Parameters
    ----------
    X : np.ndarray
        Full features array of shape `(n_samples, n_channels, lookback, n_assets)`.

    overlap : bool
        If False, then only using the most recent timestep. This will guarantee that not counting
        the same thing multiple times.

    indices : list or None
        List of indices to consider from the `X.shape[0]` dimension. If None
        then considering all the samples.

    Returns
    -------
    means : np.ndarray
        Mean of each channel. Shape `(n_channels,)`.

    stds : np.ndarray
        Standard deviation of each channel. Shape `(n_channels,)`.

    """
    indices = indices if indices is not None else list(range(len(X)))
    considered_values = X[indices, ...] if overlap else X[indices, :, -1:, :]

    means = considered_values.mean(axis=(0, 2, 3))
    stds = considered_values.std(axis=(0, 2, 3))

    return means, stds


def prepare_robust_scaler(
    X, overlap=False, indices=None, percentile_range=(25, 75)
):
    """Compute median and percentile range for each channel.

    Parameters
    ----------
    X : np.ndarray
        Full features array of shape `(n_samples, n_channels, lookback, n_assets)`.

    overlap : bool
        If False, then only using the most recent timestep. This will guarantee that not counting
        the same thing multiple times.

    indices : list or None
        List of indices to consider from the `X.shape[0]` dimension. If None
        then considering all the samples.

    percentile_range : tuple
        The left and right percentile to consider. Needs to be in [0, 100].

    Returns
    -------
    medians : np.ndarray
        Median of each channel. Shape `(n_channels,)`.

    ranges : np.ndarray
        Interquantile range for each channel. Shape `(n_channels,)`.

    """
    if not 0 <= percentile_range[0] < percentile_range[1] <= 100:
        raise ValueError(
            "The percentile range needs to be in [0, 100] and left < right"
        )

    indices = indices if indices is not None else list(range(len(X)))
    considered_values = X[indices, ...] if overlap else X[indices, :, -1:, :]

    medians = np.median(considered_values, axis=(0, 2, 3))
    percentiles = np.percentile(
        considered_values, percentile_range, axis=(0, 2, 3)
    )  # (2, n_channels)

    ranges = percentiles[1] - percentiles[0]

    return medians, ranges


class Compose:
    """Meta transform inspired by torchvision.

    Parameters
    ----------
    transforms : list
        List of callables that represent transforms to be composed.

    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, X_sample, y_sample, timestamps_sample, asset_names):
        """Transform.

        Parameters
        ----------
        X_sample : torch.Tensor
            Feature vector of shape `(n_channels, lookback, n_assets)`.

        y_sample : torch.Tesnor
            Target vector of shape `(n_channels, horizon, n_assets)`.

        timestamps_sample : datetime
            Time stamp of the sample.

        asset_names
            Asset names corresponding to the last channel of `X_sample` and `y_sample`.

        Returns
        -------
        X_sample_new : torch.Tensor
            Transformed version of `X_sample`.

        y_sample_new : torch.Tesnor
            Transformed version of `y_sample`.

        timestamps_sample_new : datetime
            Transformed version of `timestamps_sample`.

        asset_names_new
            Transformed version of `asset_names`.
        """
        for t in self.transforms:
            X_sample, y_sample, timestamps_sample, asset_names = t(
                X_sample, y_sample, timestamps_sample, asset_names
            )

        return X_sample, y_sample, timestamps_sample, asset_names


class Dropout:
    """Set random elements of the input to zero with probability p.

    Parameters
    ----------
    p : float
        Probability of setting an element to zero.

    training : bool
        If False, then dropout disabled no matter what the `p` is. Note that if True then
        dropout enabled and at the same time all the elements are scaled by `1/p`.
    """

    def __init__(self, p=0.2, training=True):
        self.p = p
        self.training = training

    def __call__(self, X_sample, y_sample, timestamps_sample, asset_names):
        """Perform transform.

        Parameters
        ----------
        X_sample : torch.Tensor
            Feature vector of shape `(n_channels, lookback, n_assets)`.

        y_sample : torch.Tesnor
            Target vector of shape `(n_channels, horizon, n_assets)`.

        timestamps_sample : datetime
            Time stamp of the sample.

        asset_names
            Asset names corresponding to the last channel of `X_sample` and `y_sample`.

        Returns
        -------
        X_sample_new : torch.Tensor
            Feature vector of shape `(n_channels, lookback, n_assets)` with some elements being set to zero.

        y_sample : torch.Tensor
            Same as input.

        timestamps_sample : datetime
            Same as input.

        asset_names
            Same as input.
        """
        X_sample_new = torch.nn.functional.dropout(
            X_sample, p=self.p, training=self.training
        )

        return X_sample_new, y_sample, timestamps_sample, asset_names


class Multiply:
    """Transform multiplying the feature tensor X with a constant."""

    def __init__(self, c=100):
        self.c = c

    def __call__(self, X_sample, y_sample, timestamps_sample, asset_names):
        """Perform transform.

        Parameters
        ----------
        X_sample : torch.Tensor
            Feature vector of shape `(n_channels, lookback, n_assets)`.

        y_sample : torch.Tesnor
            Target vector of shape `(n_channels, horizon, n_assets)`.

        timestamps_sample : datetime
            Time stamp of the sample.

        asset_names
            Asset names corresponding to the last channel of `X_sample` and `y_sample`.

        Returns
        -------
        X_sample_new : torch.Tensor
            Feature vector of shape `(n_channels, lookback, n_assets)` multiplied by a constant `self.c`.

        y_sample : torch.Tesnor
            Same as input.

        timestamps_sample : datetime
            Same as input.

        asset_names
            Same as input.
        """
        return self.c * X_sample, y_sample, timestamps_sample, asset_names


class Noise:
    """Add noise to each of the channels.

    Random (Gaussian) noise is added to the original features X. One can control the standard deviation of the noise
    via the `frac` parameter. Mathematically, `std(X_noise) = std(X) * frac` for each channel.


    """

    def __init__(self, frac=0.2):
        self.frac = frac

    def __call__(self, X_sample, y_sample, timestamps_sample, asset_names):
        """Perform transform.

        Parameters
        ----------
        X_sample : torch.Tensor
            Feature vector of shape `(n_channels, lookback, n_assets)`.

        y_sample : torch.Tensor
            Target vector of shape `(n_channels, horizon, n_assets)`.

        timestamps_sample : datetime
            Time stamp of the sample.

        asset_names
            Asset names corresponding to the last channel of `X_sample` and `y_sample`.

        Returns
        -------
        X_sample_new : torch.Tensor
            Feature vector of shape `(n_channels, lookback, n_assets)` with some added noise.

        y_sample : torch.Tesnor
            Same as input.

        timestamps_sample : datetime
            Same as input.

        asset_names
            Same as input.
        """
        X_sample_new = (
            self.frac
            * X_sample.std([1, 2], keepdim=True)
            * torch.randn_like(X_sample)
            + X_sample
        )

        return X_sample_new, y_sample, timestamps_sample, asset_names


class Scale:
    """Scale input features.

    The input features are per channel centered to zero and scaled to one. We use the same
    terminology as scikit-learn. However, the equivalent in torchvision is `Normalize`.

    Parameters
    ----------
    center : np.ndarray
        1D array of shape `(n_channels,)` representing the center of the features (mean or median).
        Needs to be precomputed in advance.

    scale : np.ndarray
        1D array of shape `(n_channels,)` representing the scale of the features (standard deviation
        or quantile range). Needs to be precomputed in advance.

    See Also
    --------
    prepare_robust_scaler
    prepare_standard_scaler
    """

    def __init__(self, center, scale):
        if len(center) != len(scale):
            raise ValueError(
                "The center and scale need to have the same size."
            )

        if np.any(scale <= 0):
            raise ValueError("The scale parameters need to be positive.")

        self.center = center
        self.scale = scale
        self.n_channels = len(self.center)

    def __call__(self, X_sample, y_sample, timestamps_sample, asset_names):
        """Perform transform.

        Parameters
        ----------
        X_sample : torch.Tensor
            Feature vector of shape `(n_channels, lookback, n_assets)`.

        y_sample : torch.Tensor
            Target vector of shape `(n_channels, horizon, n_assets)`.

        timestamps_sample : datetime
            Time stamp of the sample.

        asset_names
            Asset names corresponding to the last channel of `X_sample` and `y_sample`.

        Returns
        -------
        X_sample_new : torch.Tensor
            Feature vector of shape `(n_channels, lookback, n_assets)` scaled appropriately.

        y_sample : torch.Tesnor
            Same as input.

        timestamps_sample : datetime
            Same as input.

        asset_names
            Same as input.
        """
        n_channels = X_sample.shape[0]
        if n_channels != self.n_channels:
            raise ValueError(
                "Expected {} channels in X, got {}".format(
                    self.n_channels, n_channels
                )
            )

        X_sample_new = X_sample.clone()
        dtype, device = X_sample_new.dtype, X_sample_new.device

        center = torch.as_tensor(self.center, dtype=dtype, device=device)[
            :, None, None
        ]
        scale = torch.as_tensor(self.scale, dtype=dtype, device=device)[
            :, None, None
        ]

        X_sample_new.sub_(center).div_(scale)

        return X_sample_new, y_sample, timestamps_sample, asset_names
