"""Collection of callable functions that augment deepdow tensors."""

import torch


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
            X_sample, y_sample, timestamps_sample, asset_names = t(X_sample, y_sample, timestamps_sample, asset_names)

        return X_sample, y_sample, timestamps_sample, asset_names


class Dropout:
    """Set random elements of the input to zero with probability p.

    Parameters
    ----------
    p : float
        Probability of setting an element to zero.

    training : bool
        If False, then dropout disabled no matter what the `p` is.
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
        X_sample_new = torch.nn.functional.dropout(X_sample, p=self.p, training=self.training)

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
        X_sample_new = self.frac * X_sample.std([1, 2], keepdim=True) * torch.randn_like(X_sample) + X_sample

        return X_sample_new, y_sample, timestamps_sample, asset_names
