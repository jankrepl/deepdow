"""Collection of benchmarks."""
from abc import ABC, abstractmethod

import torch


class Benchmark(ABC):
    """Abstract benchmark class.

    The idea is to create some benchmarks that we can use for comparison to our neural networks.

    """

    @abstractmethod
    def __call__(self, X):
        """Prediction of the model."""

    def fit(self, *args, **kwargs):
        """Fitting of the model. By default does nothing."""
        return self


class OneOverN(Benchmark):
    """Equally weighted portfolio."""

    def __call__(self, X):
        """Predict weights.

        Parameters
        ----------
        X : torch.Tensor
            Tensor of shape `(n_samples, 1, lookback, n_assets)`.

        Returns
        -------
        weights : torch.Tensor
            Tensor of shape `(n_samples, n_assets)` representing the predicted weights.

        """
        n_samples, n_channels, lookback, n_assets = X.shape

        return torch.ones((n_samples, n_assets)) / n_assets


class Random(Benchmark):
    """Random allocation for each prediction."""

    def __call__(self, X):
        """Predict weights.

        Parameters
        ----------
        X : torch.Tensor
            Tensor of shape `(n_samples, 1, lookback, n_assets)`.

        Returns
        -------
        weights : torch.Tensor
            Tensor of shape `(n_samples, n_assets)` representing the predicted weights.

        """
        n_samples, n_channels, lookback, n_assets = X.shape

        weights_unscaled = torch.rand((n_samples, n_assets))
        weights_sums = weights_unscaled.sum(dim=1, keepdim=True).repeat(1, n_assets)

        return weights_unscaled / weights_sums


class Singleton(Benchmark):
    """Predict a single asset."""

    def __init__(self, asset_ix):
        """Construct.

        Parameters
        ----------
        asset_ix : int
            Index of the asset to predict.
        """
        self.asset_ix = asset_ix

    def __call__(self, X):
        """Predict weights.

        Parameters
        ----------
        X : torch.Tensor
            Tensor of shape `(n_samples, 1, lookback, n_assets)`.

        Returns
        -------
        weights : torch.Tensor
            Tensor of shape `(n_samples, n_assets)` representing the predicted weights.

        """
        n_samples, n_channels, lookback, n_assets = X.shape

        if self.asset_ix not in set(range(n_assets)):
            raise IndexError('The selected asset index is out of range.')

        weights = torch.zeros((n_samples, n_assets))
        weights[:, self.asset_ix] = 1

        return weights
