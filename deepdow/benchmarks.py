"""Collection of benchmarks."""
from abc import ABC, abstractmethod

import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
import torch

from deepdow.nn import CovarianceMatrix


class Benchmark(ABC):
    """Abstract benchmark class.

    The idea is to create some benchmarks that we can use for comparison to our neural networks.

    """

    @abstractmethod
    def __call__(self, X):
        """Prediction of the model."""

    @property
    @abstractmethod
    def lookback_invariant(self):
        """Determine whether model yields the same weights irrespective of the lookback size."""

    def fit(self, *args, **kwargs):
        """Fitting of the model. By default does nothing."""
        return self


class MaximumReturn(Benchmark):
    """Markowitz portfolio optimization - maximum return."""

    def __init__(self, max_weight=1):
        """Construct.

        Parameters
        ----------
        max_weight : float
            A number in (0, 1] representing the maximum weight per asset.
        """
        self.max_weight = max_weight

    @property
    def lookback_invariant(self):
        return False

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
        n_samples, _, lookback, n_assets = X.shape

        # Problem setup
        rets = cp.Parameter(n_assets)
        w = cp.Variable(n_assets)

        ret = rets @ w
        prob = cp.Problem(cp.Maximize(ret), [cp.sum(w) == 1,
                                             w >= 0,
                                             w <= self.max_weight])

        cvxpylayer = CvxpyLayer(prob, parameters=[rets], variables=[w])

        # problem solver
        rets_estimate = X[:, 0, :, :].mean(dim=1)
        results_list = [cvxpylayer(rets_estimate[i])[0] for i in range(n_samples)]

        return torch.stack(results_list, dim=0)


class MinimumVariance(Benchmark):
    """Markowitz portfolio optimization - minimum variance."""

    def __init__(self, max_weight=1):
        """Construct.

        Parameters
        ----------
        max_weight : float
            A number in (0, 1] representing the maximum weight per asset.
        """
        self.max_weight = max_weight

    @property
    def lookback_invariant(self):
        return False

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
        n_samples, _, lookback, n_assets = X.shape

        # Problem setup
        covmat_sqrt = cp.Parameter((n_assets, n_assets))
        w = cp.Variable(n_assets)

        risk = cp.sum_squares(covmat_sqrt @ w)
        prob = cp.Problem(cp.Minimize(risk), [cp.sum(w) == 1,
                                              w >= 0,
                                              w <= self.max_weight])

        cvxpylayer = CvxpyLayer(prob, parameters=[covmat_sqrt], variables=[w])

        # problem solver
        covmat_sqrt_estimates = CovarianceMatrix(sqrt=True)(X[:, 0, :, :])
        results_list = [cvxpylayer(covmat_sqrt_estimates[i])[0] for i in range(n_samples)]

        return torch.stack(results_list, dim=0)


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

        return torch.ones((n_samples, n_assets), dtype=X.dtype) / n_assets

    @property
    def lookback_invariant(self):
        return True


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

        weights_unscaled = torch.rand((n_samples, n_assets), dtype=X.dtype)
        weights_sums = weights_unscaled.sum(dim=1, keepdim=True).repeat(1, n_assets)

        return weights_unscaled / weights_sums

    @property
    def lookback_invariant(self):
        return True


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

        weights = torch.zeros((n_samples, n_assets), dtype=X.dtype)
        weights[:, self.asset_ix] = 1

        return weights

    @property
    def lookback_invariant(self):
        return True
