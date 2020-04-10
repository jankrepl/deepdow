"""Collection of benchmarks."""
from abc import ABC, abstractmethod

import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
import torch

from .layers import CovarianceMatrix


class Benchmark(ABC):
    """Abstract benchmark class.

    The idea is to create some benchmarks that we can use for comparison to our neural networks. Note that we
    assume that benchmarks are not trainable - one can only use them for inference.

    """

    @abstractmethod
    def __call__(self, X):
        """Prediction of the model."""

    @property
    def hparams(self):
        """Hyperparamters relevant to construction of the model."""
        return {}


class MaximumReturn(Benchmark):
    """Markowitz portfolio optimization - maximum return."""

    def __init__(self, max_weight=1, n_assets=None, returns_channel=0):
        """Construct.

        Parameters
        ----------
        max_weight : float
            A number in (0, 1] representing the maximum weight per asset.

        n_assets : None or int
            If specifed the benchmark will always have to be provided with `n_assets` of assets in the `__call__`.
            This way one can achieve major speedups since the optimization problem is canonicalized only once in the
            constructor. However, when `n_assets` is None the optimization problem is canonicalized before each
            inside of `__call__` which results in overhead but allows for variable number of assets.

        returns_channel : int
            Which channel in the `X` feature matrix to consider (the 2nd dimension) as returns.
        """
        self.max_weight = max_weight
        self.n_assets = n_assets
        self.return_channel = returns_channel

        self.optlayer = self._construct_problem(n_assets, max_weight) if self.n_assets is not None else None

    @staticmethod
    def _construct_problem(n_assets, max_weight):
        """Construct cvxpylayers problem."""
        rets = cp.Parameter(n_assets)
        w = cp.Variable(n_assets)

        ret = rets @ w
        prob = cp.Problem(cp.Maximize(ret), [cp.sum(w) == 1,
                                             w >= 0,
                                             w <= max_weight])

        return CvxpyLayer(prob, parameters=[rets], variables=[w])

    def __call__(self, X):
        """Predict weights.

        Parameters
        ----------
        X : torch.Tensor
            Tensor of shape `(n_samples, n_input_channels, lookback, n_assets)`.

        Returns
        -------
        weights : torch.Tensor
            Tensor of shape `(n_samples, n_assets)` representing the predicted weights.

        """
        n_samples, _, lookback, n_assets = X.shape

        # Problem setup
        if self.optlayer is not None:
            if self.n_assets != n_assets:
                raise ValueError('Incorrect number of assets: {}, expected: {}'.format(n_assets, self.n_assets))

            optlayer = self.optlayer
        else:
            optlayer = self._construct_problem(n_assets, self.max_weight)

        rets_estimate = X[:, self.return_channel, :, :].mean(dim=1)  # (n_samples, n_assets)

        return optlayer(rets_estimate)[0]


class MinimumVariance(Benchmark):
    """Markowitz portfolio optimization - minimum variance."""

    def __init__(self, max_weight=1, returns_channel=0, n_assets=None):
        """Construct.

        Parameters
        ----------
        max_weight : float
            A number in (0, 1] representing the maximum weight per asset.
        """
        self.n_assets = n_assets
        self.return_channel = returns_channel
        self.max_weight = max_weight

        self.optlayer = self._construct_problem(n_assets, max_weight) if self.n_assets is not None else None

    @staticmethod
    def _construct_problem(n_assets, max_weight):
        """Construct cvxpylayers problem."""
        covmat_sqrt = cp.Parameter((n_assets, n_assets))
        w = cp.Variable(n_assets)

        risk = cp.sum_squares(covmat_sqrt @ w)
        prob = cp.Problem(cp.Minimize(risk), [cp.sum(w) == 1,
                                              w >= 0,
                                              w <= max_weight])

        return CvxpyLayer(prob, parameters=[covmat_sqrt], variables=[w])

    def __call__(self, X):
        """Predict weights.

        Parameters
        ----------
        X : torch.Tensor
            Tensor of shape `(n_samples, n_input_channels, lookback, n_assets)`.

        Returns
        -------
        weights : torch.Tensor
            Tensor of shape `(n_samples, n_assets)` representing the predicted weights.

        """
        n_samples, _, lookback, n_assets = X.shape

        # Problem setup
        if self.optlayer is not None:
            if self.n_assets != n_assets:
                raise ValueError('Incorrect number of assets: {}, expected: {}'.format(n_assets, self.n_assets))

            optlayer = self.optlayer
        else:
            optlayer = self._construct_problem(n_assets, self.max_weight)

        # problem solver
        covmat_sqrt_estimates = CovarianceMatrix(sqrt=True)(X[:, self.return_channel, :, :])

        return optlayer(covmat_sqrt_estimates)[0]


class OneOverN(Benchmark):
    """Equally weighted portfolio."""

    def __call__(self, X):
        """Predict weights.

        Parameters
        ----------
        X : torch.Tensor
            Tensor of shape `(n_samples, n_input_channels, lookback, n_assets)`.

        Returns
        -------
        weights : torch.Tensor
            Tensor of shape `(n_samples, n_assets)` representing the predicted weights.

        """
        n_samples, n_channels, lookback, n_assets = X.shape

        return torch.ones((n_samples, n_assets), dtype=X.dtype, device=X.device) / n_assets


class Random(Benchmark):
    """Random allocation for each prediction."""

    def __call__(self, X):
        """Predict weights.

        Parameters
        ----------
        X : torch.Tensor
            Tensor of shape `(n_samples, n_input_channels, lookback, n_assets)`.

        Returns
        -------
        weights : torch.Tensor
            Tensor of shape `(n_samples, n_assets)` representing the predicted weights.

        """
        n_samples, n_channels, lookback, n_assets = X.shape

        weights_unscaled = torch.rand((n_samples, n_assets), dtype=X.dtype, device=X.device)
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

        weights = torch.zeros((n_samples, n_assets), dtype=X.dtype, device=X.device)
        weights[:, self.asset_ix] = 1

        return weights
