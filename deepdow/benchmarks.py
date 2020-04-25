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
    def __call__(self, x):
        """Prediction of the model."""

    @property
    def hparams(self):
        """Hyperparamters relevant to construction of the model."""
        return {}


class InverseVolatility(Benchmark):
    """Allocation only considering volatility of individual assets.

    Parameters
    ----------
    use_std : bool
        If True, then we use standard deviation as a measure of volatility. Otherwise variance is used.

    returns_channel : int
        Which channel in the `x` feature matrix to consider (the 2nd dimension) as returns.

    """

    def __init__(self, use_std=False, returns_channel=0):
        self.use_std = use_std
        self.returns_channel = returns_channel

    def __call__(self, x):
        """Predict weights.

        Parameters
        ----------
        x : torch.Tensor
            Tensor of shape `(n_samples, n_channels, lookback, n_assets)`.

        Returns
        -------
        weights : torch.Tensor
            Tensor of shape `(n_samples, n_assets)` representing the predicted weights.

        """
        eps = 1e-6
        x_rets = x[:, self.returns_channel, ...]
        vols = x_rets.std(dim=1) if self.use_std else x_rets.var(dim=1)
        ivols = 1 / (vols + eps)
        weights = ivols / ivols.sum(dim=1, keepdim=True)

        return weights

    @property
    def hparams(self):
        """Hyperparamters relevant to construction of the model."""
        return {'use_std': self.use_std,
                'returns_channel': self.returns_channel}


class MaximumReturn(Benchmark):
    """Markowitz portfolio optimization - maximum return.

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
        Which channel in the `x` feature matrix to consider (the 2nd dimension) as returns.

    Attributes
    ----------
    optlayer : cvxpylayers.torch.CvxpyLayer or None
        Equal to None if `n_assets` not provided in the constructor. In this case optimization problem is constructed
        with each forward pass. This allows for variable number of assets but is slower. If `n_assets` provided than
        constructed once and for all in the constructor.

    """

    def __init__(self, max_weight=1, n_assets=None, returns_channel=0):
        self.max_weight = max_weight
        self.n_assets = n_assets
        self.returns_channel = returns_channel

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

    def __call__(self, x):
        """Predict weights.

        Parameters
        ----------
        x : torch.Tensor
            Tensor of shape `(n_samples, n_channels, lookback, n_assets)`.

        Returns
        -------
        weights : torch.Tensor
            Tensor of shape `(n_samples, n_assets)` representing the predicted weights.

        """
        n_samples, _, lookback, n_assets = x.shape

        # Problem setup
        if self.optlayer is not None:
            if self.n_assets != n_assets:
                raise ValueError('Incorrect number of assets: {}, expected: {}'.format(n_assets, self.n_assets))

            optlayer = self.optlayer
        else:
            optlayer = self._construct_problem(n_assets, self.max_weight)

        rets_estimate = x[:, self.returns_channel, :, :].mean(dim=1)  # (n_samples, n_assets)

        return optlayer(rets_estimate)[0]

    @property
    def hparams(self):
        """Hyperparamters relevant to construction of the model."""
        return {'max_weight': self.max_weight,
                'returns_channel': self.returns_channel,
                'n_assets': self.n_assets}


class MinimumVariance(Benchmark):
    """Markowitz portfolio optimization - minimum variance.

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
        Which channel in the `x` feature matrix to consider (the 2nd dimension) as returns.

    Attributes
    ----------
    optlayer : cvxpylayers.torch.CvxpyLayer or None
        Equal to None if `n_assets` not provided in the constructor. In this case optimization problem is constructed
        with each forward pass. This allows for variable number of assets but is slower. If `n_assets` provided than
        constructed once and for all in the constructor.

    """

    def __init__(self, max_weight=1, returns_channel=0, n_assets=None):
        self.n_assets = n_assets
        self.returns_channel = returns_channel
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

    def __call__(self, x):
        """Predict weights.

        Parameters
        ----------
        x : torch.Tensor
            Tensor of shape `(n_samples, n_channels, lookback, n_assets)`.

        Returns
        -------
        weights : torch.Tensor
            Tensor of shape `(n_samples, n_assets)` representing the predicted weights.

        """
        n_samples, _, lookback, n_assets = x.shape

        # Problem setup
        if self.optlayer is not None:
            if self.n_assets != n_assets:
                raise ValueError('Incorrect number of assets: {}, expected: {}'.format(n_assets, self.n_assets))

            optlayer = self.optlayer
        else:
            optlayer = self._construct_problem(n_assets, self.max_weight)

        # problem solver
        covmat_sqrt_estimates = CovarianceMatrix(sqrt=True)(x[:, self.returns_channel, :, :])

        return optlayer(covmat_sqrt_estimates)[0]

    @property
    def hparams(self):
        """Hyperparamters relevant to construction of the model."""
        return {'max_weight': self.max_weight,
                'returns_channel': self.returns_channel,
                'n_assets': self.n_assets}


class OneOverN(Benchmark):
    """Equally weighted portfolio."""

    def __call__(self, x):
        """Predict weights.

        Parameters
        ----------
        x : torch.Tensor
            Tensor of shape `(n_samples, n_channels, lookback, n_assets)`.

        Returns
        -------
        weights : torch.Tensor
            Tensor of shape `(n_samples, n_assets)` representing the predicted weights.

        """
        n_samples, n_channels, lookback, n_assets = x.shape

        return torch.ones((n_samples, n_assets), dtype=x.dtype, device=x.device) / n_assets


class Random(Benchmark):
    """Random allocation for each prediction."""

    def __call__(self, x):
        """Predict weights.

        Parameters
        ----------
        x : torch.Tensor
            Tensor of shape `(n_samples, n_channels, lookback, n_assets)`.

        Returns
        -------
        weights : torch.Tensor
            Tensor of shape `(n_samples, n_assets)` representing the predicted weights.

        """
        n_samples, n_channels, lookback, n_assets = x.shape

        weights_unscaled = torch.rand((n_samples, n_assets), dtype=x.dtype, device=x.device)
        weights_sums = weights_unscaled.sum(dim=1, keepdim=True).repeat(1, n_assets)

        return weights_unscaled / weights_sums


class Singleton(Benchmark):
    """Predict a single asset.

    Parameters
    ----------
    asset_ix : int
        Index of the asset to predict.

    """

    def __init__(self, asset_ix):
        self.asset_ix = asset_ix

    def __call__(self, x):
        """Predict weights.

        Parameters
        ----------
        x : torch.Tensor
            Tensor of shape `(n_samples, n_channels, lookback, n_assets)`.

        Returns
        -------
        weights : torch.Tensor
            Tensor of shape `(n_samples, n_assets)` representing the predicted weights.

        """
        n_samples, n_channels, lookback, n_assets = x.shape

        if self.asset_ix not in set(range(n_assets)):
            raise IndexError('The selected asset index is out of range.')

        weights = torch.zeros((n_samples, n_assets), dtype=x.dtype, device=x.device)
        weights[:, self.asset_ix] = 1

        return weights

    @property
    def hparams(self):
        """Hyperparamters relevant to construction of the model."""
        return {'asset_ix': self.asset_ix}
