"""Collection of layers that are using convex optimization."""

import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
import torch.nn as nn


class Markowitz(nn.Module):
    """Convex optimization layer stylized into portfolio optimization problem.

    Parameters
    ----------
    n_assets : int
        Number of assets.

    Attributes
    ----------
    cvxpylayer : CvxpyLayer
        Custom layer used by a third party package called cvxpylayers.

    Notes
    -----
    One can also reinstantiate this layer inside of a forward pass of the main network. This way we can dynamically
    determine `n_assets` and reuse existing tensors as parameters of the optimization problem.

    References
    ----------
    [1] https://github.com/cvxgrp/cvxpylayers

    """

    def __init__(self, n_assets, max_weight=1):
        """Construct."""
        super().__init__()
        covmat_sqrt = cp.Parameter((n_assets, n_assets))
        rets = cp.Parameter(n_assets)

        w = cp.Variable(n_assets)
        ret = rets @ w
        risk = cp.sum_squares(covmat_sqrt @ w)

        prob = cp.Problem(cp.Maximize(ret - risk),
                          [cp.sum(w) == 1,
                           w >= 0,
                           w <= max_weight
                           ])

        assert prob.is_dpp()

        self.cvxpylayer = CvxpyLayer(prob, parameters=[rets, covmat_sqrt], variables=[w])

    def forward(self, rets, covmat_sqrt, gamma):
        """Perform forward pass.

        Parameters
        ----------
        rets : torch.Tensor
            Of shape (n_samples, n_assets) representing expected returns (or whatever the feature extractor decided
            to encode).

        covmat_sqrt : torch.Tensor
            Of shape (n_samples, n_assets, n_assets) representing the square of the covariance matrix. Fully determined
            by the feature extractor by in standard Markowitz setup it represents the risk model.

        gamma : torch.Tensor
            Of shape (n_samples,) representing the tradeoff between risk and return - where on efficient frontier
            we are.

        Returns
        -------
        weights : torch.Tensor
            Of shape (n_samples, n_assets) representing the optimal weights as determined by the convex optimizer.

        """
        n_samples, n_assets = rets.shape
        gamma_ = gamma.repeat((1, n_assets * n_assets)).view(n_samples, n_assets, n_assets)

        return self.cvxpylayer(rets, gamma_ * covmat_sqrt)[0]
