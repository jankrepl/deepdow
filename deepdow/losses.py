"""Collection of losses.

All losses are designed for minimization.
"""
from types import MethodType
from .layers import CovarianceMatrix

import torch


def covariance(x, y):
    """Compute covariance between two 2D tensors.

    Parameters
    ----------
    x : torch.tensor
        Torch tensor of shape `(n_samples, horizon)`

    y : torch.tensor
        Tensor of shape `(n_samples, horizon)`

    Returns
    -------
    cov : torch.tensor
        Torch tensor of shape `(n_samples,)`.
    """
    n_samples, horizon = x.shape
    mean_x = x.mean(dim=1, keepdim=True)
    mean_y = y.mean(dim=1, keepdim=True)
    xm = x - mean_x  # (n_samples, horizon)
    ym = y - mean_y  # (n_samples, horizon)

    cov = (xm * ym).sum(dim=1) / horizon

    return cov


def log2simple(x):
    """Turn log returns into simple returns.

    r_simple = exp(r_log) - 1.

    Parameters
    ----------
    x : torch.Tensor
        Tensor of any shape where each entry represents a logarithmic return.

    Returns
    -------
    torch.Tensor
        Simple returns.

    """
    return torch.exp(x) - 1


def simple2log(x):
    """Turn simple returns into log returns.

    r_log = ln(r_simple + 1).

    Parameters
    ----------
    x : torch.Tensor
        Tensor of any shape where each entry represents a simple return.

    Returns
    -------
    torch.Tensor
        Logarithmic returns.

    """
    return torch.log(x + 1)


def portfolio_returns(
    weights, y, input_type="log", output_type="simple", rebalance=False
):
    """Compute portfolio returns.

    Parameters
    ----------
    weights : torch.Tensor
        Tensor of shape (n_samples, n_assets) representing the simple buy and hold strategy over the horizon.

    y : torch.Tensor
        Tensor of shape (n_samples, horizon, n_assets) representing single period non-cumulative returns.

    input_type : str, {'log', 'simple'}
        What type of returns are we dealing with in `y`.

    output_type : str, {'log', 'simple'}
        What type of returns are we dealing with in the output.

    rebalance : bool
        If True, each timestep the weights are adjusted to be equal to the original ones. Note that
        this assumes that we tinker with the portfolio. If False, the portfolio evolves untouched.

    Returns
    -------
    portfolio_returns : torch.Tensor
        Of shape (n_samples, horizon) representing per timestep portfolio returns.

    """
    if input_type == "log":
        simple_returns = log2simple(y)

    elif input_type == "simple":
        simple_returns = y

    else:
        raise ValueError("Unsupported input type: {}".format(input_type))

    n_samples, horizon, n_assets = simple_returns.shape

    weights_ = weights.view(n_samples, 1, n_assets).repeat(
        1, horizon, 1
    )  # (n_samples, horizon, n_assets)

    if not rebalance:
        weights_unscaled = (1 + simple_returns).cumprod(1)[
            :, :-1, :
        ] * weights_[:, 1:, :]
        weights_[:, 1:, :] = weights_unscaled / weights_unscaled.sum(
            2, keepdim=True
        )

    out = (simple_returns * weights_).sum(-1)

    if output_type == "log":
        return simple2log(out)

    elif output_type == "simple":
        return out

    else:
        raise ValueError("Unsupported output type: {}".format(output_type))


def portfolio_cumulative_returns(
    weights, y, input_type="log", output_type="simple", rebalance=False
):
    """Compute cumulative portfolio returns.

    Parameters
    ----------
    weights : torch.Tensor
        Tensor of shape `(n_samples, n_assets)` representing the predicted weights by our portfolio optimizer.

    y : torch.Tensor
        Tensor of shape `(n_samples, horizon, n_assets)` representing the log return evolution over the next
        `horizon` timesteps.

    input_type : str, {'log', 'simple'}
        What type of returns are we dealing with in `y`.

    output_type : str, {'log', 'simple'}
        What type of returns are we dealing with in the output.

    rebalance : bool
        If True, each timestep the weights are adjusted to be equal to be equal to the original ones. Note that
        this assumes that we tinker with the portfolio. If False, the portfolio evolves untouched.

    Returns
    -------
    torch.Tensor
        Tensor of shape `(n_samples, horizon)`.

    """
    prets = portfolio_returns(
        weights,
        y,
        input_type=input_type,
        output_type="log",
        rebalance=rebalance,
    )
    log_prets = torch.cumsum(
        prets, dim=1
    )  # we can aggregate log returns over time by sum

    if output_type == "log":
        return log_prets

    elif output_type == "simple":
        return log2simple(log_prets)

    else:
        raise ValueError("Unsupported output type: {}".format(output_type))


class Loss:
    """Parent class for all losses.

    Additionally it implement +, -, * and / operation between losses.
    """

    def _call(self, weights, y):
        raise NotImplementedError()

    def _repr(self):
        raise NotImplementedError()

    def __call__(self, weights, y):
        """Compute loss.

        Parameters
        ----------
        weights : torch.Tensor
            Tensor of shape `(n_samples, n_assets)` representing the predicted weights by our portfolio optimizer.

        y : torch.Tensor
            Tensor of shape `(n_samples, n_input_channels, horizon, n_assets)` representing ground truth labels
            over the `horizon` of steps. The idea is that the channel dimensions can be given a specific meaning
            in the constructor.

        Returns
        -------
        torch.Tensor
            Tensor of shape `(n_samples,)` representing the per sample loss.

        """
        return self._call(weights, y)

    def __repr__(self):
        """Generate representation string.

        The goal is two generate a string `s` that we can `eval(s)` to instantiate the loss.
        """
        return self._repr()

    def __add__(self, other):
        """Add two losses together.

        Parameters
        ----------
        other : Loss or int or float
            If instance of ``Loss`` then creates a new loss that represents the sum of `self` and `other`. If a number
            then create a new loss that is equal to `self` plus a constant.

        Returns
        -------
        new : Loss
            Instance of a ``Loss`` representing the addition operation.
        """
        if isinstance(other, Loss):
            new_instance = Loss()
            new_instance._call = MethodType(
                lambda inst, weights, y: self(weights, y) + other(weights, y),
                new_instance,
            )
            new_instance._repr = MethodType(
                lambda inst: "{} + {}".format(
                    self.__repr__(), other.__repr__()
                ),
                new_instance,
            )

            return new_instance

        elif isinstance(other, (int, float)):
            new_instance = Loss()
            new_instance._call = MethodType(
                lambda inst, weights, y: self(weights, y) + other, new_instance
            )
            new_instance._repr = MethodType(
                lambda inst: "{} + {}".format(self.__repr__(), other),
                new_instance,
            )

            return new_instance
        else:
            raise TypeError("Unsupported type: {}".format(type(other)))

    def __radd__(self, other):
        """Add two losses together.

        Parameters
        ----------
        other : Loss or int or float
            If instance of ``Loss`` then creates a new loss that represents the sum of `self` and `other`. If a number
            then create a new loss that is equal to `self` plus a constant.

        Returns
        -------
        new : Loss
            Instance of a ``Loss`` representing the addition operation.
        """
        return self.__add__(other)

    def __mul__(self, other):
        """Multiply two losses together.

        Parameters
        ----------
        other : Loss or int or float
            If instance of ``Loss`` then creates a new loss that represents the product of `self` and `other`. If a
            number then create a new loss that is equal to `self` times a constant.

        Returns
        -------
        new : Loss
            Instance of a ``Loss`` representing the multiplication operation.
        """
        if isinstance(other, Loss):
            new_instance = Loss()
            new_instance._call = MethodType(
                lambda inst, weights, y: self(weights, y) * other(weights, y),
                new_instance,
            )
            new_instance._repr = MethodType(
                lambda inst: "{} * {}".format(
                    self.__repr__(), other.__repr__()
                ),
                new_instance,
            )

            return new_instance

        elif isinstance(other, (int, float)):
            new_instance = Loss()
            new_instance._call = MethodType(
                lambda inst, weights, y: self(weights, y) * other, new_instance
            )
            new_instance._repr = MethodType(
                lambda inst: "{} * {}".format(self.__repr__(), other),
                new_instance,
            )

            return new_instance
        else:
            raise TypeError("Unsupported type: {}".format(type(other)))

    def __rmul__(self, other):
        """Multiply two losses together.

        Parameters
        ----------
        other : Loss or int or float
            If instance of ``Loss`` then creates a new loss that represents the product of `self` and `other`. If a
            number then create a new loss that is equal to `self` times a constant.

        Returns
        -------
        new : Loss
            Instance of a ``Loss`` representing the multiplication operation.
        """
        return self.__mul__(other)

    def __truediv__(self, other):
        """Divide two losses together.

        Parameters
        ----------
        other : Loss or int or float
            If instance of ``Loss`` then creates a new loss that represents the ratio of `self` and `other`. If a
            number then create a new loss that is equal to `self` divided a constant.

        Returns
        -------
        new : Loss
            Instance of a ``Loss`` representing the division operation.
        """
        if isinstance(other, Loss):
            new_instance = Loss()
            new_instance._call = MethodType(
                lambda inst, weights, y: self(weights, y) / other(weights, y),
                new_instance,
            )
            new_instance._repr = MethodType(
                lambda inst: "{} / {}".format(
                    self.__repr__(), other.__repr__()
                ),
                new_instance,
            )

            return new_instance

        elif isinstance(other, (int, float)):
            if other == 0:
                raise ZeroDivisionError()

            new_instance = Loss()
            new_instance._call = MethodType(
                lambda inst, weights, y: self(weights, y) / other, new_instance
            )
            new_instance._repr = MethodType(
                lambda inst: "{} / {}".format(self.__repr__(), other),
                new_instance,
            )

            return new_instance
        else:
            raise TypeError("Unsupported type: {}".format(type(other)))

    def __pow__(self, power):
        """Put a loss to a power.

        Parameters
        ----------
        power : int or float
            Number representing the exponent

        Returns
        -------
        new : Loss
            Instance of a ``Loss`` representing the `self ** power`.
        """
        if isinstance(power, (int, float)):
            new_instance = Loss()
            new_instance._call = MethodType(
                lambda inst, weights, y: self(weights, y) ** power,
                new_instance,
            )
            new_instance._repr = MethodType(
                lambda inst: "({}) ** {}".format(self.__repr__(), power),
                new_instance,
            )

            return new_instance
        else:
            raise TypeError("Unsupported type: {}".format(type(power)))


class Alpha(Loss):
    """Negative alpha with respect to a selected portfolio.

    Parameters
    ----------
    benchmark_weights : torch.tensor or None
        Weights of the benchmark portfolio of shape `(n_assets,). Note that this loss assumes it will be always located
        under this index in the `y` tensor. If None then equally weighted portfolio.

    returns_channel : int
        Which channel of the `y` target represents returns.

    input_type : str, {'log', 'simple'}
        What type of returns are we dealing with in `y`.

    """

    def __init__(
        self, benchmark_weights=None, returns_channel=0, input_type="log"
    ):
        self.benchmark_weights = benchmark_weights
        self.returns_channel = returns_channel
        self.input_type = input_type

    def __call__(self, weights, y):
        """Compute negative alpha with respect to the benchmark portfolio.

        Parameters
        ----------
        weights : torch.Tensor
            Tensor of shape `(n_samples, n_assets)` representing the predicted weights by our portfolio optimizer.

        y : torch.Tensor
            Tensor of shape `(n_samples, n_channels, horizon, n_assets)` representing the evolution over the next
            `horizon` timesteps.

        Returns
        -------
        torch.Tensor
            Tensor of shape `(n_samples,)` representing the per sample negative alpha.

        """
        n_samples, n_assets = weights.shape
        device, dtype = weights.device, weights.dtype
        portfolio_rets = portfolio_returns(
            weights,
            y[:, self.returns_channel, ...],
            input_type=self.input_type,
            output_type="simple",
        )  # (n_samples, horizon)

        if self.benchmark_weights is None:
            benchmark_weights = (
                torch.ones(n_samples, n_assets, dtype=dtype, device=device)
                / n_assets
            )
        else:
            benchmark_weights = (
                self.benchmark_weights[None, :]
                .repeat(n_samples, 1)
                .to(device=device, dtype=dtype)
            )

        benchmark_rets = portfolio_returns(
            benchmark_weights,
            y[:, self.returns_channel, ...],
            input_type=self.input_type,
            output_type="simple",
        )  # (n_samples, horizon)

        cov = covariance(benchmark_rets, portfolio_rets)
        beta = cov / benchmark_rets.var(dim=1)
        alpha = portfolio_rets.mean(dim=1) - beta * benchmark_rets.mean(dim=1)

        return -alpha

    def __repr__(self):
        """Generate representation string."""
        return "{}(benchmark_weights={},returns_channel={}, input_type='{}')".format(
            self.__class__.__name__,
            self.benchmark_weights,
            self.returns_channel,
            self.input_type,
        )


class CumulativeReturn(Loss):
    """Negative cumulative returns.

    Parameters
    ----------
    returns_channel : int
        Which channel of the `y` target represents returns.

    input_type : str, {'log', 'simple'}
        What type of returns are we dealing with in `y`.
    """

    def __init__(self, returns_channel=0, input_type="log"):
        self.returns_channel = returns_channel
        self.input_type = input_type

    def __call__(self, weights, y):
        """Compute negative simple cumulative returns.

        Parameters
        ----------
        weights : torch.Tensor
            Tensor of shape `(n_samples, n_assets)` representing the predicted weights by our portfolio optimizer.

        y : torch.Tensor
            Tensor of shape `(n_samples, n_channels, horizon, n_assets)` representing the evolution over the next
            `horizon` timesteps.

        Returns
        -------
        torch.Tensor
            Tensor of shape `(n_samples,)` representing the per sample negative simple cumulative returns.

        """
        crets = portfolio_cumulative_returns(
            weights,
            y[:, self.returns_channel, ...],
            input_type=self.input_type,
            output_type="simple",
        )

        return -crets[:, -1]

    def __repr__(self):
        """Generate representation string."""
        return "{}(returns_channel={}, input_type='{}')".format(
            self.__class__.__name__, self.returns_channel, self.input_type
        )


class LargestWeight(Loss):
    """Largest weight loss.

    Loss function representing the largest weight among all the assets. It is supposed to encourage diversification
    since its minimal value is `1/n_asssets` for the equally weighted portfolio (assuming full investment).
    """

    def __init__(self):
        pass

    def __call__(self, weights, *args):
        """Compute largest weight.

        Parameters
        ----------
        weights : torch.Tensor
            Tensor of shape `(n_samples, n_assets)` representing the predicted weights by our portfolio optimizer.

        args : list
            Additional arguments. Just used for compatibility. Not used.

        Returns
        -------
        torch.Tensor
            Tensor of shape `(n_samples,)` representing the per sample largest weight.

        """
        return weights.max(dim=1)[0]

    def __repr__(self):
        """Generate representation string."""
        return "{}()".format(self.__class__.__name__)


class MaximumDrawdown(Loss):
    """Negative of the maximum drawdown."""

    def __init__(self, returns_channel=0, input_type="log"):
        self.returns_channel = returns_channel
        self.input_type = input_type

    def __call__(self, weights, y):
        """Compute maximum drawdown.

        Parameters
        ----------
        weights : torch.Tensor
            Tensor of shape `(n_samples, n_assets)` representing the predicted weights by our portfolio optimizer.

        y : torch.Tensor
            Tensor of shape `(n_samples, n_channels, horizon, n_assets)` representing the evolution over the next
            `horizon` timesteps.

        Returns
        -------
        torch.Tensor
            Tensor of shape `(n_samples,)` representing the per sample maximum drawdown.

        """
        cumrets = 1 + portfolio_cumulative_returns(
            weights,
            y[:, self.returns_channel, ...],
            input_type=self.input_type,
            output_type="simple",
        )

        cummax = torch.cummax(cumrets, 1)[0]  # (n_samples, n_timesteps)

        div = (cumrets / cummax) - 1  # (n_samples, n_timesteps)

        end = div.argmin(dim=1)  # (n_samples,)
        mdd = div.gather(1, end.view(-1, 1)).view(-1)

        return -mdd

    def __repr__(self):
        """Generate representation string."""
        return "{}(returns_channel={}, input_type='{}')".format(
            self.__class__.__name__, self.returns_channel, self.input_type
        )


class MeanReturns(Loss):
    """Negative mean returns."""

    def __init__(
        self, returns_channel=0, input_type="log", output_type="simple"
    ):
        self.returns_channel = returns_channel
        self.input_type = input_type
        self.output_type = output_type

    def __call__(self, weights, y):
        """Compute negative mean returns.

        Parameters
        ----------
        weights : torch.Tensor
            Tensor of shape `(n_samples, n_assets)` representing the predicted weights by our portfolio optimizer.

        y : torch.Tensor
            Tensor of shape `(n_samples, n_channels, horizon, n_assets)` representing the evolution over the next
            `horizon` timesteps.

        Returns
        -------
        torch.Tensor
            Tensor of shape `(n_samples,)` representing the per sample negative mean returns.

        """
        prets = portfolio_returns(
            weights,
            y[:, self.returns_channel, ...],
            input_type=self.input_type,
            output_type=self.output_type,
        )

        return -prets.mean(dim=1)

    def __repr__(self):
        """Generate representation string."""
        return (
            "{}(returns_channel={}, input_type='{}', output_type='{}')".format(
                self.__class__.__name__,
                self.returns_channel,
                self.input_type,
                self.output_type,
            )
        )


class Quantile(Loss):
    """Compute negative percentile.

    Parameters
    ----------
    q : float
        Number from (0, 1) representing the quantile.

    """

    def __init__(self, returns_channel=0, q=0.1):
        self.returns_channel = returns_channel
        self.q = q

    def __call__(self, weights, y):
        """Compute negative quantile.

        Parameters
        ----------
        weights : torch.Tensor
            Tensor of shape `(n_samples, n_assets)` representing the predicted weights by our portfolio optimizer.

        y : torch.Tensor
            Tensor of shape `(n_samples, n_channels, horizon, n_assets)` representing the evolution over the next
            `horizon` timesteps.

        Returns
        -------
        torch.Tensor
            Tensor of shape `(n_samples,)` representing the per sample negative quantile.

        """
        prets = portfolio_returns(
            weights, y[:, self.returns_channel, ...]
        )  # (n_samples, horizon)

        _, horizon = prets.shape

        k = 1 + round(self.q * (horizon - 1))
        return -prets.kthvalue(k)[0]

    def __repr__(self):
        """Generate representation string."""
        return "{}(returns_channel={})".format(
            self.__class__.__name__, self.returns_channel
        )


class SharpeRatio(Loss):
    """Negative Sharpe ratio.

    Parameters
    ----------
    rf : float
        Risk-free rate.

    returns_channel : int
        Which channel of the `y` target represents returns.

    input_type : str, {'log', 'simple'}
        What type of returns are we dealing with in `y`.

    output_type : str, {'log', 'simple'}
        What type of returns are we dealing with in the output.

    eps : float
        Additional constant added to the denominator to avoid division by zero.
    """

    def __init__(
        self,
        rf=0,
        returns_channel=0,
        input_type="log",
        output_type="simple",
        eps=1e-4,
    ):
        self.rf = rf
        self.returns_channel = returns_channel
        self.input_type = input_type
        self.output_type = output_type
        self.eps = eps

    def __call__(self, weights, y):
        """Compute negative sharpe ratio.

        Parameters
        ----------
        weights : torch.Tensor
            Tensor of shape `(n_samples, n_assets)` representing the predicted weights by our portfolio optimizer.

        y : torch.Tensor
            Tensor of shape `(n_samples, n_channels, horizon, n_assets)` representing the evolution over the next
            `horizon` timesteps.

        Returns
        -------
        torch.Tensor
            Tensor of shape `(n_samples,)` representing the per sample negative sharpe ratio.

        """
        prets = portfolio_returns(
            weights,
            y[:, self.returns_channel, ...],
            input_type=self.input_type,
            output_type=self.output_type,
        )

        return -(prets.mean(dim=1) - self.rf) / (prets.std(dim=1) + self.eps)

    def __repr__(self):
        """Generate representation string."""
        return "{}(rf={}, returns_channel={}, input_type='{}', output_type='{}', eps={})".format(
            self.__class__.__name__,
            self.rf,
            self.returns_channel,
            self.input_type,
            self.output_type,
            self.eps,
        )


class Softmax(Loss):
    """Softmax of per asset cumulative returns as the target."""

    def __init__(self, returns_channel=0):
        self.returns_channel = returns_channel

    def __call__(self, weights, y):
        """Compute softmax loss.

        Parameters
        ----------
        weights : torch.Tensor
            Tensor of shape `(n_samples, n_assets)` representing the predicted weights by our portfolio optimizer.

        y : torch.Tensor
            Tensor of shape `(n_samples, n_channels, horizon, n_assets)` representing the evolution over the next
            `horizon` timesteps.

        Returns
        -------
        torch.Tensor
            Tensor of shape `(n_samples,)` representing the per sample negative worst return over the horizon.

        """
        cumrets = y[:, self.returns_channel, ...].sum(dim=1)

        return ((weights - cumrets.softmax(dim=1)) ** 2).sum(dim=1)

    def __repr__(self):
        """Generate representation string."""
        return "{}(returns_channel={})".format(
            self.__class__.__name__, self.returns_channel
        )


class SortinoRatio(Loss):
    """Negative Sortino ratio.

    Parameters
    ----------
    rf : float
        Risk-free rate.

    returns_channel : int
        Which channel of the `y` target represents returns.

    input_type : str, {'log', 'simple'}
        What type of returns are we dealing with in `y`.

    output_type : str, {'log', 'simple'}
        What type of returns are we dealing with in the output.

    eps : float
        Additional constant added to the denominator to avoid division by zero.
    """

    def __init__(
        self,
        rf=0,
        returns_channel=0,
        input_type="log",
        output_type="simple",
        eps=1e-4,
    ):
        self.rf = rf
        self.returns_channel = returns_channel
        self.input_type = input_type
        self.output_type = output_type
        self.eps = eps

    def __call__(self, weights, y):
        """Compute negative Sortino ratio of portfolio return over the horizon.

        Parameters
        ----------
        weights : torch.Tensor
            Tensor of shape `(n_samples, n_assets)` representing the predicted weights by our portfolio optimizer.

        y : torch.Tensor
            Tensor of shape `(n_samples, n_channels, horizon, n_assets)` representing the evolution over the next
            `horizon` timesteps.

        Returns
        -------
        torch.Tensor
            Tensor of shape `(n_samples,)` representing the per sample negative worst return over the horizon.

        """
        prets = portfolio_returns(
            weights,
            y[:, self.returns_channel, ...],
            input_type=self.input_type,
            output_type=self.output_type,
        )

        return -(prets.mean(dim=1) - self.rf) / (
            torch.sqrt(torch.mean(torch.relu(-prets) ** 2, dim=1)) + self.eps
        )

    def __repr__(self):
        """Generate representation string."""
        return "{}(rf={}, returns_channel={}, input_type='{}', output_type='{}', eps={})".format(
            self.__class__.__name__,
            self.rf,
            self.returns_channel,
            self.input_type,
            self.output_type,
            self.eps,
        )


class SquaredWeights(Loss):
    """Sum of squared weights.

    Diversification loss. The equally weighted portfolio has a loss of `1 / n_assets`, the lowest possible. The
    single asset portfolio has a loss of 1.
    """

    def __init__(self):
        pass

    def __call__(self, weights, *args):
        """Compute sum of squared weights.

        Parameters
        ----------
        weights : torch.Tensor
            Tensor of shape `(n_samples, n_assets)` representing the predicted weights by our portfolio optimizer.

        args : list
            Additional arguments. Just used for compatibility. Not used.

        Returns
        -------
        torch.Tensor
            Tensor of shape `(n_samples,)` representing the per sample sum of squared weights.

        Notes
        -----
        If single asset then equal to `1`. If equally weighted portfolio then `1/N`.

        """
        return (weights**2).sum(dim=1)

    def __repr__(self):
        """Generate representation string."""
        return "{}()".format(self.__class__.__name__)


class StandardDeviation(Loss):
    """Standard deviation."""

    def __init__(
        self, returns_channel=0, input_type="log", output_type="simple"
    ):
        self.returns_channel = returns_channel
        self.input_type = input_type
        self.output_type = output_type

    def __call__(self, weights, y):
        """Compute standard deviation.

        Parameters
        ----------
        weights : torch.Tensor
            Tensor of shape `(n_samples, n_assets)` representing the predicted weights by our portfolio optimizer.

        y : torch.Tensor
            Tensor of shape `(n_samples, n_channels, horizon, n_assets)` representing the evolution over the next
            `horizon` timesteps.

        Returns
        -------
        torch.Tensor
            Tensor of shape `(n_samples,)` representing the per sample standard deviation.

        """
        prets = portfolio_returns(
            weights,
            y[:, self.returns_channel, ...],
            input_type=self.input_type,
            output_type=self.output_type,
        )

        return prets.std(dim=1)

    def __repr__(self):
        """Generate representation string."""
        return (
            "{}(returns_channel={}, input_type='{}', output_type='{}')".format(
                self.__class__.__name__,
                self.returns_channel,
                self.input_type,
                self.output_type,
            )
        )


class TargetMeanReturn(Loss):
    """Target mean return.

    Difference between some desired mean return and the realized one.
    """

    def __init__(
        self,
        target=0.01,
        p=2,
        returns_channel=0,
        input_type="log",
        output_type="simple",
    ):
        self.p = p
        self.target = target
        self.returns_channel = returns_channel
        self.input_type = input_type
        self.output_type = output_type

    def __call__(self, weights, y):
        """Compute distance from the target return.

        Parameters
        ----------
        weights : torch.Tensor
            Tensor of shape `(n_samples, n_assets)` representing the predicted weights by our portfolio optimizer.

        y : torch.Tensor
            Tensor of shape `(n_samples, n_channels, horizon, n_assets)` representing the evolution over the next
            `horizon` timesteps.

        Returns
        -------
        torch.Tensor
            Tensor of shape `(n_samples,)` representing the per sample negative mean returns.

        """

        def mapping(x):
            return abs(x - self.target) ** self.p

        prets = portfolio_returns(
            weights,
            y[:, self.returns_channel, ...],
            input_type=self.input_type,
            output_type=self.output_type,
        )

        return mapping(prets.mean(dim=1))  # (n_shapes,)

    def __repr__(self):
        """Generate representation string."""
        return "{}(target={}, p={}, returns_channel={}, input_type='{}', output_type='{}')".format(
            self.__class__.__name__,
            self.target,
            self.p,
            self.returns_channel,
            self.input_type,
            self.output_type,
        )


class TargetStandardDeviation(Loss):
    """Target standard deviation return.

    Difference between some desired standard deviation and the realized one.
    """

    def __init__(
        self,
        target=0.01,
        p=2,
        returns_channel=0,
        input_type="log",
        output_type="simple",
    ):
        self.p = p
        self.target = target
        self.returns_channel = returns_channel
        self.input_type = input_type
        self.output_type = output_type

    def __call__(self, weights, y):
        """Compute distance from the target return.

        Parameters
        ----------
        weights : torch.Tensor
            Tensor of shape `(n_samples, n_assets)` representing the predicted weights by our portfolio optimizer.

        y : torch.Tensor
            Tensor of shape `(n_samples, n_channels, horizon, n_assets)` representing the evolution over the next
            `horizon` timesteps.

        Returns
        -------
        torch.Tensor
            Tensor of shape `(n_samples,)` representing the per sample negative mean returns.

        """

        def mapping(x):
            return abs(x - self.target) ** self.p

        prets = portfolio_returns(
            weights,
            y[:, self.returns_channel, ...],
            input_type=self.input_type,
            output_type=self.output_type,
        )

        return mapping(prets.std(dim=1))  # (n_shapes,)

    def __repr__(self):
        """Generate representation string."""
        return "{}(target={}, p={}, returns_channel={}, input_type='{}', output_type='{}')".format(
            self.__class__.__name__,
            self.target,
            self.p,
            self.returns_channel,
            self.input_type,
            self.output_type,
        )


class WorstReturn(Loss):
    """Negative of the worst return.

    This loss is designed to discourage outliers - extremely low returns.
    """

    def __init__(
        self, returns_channel=0, input_type="log", output_type="simple"
    ):
        self.returns_channel = returns_channel
        self.input_type = input_type
        self.output_type = output_type

    def __call__(self, weights, y):
        """Compute negative of the worst return of the portfolio return over the horizon.

        Parameters
        ----------
        weights : torch.Tensor
            Tensor of shape `(n_samples, n_assets)` representing the predicted weights by our portfolio optimizer.

        y : torch.Tensor
            Tensor of shape `(n_samples, n_channels, horizon, n_assets)` representing the evolution over the next
            `horizon` timesteps.


        Returns
        -------
        torch.Tensor
            Tensor of shape `(n_samples,)` representing the per sample negative worst return over the horizon.

        """
        prets = portfolio_returns(
            weights,
            y[:, self.returns_channel, ...],
            input_type=self.input_type,
            output_type=self.output_type,
        )

        return -prets.topk(1, dim=1, largest=False)[0].view(-1)

    def __repr__(self):
        """Generate representation string."""
        return (
            "{}(returns_channel={}, input_type='{}', output_type='{}')".format(
                self.__class__.__name__,
                self.returns_channel,
                self.input_type,
                self.output_type,
            )
        )


class RiskParity(Loss):
    """Risk Parity Portfolio.

    Parameters
    ----------
    returns_channel : int
        Which channel of the `y` target represents returns.

    Attributes
    ----------
    covariance_layer : deepdow.layers.CoverianceMatrix
        Covarioance matrix layer.

    References
    ----------
    https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2297383
    """

    def __init__(self, returns_channel=0):
        self.returns_channel = returns_channel
        self.covariance_layer = CovarianceMatrix(sqrt=False)

    def __call__(self, weights, y):
        """Compute loss.

        Parameters
        ----------
        weights : torch.Tensor
            Tensor of shape `(n_samples, n_assets)` representing the predicted
            weights by our portfolio optimizer.

        y : torch.Tensor
            Tensor of shape `(n_samples, n_channels, horizon, n_assets)`
            representing the evolution over the next `horizon` timesteps.

        Returns
        -------
         torch.Tensor
            Tensor of shape `(n_samples,)` representing the per sample risk parity.
        """
        n_assets = weights.shape[-1]
        covar = self.covariance_layer(
            y[:, self.returns_channel, ...]
        )  # (n_samples, n_assets, n_assets)

        weights = weights.unsqueeze(dim=1)
        volatility = torch.sqrt(
            torch.matmul(
                weights, torch.matmul(covar, weights.permute((0, 2, 1)))
            )
        )  # (n_samples, 1, 1)
        c = (covar * weights) / volatility  # (n_samples, n_assets, n_assets)
        risk = volatility / n_assets  # (n_samples, 1, 1)

        budget = torch.matmul(weights, c)  # (n_samples, n_assets, n_assets)
        rp = torch.sum((risk - budget) ** 2, dim=-1).view(-1)  # (n_samples,)

        return rp

    def __repr__(self):
        """Generate representation string."""
        return "{}(returns_channel={})".format(
            self.__class__.__name__, self.returns_channel
        )


class DownsideRisk(Loss):
    """Downside Risk."""

    def __init__(
        self, beta=2, returns_channel=0, input_type="log", output_type="simple"
    ):
        self.beta = beta
        self.returns_channel = returns_channel
        self.input_type = input_type
        self.output_type = output_type

    def __call__(self, weights, y):
        """Compute the downside risk.

        Parameters
        ----------
        weights : torch.Tensor
            Tensor of shape `(n_samples, n_assets)` representing the predicted weights by our portfolio optimizer.

        y : torch.Tensor
            Tensor of shape `(n_samples, n_channels, horizon, n_assets)` representing the evolution over the next
            `horizon` timesteps.

        Returns
        -------
        torch.Tensor
            Tensor of shape `(n_samples,)` representing the per sample downside risk.

        """
        prets = portfolio_returns(
            weights,
            y[:, self.returns_channel, ...],
            input_type=self.input_type,
            output_type=self.output_type,
        )

        return torch.sqrt(
            torch.mean(
                torch.relu(-prets.sub(prets.mean(dim=1)[:, None]))
                ** self.beta,
                dim=1,
            )
        )

    def __repr__(self):
        """Generate representation string."""
        return "{}(beta={}, returns_channel={}, input_type='{}', output_type='{}')".format(
            self.__class__.__name__,
            self.beta,
            self.returns_channel,
            self.input_type,
            self.output_type,
        )
