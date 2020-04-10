"""Collection of losses.

All losses are designed for minimization.
"""
from types import MethodType

import torch


def simple2log(x):
    """Turn simple returns into log returns.

    r_simple = exp(r_log) - 1.

    Parameters
    ----------
    x : torch.Tensor
        Tensor of any shape where each entry represents a simple return.

    Returns
    -------
    torch.Tensor
        Logarithmic returns.

    """
    return torch.exp(x) - 1


def log2simple(x):
    """Turn log returns into simple returns.

    r_log = ln(r_simple + 1).

    Parameters
    ----------
    x : torch.Tensor
        Tensor of any shape where each entry represents a logarithmic return.

    Returns
    -------
    torch.Tensor
        Simple returns.

    """
    return torch.log(x + 1)


def portfolio_returns(weights, y, input_type='log', output_type='simple'):
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

    Returns
    -------
    portfolio_returns : torch.Tensor
        Of shape (n_samples, horizon)

    """
    if input_type == 'log':
        simple_returns = log2simple(y)

    elif input_type == 'simple':
        simple_returns = y

    else:
        raise ValueError('Unsupported input type: {}'.format(input_type))

    n_samples, horizon, n_assets = simple_returns.shape

    res = []

    for i in range(n_samples):
        res.append(simple_returns[i] @ weights[i])  # (horizon, n_assets)x(n_assets)=(horizon,)

    out = torch.stack(res, dim=0)

    if output_type == 'log':
        return simple2log(out)

    elif output_type == 'simple':
        return out

    else:
        raise ValueError('Unsupported output type: {}'.format(output_type))


def portfolio_cumulative_returns(weights, y, input_type='log', output_type='simple'):
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

    Returns
    -------
    torch.Tensor
        Tensor of shape `(n_samples, horizon)`.

    """
    prets = portfolio_returns(weights, y, input_type=input_type, output_type='log')  # we can aggregate overtime by sum
    log_prets = torch.cumsum(prets, dim=1)

    if output_type == 'log':
        return log_prets

    elif output_type == 'simple':
        return log2simple(log_prets)

    else:
        raise ValueError('Unsupported output type: {}'.format(output_type))


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
            new_instance._call = MethodType(lambda inst, weights, y: self(weights, y) + other(weights, y), new_instance)
            new_instance._repr = MethodType(lambda inst: '{} + {}'.format(self.__repr__(), other.__repr__()),
                                            new_instance)

            return new_instance

        elif isinstance(other, (int, float)):
            new_instance = Loss()
            new_instance._call = MethodType(lambda inst, weights, y: self(weights, y) + other, new_instance)
            new_instance._repr = MethodType(lambda inst: '{} + {}'.format(self.__repr__(), other),
                                            new_instance)

            return new_instance
        else:
            raise TypeError('Unsupported type: {}'.format(type(other)))

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
            new_instance._call = MethodType(lambda inst, weights, y: self(weights, y) * other(weights, y), new_instance)
            new_instance._repr = MethodType(lambda inst: '{} * {}'.format(self.__repr__(), other.__repr__()),
                                            new_instance)

            return new_instance

        elif isinstance(other, (int, float)):
            new_instance = Loss()
            new_instance._call = MethodType(lambda inst, weights, y: self(weights, y) * other, new_instance)
            new_instance._repr = MethodType(lambda inst: '{} * {}'.format(self.__repr__(), other),
                                            new_instance)

            return new_instance
        else:
            raise TypeError('Unsupported type: {}'.format(type(other)))

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
            new_instance._call = MethodType(lambda inst, weights, y: self(weights, y) / other(weights, y), new_instance)
            new_instance._repr = MethodType(lambda inst: '{} / {}'.format(self.__repr__(), other.__repr__()),
                                            new_instance)

            return new_instance

        elif isinstance(other, (int, float)):
            if other == 0:
                raise ZeroDivisionError()

            new_instance = Loss()
            new_instance._call = MethodType(lambda inst, weights, y: self(weights, y) / other, new_instance)
            new_instance._repr = MethodType(lambda inst: '{} / {}'.format(self.__repr__(), other),
                                            new_instance)

            return new_instance
        else:
            raise TypeError('Unsupported type: {}'.format(type(other)))

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
            new_instance._call = MethodType(lambda inst, weights, y: self(weights, y) ** power, new_instance)
            new_instance._repr = MethodType(lambda inst: '({}) ** {}'.format(self.__repr__(), power),
                                            new_instance)

            return new_instance
        else:
            raise TypeError('Unsupported type: {}'.format(type(power)))


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


class MeanReturns(Loss):
    """Negative mean returns."""

    def __init__(self, returns_channel=0, input_type='log', output_type='simple'):
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
        prets = portfolio_returns(weights,
                                  y[:, self.returns_channel, ...],
                                  input_type=self.input_type,
                                  output_type=self.output_type)

        return -prets.mean(dim=1)

    def __repr__(self):
        """Generate representation string."""
        return "{}(returns_channel={}, input_type='{}', output_type='{}')".format(self.__class__.__name__,
                                                                                  self.returns_channel,
                                                                                  self.input_type,
                                                                                  self.output_type)


class SharpeRatio(Loss):
    """Negative Sharpe ratio."""

    def __init__(self, returns_channel=0, input_type='log', output_type='simple'):
        self.returns_channel = returns_channel
        self.input_type = input_type
        self.output_type = output_type

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
        prets = portfolio_returns(weights,
                                  y[:, self.returns_channel, ...],
                                  input_type=self.input_type,
                                  output_type=self.output_type)

        return -prets.mean(dim=1) / prets.std(dim=1)

    def __repr__(self):
        """Generate representation string."""
        return "{}(returns_channel={}, input_type='{}', output_type='{}')".format(self.__class__.__name__,
                                                                                  self.returns_channel,
                                                                                  self.input_type,
                                                                                  self.output_type)


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
        return "{}(returns_channel={})".format(self.__class__.__name__, self.returns_channel)


class SortinoRatio(Loss):
    """Negative Sortino ratio."""

    def __init__(self, returns_channel=0, input_type='log', output_type='simple'):
        self.returns_channel = returns_channel
        self.input_type = input_type
        self.output_type = output_type

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
        prets = portfolio_returns(weights,
                                  y[:, self.returns_channel, ...],
                                  input_type=self.input_type,
                                  output_type=self.output_type)

        return -prets.mean(dim=1) / (torch.sqrt(torch.mean(torch.relu(-prets) ** 2, dim=1)) + 1e-6)

    def __repr__(self):
        """Generate representation string."""
        return "{}(returns_channel={}, input_type='{}', output_type='{}')".format(self.__class__.__name__,
                                                                                  self.returns_channel,
                                                                                  self.input_type,
                                                                                  self.output_type)


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
        return (weights ** 2).sum(dim=1)

    def __repr__(self):
        """Generate representation string."""
        return "{}()".format(self.__class__.__name__)


class StandardDeviation(Loss):
    """Standard deviation."""

    def __init__(self, returns_channel=0, input_type='log', output_type='simple'):
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
        prets = portfolio_returns(weights,
                                  y[:, self.returns_channel, ...],
                                  input_type=self.input_type,
                                  output_type=self.output_type)

        return prets.std(dim=1)

    def __repr__(self):
        """Generate representation string."""
        return "{}(returns_channel={}, input_type='{}', output_type='{}')".format(self.__class__.__name__,
                                                                                  self.returns_channel,
                                                                                  self.input_type,
                                                                                  self.output_type)


class TargetMeanReturn(Loss):
    """Target mean return.

    Difference between some desired mean return and the realized one.
    """

    def __init__(self, target=0.01, p=2, returns_channel=0, input_type='log', output_type='simple'):
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

        prets = portfolio_returns(weights,
                                  y[:, self.returns_channel, ...],
                                  input_type=self.input_type,
                                  output_type=self.output_type)

        return mapping(prets.mean(dim=1))  # (n_shapes,)

    def __repr__(self):
        """Generate representation string."""
        return "{}(target={}, p={}, returns_channel={}, input_type='{}', output_type='{}')".format(
            self.__class__.__name__,
            self.target,
            self.p,
            self.returns_channel,
            self.input_type,
            self.output_type)


class TargetStandardDeviation(Loss):
    """Target standard deviation return.

    Difference between some desired standard deviation and the realized one.
    """

    def __init__(self, target=0.01, p=2, returns_channel=0, input_type='log', output_type='simple'):
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

        prets = portfolio_returns(weights,
                                  y[:, self.returns_channel, ...],
                                  input_type=self.input_type,
                                  output_type=self.output_type)

        return mapping(prets.std(dim=1))  # (n_shapes,)

    def __repr__(self):
        """Generate representation string."""
        return "{}(target={}, p={}, returns_channel={}, input_type='{}', output_type='{}')".format(
            self.__class__.__name__,
            self.target,
            self.p,
            self.returns_channel,
            self.input_type,
            self.output_type)


class WorstReturn(Loss):
    """Negative of the worst return.

    This loss is designed to discourage outliers - extremely low returns.
    """

    def __init__(self, returns_channel=0, input_type='log', output_type='simple'):
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
        prets = portfolio_returns(weights,
                                  y[:, self.returns_channel, ...],
                                  input_type=self.input_type,
                                  output_type=self.output_type)

        return -prets.topk(1, dim=1, largest=False)[0].view(-1)

    def __repr__(self):
        """Generate representation string."""
        return "{}(returns_channel={}, input_type='{}', output_type='{}')".format(self.__class__.__name__,
                                                                                  self.returns_channel,
                                                                                  self.input_type,
                                                                                  self.output_type)
