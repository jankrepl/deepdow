"""Collection of losses."""

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
        Logarithimc returnes

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
        Simple returns

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


def sharpe_ratio(weights, y, input_type='log', output_type='simple'):
    """Compute negative sharpe ratio per sample.

    Parameters
    ----------
    weights : torch.Tensor
        Tensor of shape `(n_samples, n_assets)` representing the predicted weights by our portfolio optimizer.

    y : torch.Tensor
        Tensor of shape `(n_samples, horizon, n_assets)` representing the return evolution over the next
        `horizon` timesteps.

    input_type : str, {'log', 'simple'}
        What type of returns are we dealing with in `y`.

    output_type : str, {'log', 'simple'}
        What type of returns are we dealing with in the output.

    Returns
    -------
    torch.Tensor
        Tensor of shape `(n_samples,)` representing the per sample negative sharpe ratios.

    """
    prets = portfolio_returns(weights, y, input_type=input_type, output_type=output_type)

    return -prets.mean(dim=1) / prets.std(dim=1)


def std(weights, y, input_type='log', output_type='simple'):
    """Compute standard deviation.

    Parameters
    ----------
    weights : torch.Tensor
        Tensor of shape `(n_samples, n_assets)` representing the predicted weights by our portfolio optimizer.

    y : torch.Tensor
        Tensor of shape `(n_samples, horizon, n_assets)` representing the return evolution over the next
        `horizon` timesteps.

    input_type : str, {'log', 'simple'}
        What type of returns are we dealing with in `y`.

    output_type : str, {'log', 'simple'}
        What type of returns are we dealing with in the output.

    Returns
    -------
    torch.Tensor
        Tensor of shape `(n_samples,)` representing the per sample standard deviation over the horizon.

    """
    prets = portfolio_returns(weights, y, input_type=input_type, output_type=output_type)

    return prets.std(dim=1)


def mean_returns(weights, y, input_type='log', output_type='simple'):
    """Compute negative mean portfolio return over the horizon.

    Parameters
    ----------
    weights : torch.Tensor
        Tensor of shape `(n_samples, n_assets)` representing the predicted weights by our portfolio optimizer.

    y : torch.Tensor
        Tensor of shape `(n_samples, horizon, n_assets)` representing the return evolution over the next
        `horizon` timesteps.

    input_type : str, {'log', 'simple'}
        What type of returns are we dealing with in `y`.

    output_type : str, {'log', 'simple'}
        What type of returns are we dealing with in the output.

    Returns
    -------
    torch.Tensor
        Tensor of shape `(n_samples,)` representing the per sample negative mean return over the horizon.

    """
    prets = portfolio_returns(weights, y, input_type=input_type, output_type=output_type)

    return -prets.mean(dim=1)


def worst_return(weights, y, input_type='log', output_type='simple'):
    """Compute negative of the worst return of the portfolio return over the horizon.

    Parameters
    ----------
    weights : torch.Tensor
        Tensor of shape `(n_samples, n_assets)` representing the predicted weights by our portfolio optimizer.

    y : torch.Tensor
        Tensor of shape `(n_samples, horizon, n_assets)` representing the return evolution over the next
        `horizon` timesteps.

    input_type : str, {'log', 'simple'}
        What type of returns are we dealing with in `y`.

    output_type : str, {'log', 'simple'}
        What type of returns are we dealing with in the output.

    Returns
    -------
    torch.Tensor
        Tensor of shape `(n_samples,)` representing the per sample negative worst return over the horizon.

    """
    prets = portfolio_returns(weights, y, input_type=input_type, output_type=output_type)

    return -prets.topk(1, dim=1, largest=False)[0].view(-1)


def sortino_ratio(weights, y, input_type='log', output_type='simple'):
    """Compute negative sortino ratio of portfolio return over the horizon.

    Parameters
    ----------
    weights : torch.Tensor
        Tensor of shape `(n_samples, n_assets)` representing the predicted weights by our portfolio optimizer.

    y : torch.Tensor
        Tensor of shape `(n_samples, horizon, n_assets)` representing the return evolution over the next
        `horizon` timesteps.

    input_type : str, {'log', 'simple'}
        What type of returns are we dealing with in `y`.

    output_type : str, {'log', 'simple'}
        What type of returns are we dealing with in the output.

    Returns
    -------
    torch.Tensor
        Tensor of shape `(n_samples,)` representing the per sample negative worst return over the horizon.

    """
    prets = portfolio_returns(weights, y, input_type=input_type, output_type=output_type)

    return -prets.mean(dim=1) / torch.sqrt(torch.mean(torch.relu(-prets) ** 2, dim=1))


def single_punish(weights):
    """Compute negative of the largest weight.

    This loss function reaches a minimum for equally weighted portfolio (assuming we have the full investment
    constraint).

    Parameters
    ----------
    weights : torch.Tensor
        Tensor of shape `(n_samples, n_assets)` representing the predicted weights by our portfolio optimizer.

    Returns
    -------
    torch.Tensor
        Tensor of shape `(n_samples,)` representing the per sample negative largest weight.
    """
    return weights.max(dim=1)[0]


def number_of_unused_assets(weights):
    """Compute number of assets with zero weights.

    Parameters
    ----------
    weights : torch.Tensor
        Tensor of shape `(n_samples, n_assets)` representing the predicted weights by our portfolio optimizer.

    Returns
    -------
    torch.Tensor
        Tensor of shape `(n_samples,)` representing the per sample number of assets with zero weights.

    """
    return (weights == 0).sum(dim=1)


def squared_weights(weights):
    """Squared weights.

    Parameters
    ----------
    weights : torch.Tensor
        Tensor of shape `(n_samples, n_assets)` representing the predicted weights by our portfolio optimizer.

    Returns
    -------
    torch.Tensor
        Tensor of shape `(n_samples,)` representing the per sample squared weights.

    """
    return (weights ** 2).sum(dim=1)
