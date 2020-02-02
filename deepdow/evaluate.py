"""Evaluation of performance."""

import pandas as pd

from deepdow.losses import mean_returns, sharpe_ratio, sortino_ratio, std, worst_return


def evaluate_models(X, y, models_dict):
    """Evaluate performance of models.

    Parameters
    ----------
    X : torch.Tensor
        Tensor of shape `(n_samples, 1, lookback, n_assets)` representing the features.

    y : torch.Tensor
        Tensor of shape `(n_samples, horizon, n_assets)` representing the labels.

    models_dict : dict
        Values represent model names and values are instances of classes implementing `__call__` for prediction. Also
        potentially `fit` method to fit the model.

    Returns
    -------
    results : dict
        Keys are model names values are `pd.DataFrame` with rows being samples and columns being losses.

    results_means : pd.DataFrame
        Rows represent different loss functions, columns are different models.
    """
    used_losses = {'mean_returns': mean_returns,
                   'sharpe_ratio': sharpe_ratio,
                   'sortino_ratio': sortino_ratio,
                   'std': std,
                   'worst_return': worst_return}

    results = {}

    for model_name, model in models_dict.items():
        if hasattr(model, 'fit'):
            model.fit(X)

        try:
            weights_pred = model(X, debug_mode=False)
        except TypeError:
            weights_pred = model(X)

        results[model_name] = pd.DataFrame(
            {loss_name: loss(weights_pred, y).detach().numpy() for loss_name, loss in used_losses.items()})

    results_means = pd.DataFrame({model_name: res.mean() for model_name, res in results.items()})

    return results_means, results
