"""Test related to the `evaluate` module."""
import pandas as pd
import torch

from deepdow.benchmarks import OneOverN
from deepdow.evaluate import evaluate_models


class TestEvaluateModels:
    def test_basic(self):
        n_samples, lookback, n_assets, horizon = 3, 5, 4, 8

        X = torch.ones((n_samples, 1, lookback, n_assets)) * 0.01
        y = torch.ones((n_samples, horizon, n_assets)) * 0.02

        class TempModel:
            def __init__(self):
                self.was_fitted = False

            def fit(self, X):
                self.was_fitted = True
                return self

            def __call__(self, X):
                return torch.ones((X.shape[0], X.shape[-1]))

        temp_model = TempModel()
        models_dict = {'1overN': OneOverN(),
                       'temp': temp_model}

        results_means, results = evaluate_models(X, y, models_dict)

        assert isinstance(results_means, pd.DataFrame)
        assert isinstance(results, dict)
        assert temp_model.was_fitted
