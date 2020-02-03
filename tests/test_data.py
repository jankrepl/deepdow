"""Tests focused on the data module."""
import numpy as np
import pandas as pd
import pytest
import torch

from deepdow.data import InRAMDataset, returns_to_Xy


class TestInRAMDataset:
    def test_incorrect_input(self):
        with pytest.raises(ValueError):
            InRAMDataset(np.zeros((2, 1, 3, 4)), np.zeros((3, 5, 4)))

        with pytest.raises(ValueError):
            InRAMDataset(np.zeros((2, 1, 3, 4)), np.zeros((2, 9, 5)))

    def test_default_device(self):
        dset = InRAMDataset(np.zeros((2, 1, 3, 4)), np.zeros((2, 6, 4)), device=None)

        assert dset.device == torch.device('cpu')

    @pytest.mark.parametrize('n_samples', [1, 3, 6])
    def test_lenght(self, n_samples):
        dset = InRAMDataset(np.zeros((n_samples, 1, 3, 4)), np.zeros((n_samples, 6, 4)))

        assert len(dset) == n_samples

    def test_get_item(self):
        n_samples = 3

        X = np.zeros((n_samples, 1, 3, 4))
        y = np.zeros((n_samples, 6, 4))

        for i in range(n_samples):
            X[i] = i
            y[i] = i

        dset = InRAMDataset(X, y)

        for i in range(n_samples):
            X_sample, y_sample = dset[i]

            assert torch.is_tensor(X_sample)
            assert torch.is_tensor(y_sample)

            assert X_sample.shape == (1, 3, 4)
            assert y_sample.shape == (6, 4)

            assert torch.allclose(X_sample, torch.ones_like(X_sample) * i)
            assert torch.allclose(y_sample, torch.ones_like(y_sample) * i)


class TestReturnsToXY:

    @pytest.mark.parametrize('lookback', [3, 5])
    @pytest.mark.parametrize('horizon', [4, 6])
    def test_basic(self, returns_dummy, lookback, horizon):
        n_timesteps = len(returns_dummy.index)
        n_assets = len(returns_dummy.columns)
        n_samples = n_timesteps - lookback - horizon + 1

        X, timesteps, y = returns_to_Xy(returns_dummy, lookback=lookback, horizon=horizon)

        assert isinstance(X, np.ndarray)
        assert isinstance(timesteps, pd.DatetimeIndex)
        assert isinstance(y, np.ndarray)

        assert X.shape == (n_samples, 1, lookback, n_assets)
        assert len(timesteps) == n_samples
        assert y.shape == (n_samples, horizon, n_assets)
