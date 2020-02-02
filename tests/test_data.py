"""Tests focused on the data module."""
import numpy as np
import pandas as pd
import pytest

from deepdow.data import returns_to_Xy


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
