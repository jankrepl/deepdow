"""Collection of tests focused on the utils module."""

import numpy as np
import pandas as pd
import pytest

from deepdow.utils import PandasChecks, prices_to_returns, returns_to_Xy

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

class TestPandasChecks:

    def test_check_no_gaps(self):
        index_incorrect = [1, 2]
        index_without_gaps = pd.date_range('1/1/2000', periods=4, freq='M')
        index_with_gaps = pd.DatetimeIndex([x for i, x in enumerate(index_without_gaps) if i != 2])

        with pytest.raises(TypeError):
            PandasChecks.check_no_gaps(index_incorrect)

        with pytest.raises(IndexError):
            PandasChecks.check_no_gaps(index_with_gaps)

        PandasChecks.check_no_gaps(index_without_gaps)

    def test_check_valid_entries(self):
        table_incorrect = 'table'
        table_invalid_1 = pd.Series([1, np.nan])
        table_invalid_2 = pd.DataFrame([[1, 2], [np.inf, 3]])
        table_valid = pd.DataFrame([[1, 2], [2, 4]])

        with pytest.raises(TypeError):
            PandasChecks.check_valid_entries(table_incorrect)

        with pytest.raises(ValueError):
            PandasChecks.check_valid_entries(table_invalid_1)

        with pytest.raises(ValueError):
            PandasChecks.check_valid_entries(table_invalid_2)

        PandasChecks.check_valid_entries(table_valid)

    def test_indices_agree(self):
        index_correct = ['A', 'B']
        index_wrong = ['A', 'C']

        with pytest.raises(TypeError):
            PandasChecks.check_indices_agree([], 'a')

        with pytest.raises(IndexError):
            PandasChecks.check_indices_agree(pd.Series(index=index_correct), pd.Series(index=index_wrong))

        with pytest.raises(IndexError):
            PandasChecks.check_indices_agree(pd.Series(index=index_correct), pd.DataFrame(index=index_correct,
                                                                                          columns=index_wrong))

        PandasChecks.check_indices_agree(pd.Series(index=index_correct), pd.DataFrame(index=index_correct,
                                                                                      columns=index_correct))


class TestPricesToReturns:

    @pytest.mark.parametrize('use_log', [True, False])
    def test_dummy(self, prices_dummy, use_log):
        returns = prices_to_returns(prices_dummy, use_log=use_log)

        assert isinstance(returns, pd.DataFrame)
        assert returns.index.equals(prices_dummy.index[1:])
        assert returns.columns.equals(prices_dummy.columns)

        if use_log:
            assert np.log(prices_dummy.iloc[2, 3] / prices_dummy.iloc[1, 3]) == pytest.approx(returns.iloc[1, 3])
        else:
            assert (prices_dummy.iloc[2, 3] / prices_dummy.iloc[1, 3]) - 1 == pytest.approx(returns.iloc[1, 3])
