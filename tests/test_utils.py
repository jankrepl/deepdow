"""Collection of tests focused on the utils module."""
import pathlib

import numpy as np
import pandas as pd
import pytest

from deepdow.utils import ChangeWorkingDirectory, PandasChecks, prices_to_returns, raw_to_Xy, returns_to_Xy


class TestChangeWorkingDirectory:
    def test_construction(self, tmpdir):
        dir_str = str(tmpdir)
        dir_path = pathlib.Path(dir_str)

        assert ChangeWorkingDirectory(dir_str).directory == dir_path
        assert ChangeWorkingDirectory(dir_path).directory == dir_path

        with pytest.raises(NotADirectoryError):
            ChangeWorkingDirectory('/fake/directory/')

    def test_working(self, tmpdir):
        dir_path = pathlib.Path(str(tmpdir))

        cwd_before = pathlib.Path.cwd()

        with ChangeWorkingDirectory(dir_path):
            cwd_inside = pathlib.Path.cwd()

        cwd_after = pathlib.Path.cwd()

        assert cwd_before == cwd_after
        assert cwd_before != cwd_inside
        assert cwd_inside == dir_path


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
    def test_dummy_(self, raw_data, use_log):
        prices_dummy, _, _ = raw_data

        returns = prices_to_returns(prices_dummy, use_log=use_log)

        assert isinstance(returns, pd.DataFrame)
        assert returns.index.equals(prices_dummy.index[1:])
        assert returns.columns.equals(prices_dummy.columns)

        if use_log:
            assert np.log(prices_dummy.iloc[2, 3] / prices_dummy.iloc[1, 3]) == pytest.approx(returns.iloc[1, 3])
        else:
            assert (prices_dummy.iloc[2, 3] / prices_dummy.iloc[1, 3]) - 1 == pytest.approx(returns.iloc[1, 3])


class TestRawToXy:
    def test_wrong(self, raw_data):
        df, n_missing_entries, true_freq = raw_data

        with pytest.raises(ValueError):
            raw_to_Xy(df, lookback=len(df) + n_missing_entries, freq=true_freq)

        with pytest.raises(ValueError):
            raw_to_Xy(df, freq=None)

    @pytest.mark.parametrize('included_assets',
                             [None, ['asset_1', 'asset_3']],
                             ids=['all_assets', 'some_assets'])
    @pytest.mark.parametrize('included_indicators',
                             [None, ['indicator_0', 'indicator_2', 'indicator_4']],
                             ids=['all_indicators', 'some_indicators'])
    def test_sanity_check(self, raw_data, included_assets, included_indicators):
        df, n_missing_entries, true_freq = raw_data

        n_timesteps = len(df)
        n_assets = len(included_assets or df.columns.levels[0])
        n_indicators = len(included_indicators or df.columns.levels[1])

        lookback = n_timesteps // 3
        horizon = n_timesteps // 4
        gap = 1

        X, timestamps, y, asset_names, indicators = raw_to_Xy(df,
                                                              lookback=lookback,
                                                              horizon=horizon,
                                                              gap=1,
                                                              freq=true_freq,
                                                              included_assets=included_assets,
                                                              included_indicators=included_indicators
                                                              )

        n_new = n_timesteps + n_missing_entries - lookback - horizon - gap + 1 - 1  # we start with prices

        # types
        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert isinstance(timestamps, pd.DatetimeIndex)
        assert timestamps.freq is not None and true_freq == timestamps.freq
        assert isinstance(asset_names, list)
        assert isinstance(indicators, list)

        # shapes
        assert X.shape == (n_new, n_indicators, lookback, n_assets)
        assert y.shape == (n_new, n_indicators, horizon, n_assets)
        assert timestamps[0] == pd.date_range(start=df.index[1], periods=lookback, freq=true_freq)[-1]  # prices
        assert len(asset_names) == n_assets
        assert len(indicators) == n_indicators

    def test_invalid_values(self, raw_data):
        df, n_missing_entries, true_freq = raw_data

        n_timesteps = len(df)
        n_assets = len(df.columns.levels[0])

        lookback = n_timesteps // 3
        horizon = n_timesteps // 4
        gap = 1

        df_invalid = df.copy()

        df_invalid.at[df.index[0], ('asset_1', 'indicator_3')] = -2

        X, timestamps, y, asset_names, indicators = raw_to_Xy(df_invalid,
                                                              lookback=lookback,
                                                              horizon=horizon,
                                                              gap=gap,
                                                              freq=true_freq
                                                              )

        assert ['asset_{}'.format(i) for i in range(n_assets) if i != 1] == asset_names


class TestReturnsToXY:

    @pytest.mark.parametrize('lookback', [3, 5])
    @pytest.mark.parametrize('horizon', [4, 6])
    def test_basic(self, raw_data, lookback, horizon):
        df, _, _ = raw_data

        returns_dummy = df.xs('indicator_1', axis=1, level=1)

        n_timesteps = len(returns_dummy.index)
        n_assets = len(returns_dummy.columns)
        n_samples = n_timesteps - lookback - horizon + 1

        X, timesteps, y = returns_to_Xy(returns_dummy, lookback=lookback, horizon=horizon)

        assert isinstance(X, np.ndarray)
        assert isinstance(timesteps, pd.DatetimeIndex)
        assert isinstance(y, np.ndarray)

        assert X.shape == (n_samples, 1, lookback, n_assets)
        assert len(timesteps) == n_samples
        assert y.shape == (n_samples, 1, horizon, n_assets)
