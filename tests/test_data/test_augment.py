"""Collection of tests focused on the `deepdow.data.augment`."""

import numpy as np
import pytest

from deepdow.data import prepare_robust_scaler, prepare_standard_scaler


@pytest.mark.parametrize('overlap', [True, False])
@pytest.mark.parametrize('indices', [None, [1, 4, 6]])
def test_prepare_standard_scaler(overlap, indices):
    n_samples, n_channels, lookback, n_assets = 10, 3, 5, 12

    X = np.random.random((n_samples, n_channels, lookback, n_assets)) - 0.5

    means, stds = prepare_standard_scaler(X, overlap=overlap, indices=indices)

    assert means.shape == (n_channels,)
    assert stds.shape == (n_channels,)
    assert np.all(stds > 0)


class TestPrepareRobustScaler:

    def test_error(self):
        with pytest.raises(ValueError):
            prepare_robust_scaler(np.ones((1, 2, 3, 4)), percentile_range=(20, 10))

        with pytest.raises(ValueError):
            prepare_robust_scaler(np.ones((1, 2, 3, 4)), percentile_range=(-2, 99))

    @pytest.mark.parametrize('overlap', [True, False])
    @pytest.mark.parametrize('indices', [None, [1, 4, 6]])
    def test_basic(self, overlap, indices):
        n_samples, n_channels, lookback, n_assets = 10, 3, 5, 12

        X = np.random.random((n_samples, n_channels, lookback, n_assets)) - 0.5

        medians, ranges = prepare_robust_scaler(X, overlap=overlap, indices=indices)

        assert medians.shape == (n_channels,)
        assert ranges.shape == (n_channels,)
        assert np.all(ranges > 0)

    def test_sanity(self):
        n_samples, n_channels, lookback, n_assets = 10, 3, 5, 12

        X = np.random.random((n_samples, n_channels, lookback, n_assets)) - 0.5

        medians_1, ranges_1 = prepare_robust_scaler(X, percentile_range=(20, 80))
        medians_2, ranges_2 = prepare_robust_scaler(X, percentile_range=(10, 90))

        assert np.all(ranges_2 > ranges_1)
