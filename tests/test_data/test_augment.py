"""Collection of tests focused on the `deepdow.data.augment`."""

import numpy as np
import pytest
import torch

from deepdow.data import (
    Compose,
    Dropout,
    Multiply,
    Noise,
    Scale,
    prepare_robust_scaler,
    prepare_standard_scaler,
)


@pytest.mark.parametrize(
    "tform",
    [
        Compose(
            [
                lambda a, b, c, d: (2 * a, b, c, d),
                lambda a, b, c, d: (3 + a, b, c, d),
            ]
        ),
        Dropout(p=0.5),
        Multiply(c=4),
        Noise(0.3),
        Scale(np.array([1.2]), np.array([5.7])),
    ],
)
def test_tforms_not_in_place_for_x(tform):
    X = torch.randn(1, 4, 5)
    X_orig = X.clone()

    X_after, _, _, _ = tform(X, None, None, None)

    assert torch.allclose(X, X_orig)
    assert not torch.allclose(X_after, X)
    assert X_after.shape == X.shape


@pytest.mark.parametrize("overlap", [True, False])
@pytest.mark.parametrize("indices", [None, [1, 4, 6]])
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
            prepare_robust_scaler(
                np.ones((1, 2, 3, 4)), percentile_range=(20, 10)
            )

        with pytest.raises(ValueError):
            prepare_robust_scaler(
                np.ones((1, 2, 3, 4)), percentile_range=(-2, 99)
            )

    @pytest.mark.parametrize("overlap", [True, False])
    @pytest.mark.parametrize("indices", [None, [1, 4, 6]])
    def test_basic(self, overlap, indices):
        n_samples, n_channels, lookback, n_assets = 10, 3, 5, 12

        X = np.random.random((n_samples, n_channels, lookback, n_assets)) - 0.5

        medians, ranges = prepare_robust_scaler(
            X, overlap=overlap, indices=indices
        )

        assert medians.shape == (n_channels,)
        assert ranges.shape == (n_channels,)
        assert np.all(ranges > 0)

    def test_sanity(self):
        n_samples, n_channels, lookback, n_assets = 10, 3, 5, 12

        X = np.random.random((n_samples, n_channels, lookback, n_assets)) - 0.5

        medians_1, ranges_1 = prepare_robust_scaler(
            X, percentile_range=(20, 80)
        )
        medians_2, ranges_2 = prepare_robust_scaler(
            X, percentile_range=(10, 90)
        )

        assert np.all(ranges_2 > ranges_1)


class TestScaler:
    def test_erorrs(self):
        with pytest.raises(ValueError):
            raise Scale(np.ones(3), np.ones(4))

        with pytest.raises(ValueError):
            raise Scale(np.array([1, -1]), np.array([9, -0.1]))

        tform = Scale(np.array([1, -1]), np.array([9, 10.0]))
        with pytest.raises(ValueError):
            tform(torch.rand(3, 4, 5), None, None, None)

    def test_overall(self):
        n_channels, lookback, n_assets = 3, 5, 12

        X = np.random.random((n_channels, lookback, n_assets))
        X_torch = torch.as_tensor(X)
        dtype = X_torch.dtype

        center = X.mean(axis=(1, 2))
        scale = X.std(
            axis=(1, 2),
        )

        tform = Scale(center, scale)
        X_scaled = tform(X_torch, None, None, None)[0]

        assert torch.is_tensor(X_scaled)
        assert X_torch.shape == X_scaled.shape
        assert not torch.allclose(X_torch, X_scaled)
        assert torch.allclose(
            X_scaled.mean(dim=(1, 2)), torch.zeros(n_channels, dtype=dtype)
        )
        assert torch.allclose(
            X_scaled.std(dim=(1, 2), unbiased=False),
            torch.ones(n_channels, dtype=dtype),
        )
