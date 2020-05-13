"""Tests focused on the data module."""
import datetime
import numpy as np
import pytest
import torch

from deepdow.data import (Compose, Dropout, FlexibleDataLoader, InRAMDataset, Multiply, Noise, RigidDataLoader,
                          collate_uniform, scale_features)


class TestCollateUniform:

    def test_incorrect_input(self):
        with pytest.raises(ValueError):
            collate_uniform([], n_assets_range=(-2, 0))

        with pytest.raises(ValueError):
            collate_uniform([], lookback_range=(3, 1))

        with pytest.raises(ValueError):
            collate_uniform([], horizon_range=(10, 10))

    def test_dummy(self):
        n_samples = 14
        max_n_assets = 10
        max_lookback = 8
        max_horizon = 5
        n_channels = 2

        batch = [(torch.zeros((n_channels, max_lookback, max_n_assets)),
                  torch.ones((n_channels, max_horizon, max_n_assets)),
                  datetime.datetime.now(),
                  ['asset_{}'.format(i) for i in range(max_n_assets)]) for _ in
                 range(n_samples)]

        X_batch, y_batch, timestamps_batch, asset_names_batch = collate_uniform(batch,
                                                                                n_assets_range=(5, 6),
                                                                                lookback_range=(4, 5),
                                                                                horizon_range=(3, 4))

        assert torch.is_tensor(X_batch)
        assert torch.is_tensor(y_batch)

        assert X_batch.shape == (n_samples, n_channels, 4, 5)
        assert y_batch.shape == (n_samples, n_channels, 3, 5)
        assert len(timestamps_batch) == n_samples
        assert len(asset_names_batch) == 5

    @pytest.mark.parametrize('scaler', [None, 'standard', 'percent'])
    def test_replicable(self, scaler):
        random_state_a = 3
        random_state_b = 5

        n_samples = 14
        max_n_assets = 10
        max_lookback = 8
        max_horizon = 5
        n_channels = 2

        batch = [(torch.rand((n_channels, max_lookback, max_n_assets)),
                  torch.rand((n_channels, max_horizon, max_n_assets)),
                  datetime.datetime.now(),
                  ['asset_{}'.format(i) for i in range(max_n_assets)]) for _ in
                 range(n_samples)]

        X_batch_1, y_batch_1, _, _ = collate_uniform(batch,
                                                     random_state=random_state_a,
                                                     n_assets_range=(4, 5),
                                                     lookback_range=(4, 5),
                                                     horizon_range=(3, 4),
                                                     scaler=scaler)
        X_batch_2, y_batch_2, _, _ = collate_uniform(batch,
                                                     random_state=random_state_a,
                                                     n_assets_range=(4, 5),
                                                     lookback_range=(4, 5),
                                                     horizon_range=(3, 4),
                                                     scaler=scaler)

        X_batch_3, y_batch_3, _, _ = collate_uniform(batch,
                                                     random_state=random_state_b,
                                                     n_assets_range=(4, 5),
                                                     lookback_range=(4, 5),
                                                     horizon_range=(3, 4),
                                                     scaler=scaler)

        assert torch.allclose(X_batch_1, X_batch_2)
        assert torch.allclose(y_batch_1, y_batch_2)

        assert not torch.allclose(X_batch_3, X_batch_1)
        assert not torch.allclose(y_batch_3, y_batch_1)

    def test_different(self):
        n_samples = 6
        max_n_assets = 27
        max_lookback = 15
        max_horizon = 12

        n_channels = 2
        batch = [(torch.rand((n_channels, max_lookback, max_n_assets)),
                  torch.rand((n_channels, max_horizon, max_n_assets)),
                  datetime.datetime.now(),
                  ['asset_{}'.format(i) for i in range(max_n_assets)]) for _ in
                 range(n_samples)]
        n_trials = 10

        n_assets_set = set()
        lookback_set = set()
        horizon_set = set()

        for _ in range(n_trials):
            X_batch, y_batch, timestamps_batch, asset_names_batch = collate_uniform(batch,
                                                                                    n_assets_range=(2, max_n_assets),
                                                                                    lookback_range=(2, max_lookback),
                                                                                    horizon_range=(2, max_lookback))

            n_assets_set.add(X_batch.shape[-1])
            lookback_set.add(X_batch.shape[-2])
            horizon_set.add(y_batch.shape[-2])

        assert len(n_assets_set) > 1
        assert len(lookback_set) > 1
        assert len(horizon_set) > 1


class TestInRAMDataset:
    def test_incorrect_input(self):
        with pytest.raises(ValueError):
            InRAMDataset(np.zeros((2, 1, 3, 4)), np.zeros((3, 1, 5, 4)))

        with pytest.raises(ValueError):
            InRAMDataset(np.zeros((2, 1, 3, 4)), np.zeros((2, 2, 6, 4)))

        with pytest.raises(ValueError):
            InRAMDataset(np.zeros((2, 1, 3, 4)), np.zeros((2, 1, 3, 6)))

    @pytest.mark.parametrize('n_samples', [1, 3, 6])
    def test_lenght(self, n_samples):
        dset = InRAMDataset(np.zeros((n_samples, 1, 3, 4)), np.zeros((n_samples, 1, 6, 4)))

        assert len(dset) == n_samples

    def test_get_item(self):
        n_samples = 3

        n_channels = 3

        X = np.zeros((n_samples, n_channels, 3, 4))
        y = np.zeros((n_samples, n_channels, 6, 4))

        for i in range(n_samples):
            X[i] = i
            y[i] = i

        dset = InRAMDataset(X, y)

        for i in range(n_samples):
            X_sample, y_sample, _, _ = dset[i]

            assert torch.is_tensor(X_sample)
            assert torch.is_tensor(y_sample)

            assert X_sample.shape == (n_channels, 3, 4)
            assert y_sample.shape == (n_channels, 6, 4)

            assert torch.allclose(X_sample, torch.ones_like(X_sample) * i)
            assert torch.allclose(y_sample, torch.ones_like(y_sample) * i)

    def test_transforms(self):
        n_samples = 13
        n_channels = 2
        lookback = 9
        horizon = 10
        n_assets = 6

        X = np.random.normal(size=(n_samples, n_channels, lookback, n_assets)) / 100
        y = np.random.normal(size=(n_samples, n_channels, horizon, n_assets)) / 100

        dataset = InRAMDataset(X, y, transform=Compose([Noise(), Dropout(p=0.5), Multiply(c=100)]))

        X_sample, y_sample, timestamps_sample, asset_names = dataset[1]

        assert (X_sample == 0).sum() > 0  # dropout
        assert X_sample.max() > 1  # multiply 100
        assert X_sample.min() < -1  # multiply 100

        assert (y_sample == 0).sum() == 0
        assert y_sample.max() < 1
        assert y_sample.min() > -1


@pytest.mark.parametrize('scaler', ['standard', 'percent', 'wrong'])
def test_scale_features(scaler):
    X = torch.rand(2, 3, 4, 5)

    if scaler == 'wrong':
        with pytest.raises(ValueError):
            scale_features(X, approach=scaler)
    else:
        X_scaled = scale_features(X, approach=scaler)

        assert X_scaled.shape == X.shape


class TestFlexibleDataLoader:
    def test_wrong_construction(self, dataset_dummy):
        max_assets = dataset_dummy.n_assets
        max_lookback = dataset_dummy.lookback
        max_horizon = dataset_dummy.horizon

        with pytest.raises(ValueError):
            FlexibleDataLoader(dataset_dummy,
                               indices=None,
                               asset_ixs=list(range(len(dataset_dummy))),
                               n_assets_range=(max_assets, max_assets + 1),
                               lookback_range=(max_lookback, max_lookback + 1),
                               horizon_range=(-2, max_horizon + 1))

        with pytest.raises(ValueError):
            FlexibleDataLoader(dataset_dummy,
                               indices=[-1],
                               n_assets_range=(max_assets, max_assets + 1),
                               lookback_range=(max_lookback, max_lookback + 1),
                               horizon_range=(max_horizon, max_horizon + 1))

        with pytest.raises(ValueError):
            FlexibleDataLoader(dataset_dummy,
                               indices=None,
                               n_assets_range=(max_assets, max_assets + 2),
                               lookback_range=(max_lookback, max_lookback + 1),
                               horizon_range=(max_horizon, max_horizon + 1))

        with pytest.raises(ValueError):
            FlexibleDataLoader(dataset_dummy,
                               indices=None,
                               n_assets_range=(max_assets, max_assets + 1),
                               lookback_range=(0, max_lookback + 1),
                               horizon_range=(max_horizon, max_horizon + 1))

        with pytest.raises(ValueError):
            FlexibleDataLoader(dataset_dummy,
                               indices=None,
                               n_assets_range=(max_assets, max_assets + 1),
                               lookback_range=(max_lookback, max_lookback + 1),
                               horizon_range=(-2, max_horizon + 1))

    def test_basic(self, dataset_dummy):
        max_assets = dataset_dummy.n_assets
        max_lookback = dataset_dummy.lookback
        max_horizon = dataset_dummy.horizon

        dl = FlexibleDataLoader(dataset_dummy,
                                indices=None,
                                n_assets_range=(max_assets, max_assets + 1),
                                lookback_range=(max_lookback, max_lookback + 1),
                                horizon_range=(max_horizon, max_horizon + 1))

        dl = FlexibleDataLoader(dataset_dummy)

        assert isinstance(dl.hparams, dict)

    def test_minimal(self, dataset_dummy):
        dl = FlexibleDataLoader(dataset_dummy, batch_size=2)

        res = next(iter(dl))

        assert len(res) == 4


class TestRidigDataLoader:
    def test_wrong_construction(self, dataset_dummy):
        max_assets = dataset_dummy.n_assets
        max_lookback = dataset_dummy.lookback
        max_horizon = dataset_dummy.horizon

        with pytest.raises(ValueError):
            RigidDataLoader(dataset_dummy,
                            indices=[-1])

        with pytest.raises(ValueError):
            RigidDataLoader(dataset_dummy,
                            asset_ixs=[max_assets + 1, max_assets + 2])

        with pytest.raises(ValueError):
            RigidDataLoader(dataset_dummy,
                            lookback=max_lookback + 1)

        with pytest.raises(ValueError):
            RigidDataLoader(dataset_dummy,
                            horizon=max_horizon + 1)

    def test_basic(self, dataset_dummy):
        dl = RigidDataLoader(dataset_dummy)

        assert isinstance(dl.hparams, dict)
