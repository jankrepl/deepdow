"""Tests focused on the data module."""
import numpy as np
import pandas as pd
import pytest
import torch

from deepdow.data import InRAMDataset, collate_uniform


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
        batch = [(torch.zeros((1, max_lookback, max_n_assets)), torch.ones((max_horizon, max_n_assets))) for _ in
                 range(n_samples)]

        n_trials = 10

        for _ in range(n_trials):
            X_batch, y_batch = collate_uniform(batch, n_assets_range=(5, 6), lookback_range=(4, 5),
                                               horizon_range=(3, 4))

            assert torch.is_tensor(X_batch)
            assert torch.is_tensor(y_batch)

            assert X_batch.shape == (n_samples, 1, 4, 5)
            assert y_batch.shape == (n_samples, 3, 5)

    def test_replicable(self):
        random_state_a = 3
        random_state_b = 5

        n_samples = 14
        max_n_assets = 10
        max_lookback = 8
        max_horizon = 5
        batch = [(torch.rand((1, max_lookback, max_n_assets)), torch.rand((max_horizon, max_n_assets))) for _ in
                 range(n_samples)]

        X_batch_1, y_batch_1 = collate_uniform(batch, random_state=random_state_a)
        X_batch_2, y_batch_2 = collate_uniform(batch, random_state=random_state_a)
        X_batch_3, y_batch_3 = collate_uniform(batch, random_state=random_state_b)

        assert torch.allclose(X_batch_1, X_batch_2)
        assert torch.allclose(y_batch_1, y_batch_2)

        assert not torch.allclose(X_batch_3, X_batch_1)
        assert not torch.allclose(y_batch_3, y_batch_1)

    def test_different(self):
        n_samples = 6
        max_n_assets = 27
        max_lookback = 15
        max_horizon = 12
        batch = [(torch.zeros((1, max_lookback, max_n_assets)), torch.ones((max_horizon, max_n_assets))) for _ in
                 range(n_samples)]

        n_trials = 10

        n_assets_set = set()
        lookback_set = set()
        horizon_set = set()

        for _ in range(n_trials):
            X_batch, y_batch = collate_uniform(batch,
                                               n_assets_range=(1, max_n_assets),
                                               lookback_range=(1, max_lookback),
                                               horizon_range=(1, max_lookback))

            n_assets_set.add(X_batch.shape[-1])
            lookback_set.add(X_batch.shape[-2])
            horizon_set.add(y_batch.shape[-2])

        assert len(n_assets_set) > 1
        assert len(lookback_set) > 1
        assert len(horizon_set) > 1


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
