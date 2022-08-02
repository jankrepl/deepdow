"""Collection of tests focused on the `data/synthetic.py` module."""

import numpy as np
import pytest

from deepdow.data.synthetic import sin_single


class TestSin:
    @pytest.mark.parametrize("n_timesteps", [50, 120])
    @pytest.mark.parametrize("period_length", [2, 5, 9])
    @pytest.mark.parametrize("amplitude", [0.1, 10])
    def test_basic(self, n_timesteps, period_length, amplitude):
        freq = 1 / period_length
        res = sin_single(
            n_timesteps, freq=freq, phase=0.4, amplitude=amplitude
        )

        assert isinstance(res, np.ndarray)
        assert res.shape == (n_timesteps,)
        assert len(np.unique(np.round(res, 5))) == period_length
        assert np.all(abs(res) <= amplitude)
