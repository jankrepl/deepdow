"""Collection of tests focused on the explain.py module."""
import pytest
import torch

from deepdow.explain import gradient_wrt_input
from deepdow.nn import BachelierNet


def test_basic(dtype_device):
    dtype, device = dtype_device
    n_channels, lookback, n_assets = 2, 3, 4

    target_weights = torch.zeros(n_assets)
    target_weights[1] = 1
    initial_guess = torch.zeros(n_channels, lookback, n_assets)

    network = BachelierNet(n_input_channels=n_channels, n_assets=n_assets, hidden_size=2)

    # WRONG MASK
    with pytest.raises(ValueError):
        gradient_wrt_input(network, target_weights=target_weights, initial_guess=initial_guess, n_iter=3,
                           dtype=dtype, device=device, mask=torch.zeros(n_channels, lookback + 1, n_assets))

    with pytest.raises(TypeError):
        gradient_wrt_input(network, target_weights=target_weights, initial_guess=initial_guess, n_iter=3,
                           dtype=dtype, device=device, mask='wrong_type')

    # NO MASK
    res, hist = gradient_wrt_input(network, target_weights=target_weights, initial_guess=initial_guess, n_iter=3,
                                   dtype=dtype, device=device, verbose=True)

    assert len(hist) == 3
    assert torch.is_tensor(res)
    assert res.shape == initial_guess.shape
    assert res.dtype == dtype
    assert res.device == device
    assert not torch.allclose(initial_guess.to(device=device, dtype=dtype), res)

    # SOME MASK
    some_mask = torch.ones_like(initial_guess, dtype=torch.bool)
    some_mask[0] = False

    res_s, _ = gradient_wrt_input(network, target_weights=target_weights, initial_guess=initial_guess, n_iter=3,
                                  dtype=dtype, device=device, mask=some_mask)

    assert torch.allclose(initial_guess.to(device=device, dtype=dtype)[0], res_s[0])
    assert not torch.allclose(initial_guess.to(device=device, dtype=dtype)[1], res_s[1])

    # EXTREME_MASK
    extreme_mask = torch.zeros_like(initial_guess, dtype=torch.bool)

    res_e, _ = gradient_wrt_input(network, target_weights=target_weights, initial_guess=initial_guess, n_iter=3,
                                  dtype=dtype, device=device, mask=extreme_mask)

    assert torch.allclose(initial_guess.to(device=device, dtype=dtype), res_e)
