"""Collection of functions generating synthetic datasets.

All these functions are outputting a 3D ``np.ndarray`` of shape
 `(n_timesteps, n_channels, n_assets)`."""

import numpy as np


def sin_single(n_timesteps, amplitude=1, freq=0.25, phase=0):
    """Generate sine waves.

    Of form :math:`amplitude * /sin(2\pi(x) * freq + phase)`.

    Parameters
    ----------
    n_timesteps : int
        Number of timesteps.

    amplitude : float
        The peak value.

    freq : float
        Frequency - number of oscilations per timestep.

    phase : float
        Offset.

    Returns
    -------
    y : np.ndarray
        1D array of shape `(n_timesteps,)`

    """
    x = np.arange(n_timesteps)

    return amplitude * np.sin(2 * np.pi * freq * x + phase)
