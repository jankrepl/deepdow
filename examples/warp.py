"""
==========
Warp layer
==========

The :ref:`layers_warp` allows for arbitrary warping of the input tensor **x** along the time
(lookback) dimension. One needs to provide element by element transformation with values in [-1, 1].
Note that this transformation can be either seen as a hyperparameter, collection of learnable parameters
(one per training set) or predicted for each sample.

To illustrate how to use this layer, let us assume that we have a single asset. We have observed
its returns over the :code:`lookback=50` previous days. Below we demonstrate 5 different
transformations to the original time series.

- **identity** - no change
- **zoom** - focusing on the last 25 days
- **backwards** - swap the time flow
- **slowdown_start** - slow down the beginning of the time series and speed up the end
- **slowdown_end** - speed up the beginning of the time series and slow down the end
"""

import matplotlib.pyplot as plt
import numpy as np
import torch

from deepdow.data.synthetic import sin_single
from deepdow.layers import Warp

lookback = 50

x_np = (np.linspace(0, 1, num=lookback) * sin_single(lookback, freq=4 / lookback))[None, None, :, None]
x = torch.as_tensor(x_np)

grid = torch.linspace(0, end=1, steps=lookback)[None, :].to(dtype=x.dtype)

transform_dict = {
    'identity': lambda x: 2 * (x - 0.5),
    'zoom': lambda x: x,
    'backwards': lambda x: -2 * (x - 0.5),
    'slowdown\_start': lambda x: 2 * (x ** 3 - 0.5),
    'slowdown\_end': lambda x: 2 * (x ** (1 / 3) - 0.5),
}

n_tforms = len(transform_dict)

_, axs = plt.subplots(n_tforms, 2, figsize=(16, 3 * n_tforms), sharex=True, sharey=True)
layer = Warp()

for i, (tform_name, tform_lambda) in enumerate(transform_dict.items()):
    tform = tform_lambda(grid)
    x_warped = layer(x, tform)

    axs[i, 0].plot(tform.numpy().squeeze(), linewidth=3, color='red')
    axs[i, 1].plot(x_warped.numpy().squeeze(), linewidth=3, color='blue')
    axs[i, 0].set_title(r'$\bf{}$ tform'.format(tform_name))
    axs[i, 1].set_title(r'$\bf{}$ warped'.format(tform_name))
