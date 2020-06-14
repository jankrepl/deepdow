"""
=====================
Softmax and Sparsemax
=====================

:code:`deepdow` offers multiple allocation layers. Among them are the :code:`SoftmaxAllocator` and
:code:`SparsemaxAllocator`. Softmax is a very popular technique that turns vectors of numbers (logits)
into probability distributions. If we do not allow for short selling (no weights below zero) and
leveraging (no weight above 1) then weight allocation can be seen as a probability distribution.
Additionally, sparsemax was proposed by [Martins2016]_ as an alternative to softmax. It enforces
sparsity. Both :code:`SoftmaxAllocator` and :code:`SparsemaxAllocator` support :code:`max_weight`
parameter controlling the maximum possible weight of a single asset and :code:`temperature`.


The below plot shows how these two allocators react to changes in :code:`max_weight` and
:code:`temperature`.

.. warning::

    Note that we are using the :code:`seaborn` to plot a heatmap.

"""

from deepdow.layers import SoftmaxAllocator, SparsemaxAllocator
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

n_assets = 10
seed = 6
temperatures = [0.2, 0.4, 1]
max_weights = [0.2, 0.5, 1]

torch.manual_seed(seed)
logits = torch.rand(size=(1, n_assets)) - 0.5

fig, axs = plt.subplots(len(temperatures),
                        len(max_weights),
                        sharex=True,
                        sharey=True,
                        figsize=(15, 5))
cbar_ax = fig.add_axes([.91, .3, .03, .4])

for r, temperature in enumerate(temperatures):
    for c, max_weight in enumerate(max_weights):
        sparsemax = SparsemaxAllocator(n_assets,
                                       max_weight=max_weight,
                                       temperature=temperature
                                       )

        softmax = SoftmaxAllocator(n_assets=n_assets,
                                   temperature=temperature,
                                   max_weight=max_weight,
                                   formulation='variational')

        w_sparsemax = sparsemax(logits).detach().numpy()
        w_softmax = softmax(logits).detach().numpy()

        df = pd.DataFrame(np.concatenate([w_softmax, w_sparsemax], axis=0),
                          index=['softmax', 'sparsemax'])

        axs[r, c].set_title('temp={}, max_weight={}'.format(temperature, max_weight))
        sns.heatmap(df,
                    vmin=0,
                    vmax=1,
                    center=0.5,
                    cmap='hot',
                    ax=axs[r, c],
                    cbar_ax=cbar_ax,
                    square=True)
