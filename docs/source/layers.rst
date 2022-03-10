.. _layers:


Layers
======

.. testsetup::

   import torch


Introduction
------------
As described in :ref:`basics` our goal is to construct a network, that inputs a 3D tensor **x** of shape
:code:`(n_channels, lookback, n_assets)` and outputs a 1D tensor **w** of shape :code:`(n_assets,)`.  One can achieve
this task by creating a **pipeline of layers**. See below an example of a pipeline

.. image:: https://i.imgur.com/RPhxF4j.png
   :align: center
   :width: 650


- **L1** - 1D convolution shared across assets, no change in dimensionality
- **L2** - mean over the channels (3D -> 2D)
- **L3** - maximum over timesteps (2D -> 1D)
- **L4** - covariance matrix of columns of **h2**
- **L5** - given **h3** and **h4** solves convex optimization problem

:code:`deepdow` groups all custom layers into 4 categories:

- Transform
    Feature extractors that do not change the dimensionality of the input tensor. **L1** in the example.
- Collapse
    Remove an entire dimension of the input via some aggregation scheme.  **L2** and **L3** in the example.
- Allocate
    Given input tensors these layers generate final portfolio weights. **L5** in the example.
- **Misc**
    Helper layers. **L4** in the example.


Note that all custom layers are simply subclasses of :code:`torch.nn.Module` and one can freely use them together
with official PyTorch layers.


.. warning::

    Almost all :code:`deepdow` layers assume that the input and output tensors have an extra dimension
    in the front—the **sample** dimension. We often omit this dimension on purpose to make the examples
    and sketches simpler.

Transform layers
----------------
**Transform** layers are supposed to extract useful features from input tensors. For the exact usage see
:ref:`layers_transform_API`.

Conv
****
This layer supports both :code:`1D` and :code:`2D` convolution controlled via the :code:`method` parameter.
In the forward pass we need to provide tensors of shape :code:`(n_samples, n_input_channels, lookback)` resp.
:code:`(n_samples, n_input_channels, lookback, n_assets)`. The padding is automatically implied by :code:`kernel_size`
such that the output tensor has the **same** size (for odd :code:`kernel_size` exactly, for even approximately).


.. testcode::

    from deepdow.layers import Conv

    n_samples, n_input_channels, lookback, n_assets = 2, 4, 20, 11
    n_output_channels = 8
    x = torch.rand(n_samples, n_input_channels, lookback, n_assets)

    layer = Conv(n_input_channels=n_input_channels,
                 n_output_channels=n_output_channels,
                 kernel_size=3,
                 method='1D')

    # Apply the same Conv1D layer to all assets
    result = torch.stack([layer(x[..., i]) for i in range(n_assets)], dim=-1)

    assert result.shape == (n_samples, n_output_channels, lookback, n_assets)

RNN
***
This layer runs the same recurrent network over all assets and then stacks the hidden layers back together.
It provides both the standard :code:`RNN` as well as :code:`LSTM`. The choice is controlled
via the parameter :code:`cell_type`. The user specifies the number of output channels via :code:`hidden_size`. This
number corresponds to the actual hidden state dimensionality if :code:`bidirectional=False` otherwise it is one half of
it.

.. testcode::

    from deepdow.layers import RNN

    n_samples, n_input_channels, lookback, n_assets = 2, 4, 20, 11
    hidden_size = 8
    x = torch.rand(n_samples, n_input_channels, lookback, n_assets)

    layer = RNN(n_channels=n_input_channels,
                 hidden_size=hidden_size,
                 cell_type='LSTM')

    result = layer(x)

    assert result.shape == (n_samples, n_output_channels, lookback, n_assets)

.. _layers_warp:

Warp
****
This layer is inspired by the problem of time series alignment (see [Weber2019]_).
It allows the user to specify per asset 1D transformations to warp the input tensor **x** with.
Note that :ref:`layers_zoom` is a special case. The :code:`tform` tensor should mostly have values
between (-1, 1) where -1 represents the beginning of the time series and 1 represents the end
(the most recent observations). This layer has two modes based on the shape of provided
:code:`tform`.

- :code:`tform.shape = (n_samples, lookback, n_assets)` - Warping each asset differently
- :code:`tform.shape = (n_samples, lookback)` - Warping each asset the same way


.. testcode::

    from deepdow.layers import Warp

    n_samples, n_channels, lookback, n_assets = 2, 4, 20, 11
    x = torch.rand(n_samples, n_channels, lookback, n_assets)
    single_tform = (torch.linspace(0, end=1, steps=lookback) ** 2 - 0.5) * 2
    tform = torch.stack(n_samples * [single_tform], dim=0)

    layer = Warp()

    result = layer(x, tform)

    assert result.shape == (n_samples, n_channels, lookback, n_assets)

Note that to prevent folding one should provide strictly monotonic transformations.

.. seealso::

    Example :ref:`sphx_glr_auto_examples_layers_warp.py`

.. _layers_zoom:

Zoom
****
Inspired by the Spatial Transformer Network [Jaderberg2015]_, this layer allows to dynamically zoom in
and out along the :code:`lookback` (time) dimension of the input **x**. In other words,
it performs dynamic time warping (with linear transformation). By providing
a scale of 1 no changes are made. If provides scale < 1 i.e. 0.5 then the time is slowed down twice
and :code:`lookback/2` most recent timesteps are considered. Conversely, if we provide scale > 1
i.e. 2 then the time is sped up twice and :code:`2 * lookback` timesteps are considered. Since
we only have :code:`lookback` timesteps available in **x** we employ padding (see below).

The :code:`method` parameter determines what interpolation is used (either :code:`'bilinear'` and
:code:`'nearest'`). The parameter :code:`padding_method` controls what to do with values that
fall outside of the grid (happens when scale > 1). The options are :code:`'zeros'`, :code:`'border'`
and :code:`'reflection'`.


.. testcode::

    from deepdow.layers import Zoom

    n_samples, n_channels, lookback, n_assets = 2, 4, 20, 11
    x = torch.rand(n_samples, n_channels, lookback, n_assets)
    scale = torch.rand(n_samples)  # values between (0, 1) representing slowing down

    layer = Zoom()

    result = layer(x, scale)

    assert result.shape == (n_samples, n_channels, lookback, n_assets)

.. seealso::

    Example :ref:`sphx_glr_auto_examples_layers_zoom.py`



Collapse layers
---------------
**Transform** layers remove entire dimension. For the exact usage see
:ref:`layers_collapse_API`.

AttentionCollapse
*****************

AverageCollapse
***************


ElementCollapse
***************

ExponentialCollapse
*******************


MaxCollapse
***********

SumCollapse
***********



Allocation layers
-----------------
For the exact usage see :ref:`layers_allocate_API`.

AnalyticalMarkowitz
*******************
The :code:`AnalyticalMarkowitz` layer has two modes. If the user provides only the covariance matrix
:math:`\boldsymbol{\Sigma}`, it returns the **Minimum variance portfolio**. However, if additionally one supplies the
expected return vector :math:`\boldsymbol{\mu}` then it computes the **Tangency portfolio** (also known as the
**Maximum Sharpe ratio portfolio**). Note that risk free rate is assumed to be zero.


.. math::

    \textbf{w}_{\text{minvar}} =  \frac{\boldsymbol{\Sigma}^{-1} \textbf{1}}{\textbf{1}^{T} \boldsymbol{\Sigma}^{-1} \textbf{1}}

    \textbf{w}_{\text{maxsharpe}} =  \frac{\boldsymbol{\Sigma}^{-1} \boldsymbol{\mu}}{\textbf{1}^{T} \boldsymbol{\Sigma}^{-1} \boldsymbol{\mu}}


Note that this allocator cannot enforce any additional constraints i.e. maximum weight per asset. For more details and
derivations see [LectureNotes]_.

NCO
***
The :code:`NCO` allocator is heavily inspired by **Nested Cluster Optimization** proposed in [Prado2019]_. The main
idea is to group assets into :code:`n_clusters` different clusters and use :code:`AnalyticalMarkowitz` inside each of
them. In the second step, we compute asset allocation across these :code:`n_clusters` new portfolios. Note that
the clustering is currently done via the :code:`KMeans` layer (see :ref:`kmeans`).

.. _layers_numericalmarkowitz:

NumericalMarkowitz
******************
While :code:`AnalyticalMarkowitz` gives us the benefit of analytical solutions, it does not allow for any additional
constraints. :code:`NumericalMarkowitz` is a generic convex optimization solver built on top of :code:`cvxpylayers`
(see [Agrawal2019]_ for more details). The statement of the problem is shown below. It is motivated by [Bodnar2013]_.

.. math::

    \begin{aligned}
    \max_{\textbf{w}} \quad & \textbf{w}^{T}\boldsymbol{\mu} - \gamma {\textbf{w}}^{T}  \boldsymbol{\Sigma} \textbf{w} - \alpha \textbf{w}^{T} \textbf{w} \\
    \textrm{s.t.} \quad & \sum_{i=1}^{N}w_i = 1 \\
    \quad & w_i >= 0, i \in \{1,...,N\}\\
    \quad & w_i <= w_{\text{max}}, i \in \{1,...,N\}\\
    \end{aligned}


The user needs to provide :code:`n_assets` (:math:`N` in the above formulation) and :code:`max_weight`
(:math:`w_{\text{max}}`) when constructing this layer. To perform a forward pass one passes the following
tensors (batched along the sample dimension):

- :code:`rets` - Corresponds to the expected returns vector :math:`\boldsymbol{\mu}`
- :code:`covmat_sqrt` - Corresponds to a (matrix) square root of the covariance matrix :math:`\boldsymbol{\Sigma}`
- :code:`gamma_sqrt` - Corresponds to a square root of :math:`\gamma` and controls risk aversion
- :code:`alpha` - Corresponds to :math:`\alpha` and determines the regularization power. Internally, its absolute value is used to prevent sign changes.



.. warning::

    The major downside of using this allocator is a significant decrease in speed.

NumericalRiskBudgeting
**********************
Proposed in [Spinu2013]_.

.. math::

    \begin{aligned}
    \min_{\textbf{w}} \quad & \frac{1}{2}{\textbf{w}}^{T}  \boldsymbol{\Sigma} \textbf{w} - \sum_{i=1}^{N} b_i  \log(w_i) \\
    \textrm{s.t.} \quad & \sum_{i=1}^{N}w_i = 1 \\
    \quad & w_i >= 0, i \in \{1,...,N\}\\
    \quad & w_i <= w_{\text{max}}, i \in \{1,...,N\}\\
    \end{aligned}

where the :math:`b_i, i=1,..,N` are the risk budgets. The user needs to provide
:code:`n_assets` (:math:`N` in the above formulation) and :code:`max_weight`
(:math:`w_{\text{max}}`) when constructing this layer. To perform a forward pass one passes the following
tensors (batched along the sample dimension):

- :code:`covmat_sqrt` - Corresponds to a (matrix) square root of the covariance matrix :math:`\boldsymbol{\Sigma}`
- :code:`b` - Risk budgets


.. warning::

    The major downside of using this allocator is a significant decrease in speed.

Resample
********
The :code:`Resample` layer is inspired by [Michaud2007]_. It is a **metallocator** that expects an instance
**base** allocator as an input. Currently supported base allocators are:

- :code:`AnalyticalMarkowitz`
- :code:`NCO`
- :code:`NumericalMarkowitz`

The premise of this metaallocator is that :math:`\boldsymbol{\mu}` and :math:`\boldsymbol{\Sigma}` are just noisy
estimates of their population counterparts. Parametric boostrapping is therefore applied. We sample
:code:`n_portfolios * n_draws` new vectors from the distribution
:math:`\mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})`. We then create estimates
:math:`\boldsymbol{\mu}_{1}, ...,\boldsymbol{\mu}_{\text{n_portfolios}}` and
:math:`\boldsymbol{\Sigma}_{1}, ..., \boldsymbol{\Sigma}_{\text{n_portfolios}}` and run the base allocator for each of
the pairs. This results in obtaining multiple allocations :math:`\textbf{w}_{1}, ...,\textbf{w}_{\text{n_portfolios}}`.
The final allocation is simply an average :math:`\textbf{w} = \sum_{i=1}^{\text{n_portfolios}}\textbf{w}_i`.


SoftmaxAllocator
****************
Inspired by portfolio optimization with reinforcement learning (i.e. [Jiang2017]_) the :code:`SoftmaxAllocator`
performs a softmax over the input. Additionally, one can also provide custom :code:`temperature`.

.. math::

    w_j = \frac{e^{\frac{z_{j}}{\text{temperature}}}}{\sum_{i} e^{\frac{z_i}{\text{temperature}}}}


Note that one can provide a single :code:`temperature` at construction that is shared across all samples. Alternatively,
one can provide per sample temperature when performing the forward pass.

The above formulation (:code:`formulation`) is **analytical**. One can also obtain the same weights
via solving a convex optimization problem (**variational** formulation). See [Agrawal2019]_  and
[Martins2017]_ for more details.

.. math::

    \begin{aligned}
    \min_{\textbf{w}} \quad & - \textbf{x}^T \textbf{w} - H(\textbf{w}) \\
    \textrm{s.t.} \quad & \sum_{i=1}^{N}w_i = 1 \\
    \quad & w_i >= 0, i \in \{1,...,N\}\\
    \quad & w_i <= w_{\text{max}}, i \in \{1,...,N\}\\
    \end{aligned}

where :math:`H(\textbf{w})=-\sum_{i=1}^{N} w_i \log(w_i)` is the entropy. Note that if
:code:`max_weight` is set to 1 then one gets the unconstrained (analytical) softmax. The benefit of
using the variational formulation is the fact that the user can decide on any :code:`max_weight`
from :code:`(0, 1]`.

.. testcode::

   from deepdow.layers import SoftmaxAllocator

   layer = SoftmaxAllocator(temperature=None)
   x = torch.tensor([[1, 2.3], [2, 4.2]])
   temperature = torch.tensor([0.2, 1])

   w = layer(x, temperature=temperature)

   assert w.shape == (2, 2)
   assert torch.allclose(w.sum(1), torch.ones(2))

.. seealso::

    Example :ref:`sphx_glr_auto_examples_layers_softmax_sparsemax.py`

SparsemaxAllocator
******************
Suggested in [Martins2016]_. It is similar to Softmax but enforces sparsity. It currently uses
:code:`cvxpylayers` as a backend. See below a mathematical formulation. note that **x** represents
the logits.

.. math::

    \begin{aligned}
    \min_{\textbf{w}} \quad & {\vert \vert \textbf{w} - \textbf{x} \vert \vert}^2_{2} \\
    \textrm{s.t.} \quad & \sum_{i=1}^{N}w_i = 1 \\
    \quad & w_i >= 0, i \in \{1,...,N\}\\
    \quad & w_i <= w_{\text{max}}, i \in \{1,...,N\}\\
    \end{aligned}

Similarly to :code:`SoftmaxAllocator` one can provide temperature either per sample or a single
one at construction. Additionally, one can control the maximum weight via the :code:`max_weight`
parameter.

.. testcode::

   from deepdow.layers import SparsemaxAllocator

   n_assets = 3
   layer = SparsemaxAllocator(n_assets, temperature=1)
   x = torch.tensor([[1, 2.3, 2.1], [2, 4.2, -1.1]])

   w = layer(x)
   w_true = torch.tensor([[-4.8035e-06, 6.0000e-01, 4.0000e-01],
        			[8.9401e-05, 1.0001e+00, -1.7873e-04]])
   assert w.shape == (2, 3)
   assert torch.allclose(w.sum(1), torch.ones(2))
   assert torch.allclose(w, w_true, atol=1e-4)

.. seealso::

    Example :ref:`sphx_glr_auto_examples_layers_softmax_sparsemax.py`

.. _weight_norm:

WeightNorm
**********
This allocation layer is supposed to be the simplest layer that could be used as a benchmark.
The goal is to fix the number of assets :code:`n_assets` and for each of them learn a non-negative
value :math:`w\_` that represents the unnormalized weight. The final allocation is then simply
computed as

.. math::

    \textbf{w} = \frac{\textbf{w}\_}{\sum_{i=1}^{\text{n_assets}}w\_}

.. testcode::

   from deepdow.layers import WeightNorm

   n_assets = 5
   layer = WeightNorm(n_assets)
   x = torch.tensor([[1, 2.3, 2.1], [2, 4.2, -1.1]])

   w = layer(x)

   assert torch.allclose(w.sum(1), torch.ones(2))
   assert torch.allclose(w[0], w[1])


Misc layers
-----------
For the exact usage see :ref:`layers_misc_API`.

Cov2Corr
********
Conversion of a covariance matrix into a correlation matrix.

.. testcode::

   from deepdow.layers import Cov2Corr

   layer = Cov2Corr()
   covmat = torch.tensor([[[4, 3], [3, 9.0]]])
   corrmat = layer(covmat)

   assert torch.allclose(corrmat, torch.tensor([[[1.0, 0.5], [0.5, 1.0]]]))


CovarianceMatrix
****************
Computes a sample covariance matrix. One can also apply shrinkage, i.e.

.. math::

    \boldsymbol{\Sigma}_{\text{shrink}} = (1 - \delta) F + \delta S

The :math:`F` is a highly structured matrix whereas :math:`S` is the sample covariance matrix.
The constant :math:`\delta` (:code:`shrinkage_coef` in the constructor) determines how
we weigh the two matrices. See [Ledoit2004]_ for additional background. :code:`deepdow` offers
multiple preset matrices :math:`F` that can be controlled via the :code:`shrinkage_strategy` parameter.

- :code:`None` - no shrinkage applied (can lead to non-PSD matrix)
- :code:`diagonal` - diagonal of :math:`S` with off-diagonal elements being zero
- :code:`identity` - identity matrix
- :code:`scaled-identity` - diagonal filled with average variance in :math:`S` and off-diagonal elements set to zero

After performing shrinkage, one can also compute the (matrix) square root of the shrinked matrix. This is controlled
by the boolean :code:`sqrt`.


.. note::

    One can also omit the :code:`shrinkage_coef` in the constructor (:code:`shrinkage_coef=None`) and
    pass it dynamically as a ``torch.Tensor`` during a forward pass.


.. testcode::

   from deepdow.layers import CovarianceMatrix

   torch.manual_seed(3)

   x = torch.rand(1, 10, 3) * 100
   layer = CovarianceMatrix(sqrt=False)
   layer_sqrt = CovarianceMatrix(sqrt=True)

   covmat = layer(x)
   covmat_sqrt = layer_sqrt(x)

   assert torch.allclose(covmat[0], covmat_sqrt[0] @ covmat_sqrt[0], atol=1e-2)

.. _kmeans:

KMeans
******
A version of the well-known clustering algorithm. The :code:`deepdow` interface is very similar to the one of
scikit-learn [sklearnkmeans]_. Most importantly, one needs to decide on the :code:`n_clusters`.


.. testcode::

   from deepdow.layers import KMeans

   x = torch.tensor([[0, 0], [0.5, 0], [0.5, 1], [1, 1.0]])
   manual_init = torch.tensor([[0, 0], [1, 1]])

   kmeans_layer = KMeans(n_clusters=2, init='manual')
   cluster_ixs, cluster_centers = kmeans_layer(x, manual_init=manual_init)

   assert torch.allclose(cluster_ixs, torch.tensor([0, 0, 1, 1]))

.. warning::

    This layer does not include additional (sample) dimension. Batching can be implemented by a naive for loop
    and stacking.

References
----------
.. [LectureNotes]
   http://faculty.washington.edu/ezivot/econ424/portfolioTheoryMatrix.pdf

.. [Prado2019]
   Lopez de Prado, M. (2019). A Robust Estimator of the Efficient Frontier. Available at SSRN 3469961.

.. [Spinu2013]
   Spinu, Florin, An Algorithm for Computing Risk Parity Weights (July 30, 2013). Available at SSRN: https://ssrn.com/abstract=2297383 or http://dx.doi.org/10.2139/ssrn.2297383

.. [Jiang2017]
   Jiang, Zhengyao, and Jinjun Liang. "Cryptocurrency portfolio management with deep reinforcement learning." 2017 Intelligent Systems Conference (IntelliSys). IEEE, 2017

.. [Weber2019]
   Weber, Ron A. Shapira, et al. "Diffeomorphic Temporal Alignment Nets." Advances in Neural Information Processing Systems. 2019.

.. [Agrawal2019]
   Agrawal, Akshay, et al. "Differentiable convex optimization layers." Advances in Neural Information Processing Systems. 2019.

.. [Michaud2007]
   Michaud, Richard O., and Robert Michaud. "Estimation error and portfolio optimization: a resampling solution." Available at SSRN 2658657 (2007).

.. [Martins2016]
   Martins, Andre, and Ramon Astudillo. "From softmax to sparsemax: A sparse model of attention and multi-label classification." International Conference on Machine Learning. 2016.

.. [Ledoit2004]
   Ledoit, Olivier, and Michael Wolf. "Honey, I shrunk the sample covariance matrix." The Journal of Portfolio Management 30.4 (2004): 110-119.

.. [sklearnkmeans]
   https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html

.. [Martins2017]
   Martins, André FT, and Julia Kreutzer. "Learning what’s easy: Fully differentiable neural easy-first taggers." Proceedings of the 2017 conference on empirical methods in natural language processing. 2017.

.. [Bodnar2013]
   Bodnar, Taras, Nestor Parolya, and Wolfgang Schmid. "On the equivalence of quadratic optimization problems commonly used in portfolio theory." European Journal of Operational Research 229.3 (2013): 637-644.

.. [Jaderberg2015]
   Jaderberg, Max, Karen Simonyan, and Andrew Zisserman. "Spatial transformer networks." Advances in neural information processing systems. 2015.
