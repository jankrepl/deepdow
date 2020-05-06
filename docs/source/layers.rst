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

    All the :code:`deepdow` layers assume that the input and output tensors have an extra dimension
    in the front - the **sample** dimension. We omit this dimension on purpose to make the examples
    and sketches simpler.

Transform layers
----------------
**Transform** layers are supposed to extract useful features from input tensors. For the exact usage see
:ref:`layers_transform_API`.

Conv
****


RNN
***


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
    This layer does not include additional (sample) dimension.

References
----------
.. [LectureNotes]
   http://faculty.washington.edu/ezivot/econ424/portfolioTheoryMatrix.pdf

.. [Prado2019]
   Lopez de Prado, M. (2019). A Robust Estimator of the Efficient Frontier. Available at SSRN 3469961.

.. [Agrawal2019]
   Agrawal, Akshay, et al. "Differentiable convex optimization layers." Advances in Neural Information Processing Systems. 2019.

.. [Michaud2007]
   Michaud, Richard O., and Robert Michaud. "Estimation error and portfolio optimization: a resampling solution." Available at SSRN 2658657 (2007).

.. [Ledoit2004]
   Ledoit, Olivier, and Michael Wolf. "Honey, I shrunk the sample covariance matrix." The Journal of Portfolio Management 30.4 (2004): 110-119.

.. [sklearnkmeans]
   https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html

.. [Bodnar2013]
   Bodnar, Taras, Nestor Parolya, and Wolfgang Schmid. "On the equivalence of quadratic optimization problems commonly used in portfolio theory." European Journal of Operational Research 229.3 (2013): 637-644.