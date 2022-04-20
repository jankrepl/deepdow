.. _losses:

Losses
======

.. testsetup::

   import torch

Introduction
------------
The loss functions are one of the main components of :code:`deepdow`. Please review the :ref:`basics_loss` in
:ref:`basics` to understand the setup. Most importantly, by a **loss function** we mean any function that
has the following **two inputs**

- :code:`weights` - :code:`torch.Tensor` of shape :code:`(n_samples, n_assets)`
- :code:`y` - :code:`torch.Tensor` of shape :code:`(n_samples, n_channels, horizon, n_assets)`

And a **single output**

- :code:`loss` - :code:`torch.Tensor` of shape :code:`(n_samples,)`



.. warning::

    Similarly to layers (see :ref:`layers`), all the :code:`deepdow` losses assume that the input and output tensors have
    an extra dimension in the front—the **sample** dimension. It serves for batching when training the networks. For
    this reason, losses must be implemented in a way that all samples are independent.

The above definition of a loss function is very general and in many cases :code:`deepdow` losses focus on a more narrow
family of functions in the background. To be more specific, one can select a single channel (:code:`returns_channel`)
from the :code:`y` tensor representing the desired returns. After this, portfolio returns **r** over each of the
:code:`horizon` steps can be computed which results in a tensor of shape :code:`(n_samples, horizon)`. By applying
some summarization function **S** over the :code:`horizon` dimension we arrive at the final output :code:`loss` of shape
:code:`(n_samples,)`.


Definitions
-----------
Before we start discussing losses themselves let us first write down multiple definitions. Let us assume, that before
investing, we have initial holdings of :math:`V` (in cash). Additionally, for each asset :math:`a` we denote its price
at time :math:`t` as :math:`p^{a}_{t}`. Given some portfolio weights :math:`\textbf{w}` over :math:`N` assets we define
portfolio value at time `t`

.. math::

    p^{\textbf{w}}_t = \sum_{a=1}^{N} p_t^a \frac{w_a V}{p_0^a}

Before we continue, notice that the above definition assumes two things

- We employ the buy and hold strategy
- Assets are perfectly divisible (one can by :math:`\frac{w_a V}{p_0^a}` units of any asset)

Let us now define two types of asset returns: **simple** and **logarithmic**



.. math::

    {}^{\text{S}}r^{a}_{t} = \frac{p^{a}_{t}}{p^{a}_{t-1}} - 1


    {}^{\text{L}}r^{a}_{t} = \log \frac{p^{a}_{t}}{p^{a}_{t-1}}


Additionally, we also consider their portfolio counterparts

.. math::

    {}^{\text{S}}r^{\textbf{w}}_{t} = \frac{p^{\textbf{w}}_{t}}{p^{\textbf{w}}_{t-1}} - 1


    {}^{\text{L}}r^{\textbf{w}}_{t} = \log \frac{p^{\textbf{w}}_{t}}{p^{\textbf{w}}_{t-1}}


Note that in both of the cases the initial holding :math:`V` cancels out and the portfolio returns are independent
of it.


Portfolio returns
-----------------
One can extract portfolio returns given asset returns via the function
:code:`portfolio_returns`. It inputs a matrix of asset returns (the returns type is controlled via :code:`input_type`)



.. math::

   \begin{bmatrix}
   r^{1}_1 & \dots  & r^{N}_1 \\
   \vdots &  \ddots  &  \vdots \\
   r^{1}_{\text{horizon}} & \dots & r^{N}_{\text{horizon}}
   \end{bmatrix}


and outputs a vector of portfolio returns (the type is controlled via :code:`output_type`)

.. math::

    \textbf{r}^{\textbf{w}} = \begin{bmatrix}
    r^{\textbf{w}}_{1} \\
    \vdots \\
    r^{\textbf{w}}_{\text{horizon}}
    \end{bmatrix}

We rely on the below relation to perform the computations

.. math::

    {}^{\text{S}}r_t^{\textbf{w}}=\frac{\sum_{a=1}^{N}{}^{\text{S}}r_{t}^{a}w_a\prod_{i=1}^{t-1}(1+{}^{\text{S}}r_{i}^{a})}{\sum_{a=1}^{N}w_a\prod_{i=1}^{t-1}(1+{}^{\text{S}}r_{i}^{a})}

.. math::

.. testcode::

    from deepdow.losses import portfolio_returns

    returns = torch.tensor([[[0.1, 0.2], [0.05, 0.02]]])  # (n_samples=1, horizon=2, n_asset=2)
    weights = torch.tensor([[0.4, 0.6]])  # (n_samples=1, n_samples=2)

    prets = portfolio_returns(weights, returns, input_type='simple', output_type='simple')

    assert prets.shape == (1, 2)  # (n_samples, horizon)
    assert torch.allclose(prets, torch.tensor([[0.1600, 0.0314]]), atol=1e-4)


Available losses
----------------
To avoid confusion, all the available losses have the *"The lower the better"* logic. If the class name suggests
otherwise (i.e. :code:`MeanReturns`) a negative is computed instead. For the exact usage see :ref:`losses_API`.


Alpha
*****
Negative alpha with respect to a predefined portfolio of assets. If :code:`benchmark_weights=None` then
considering the equally weighted portfolio by default.


CumulativeReturn
****************
Negative simple cumulative of the buy and hold portfolio at the end of the :code:`horizon` steps.

.. math::

     \frac{p^{\textbf{w}}_{t + \text{horizon}}}{p^{\textbf{w}}_{t}} - 1


LargestWeight
*************
Loss function independent of :code:`y`, only taking into account the :code:`weights`.

.. math::

    max(\textbf{w})

MaximumDrawdown
***************
The **negative** of the maximum drawdown.


MeanReturns
***********
The **negative** of mean portfolio returns over the :code:`horizon` time steps.


.. math::

    {\mu}^{\textbf{w}} = \frac{\sum_{i}^{\text{horizon}} r^{\textbf{w}}_{i} }{\text{horizon}}

RiskParity
**********

.. math::

   \sum_{i=1}^{N}\Big(\frac{\sigma}{N} - w_i \big(\frac{\Sigma\textbf{w}}{\sigma}\big)_i\Big) ^ 2

where :math:`\sigma=\sqrt{\textbf{w}^T\Sigma\textbf{w}}` and :math:`\Sigma` is
the covariance matrix of asset returns.

Quantile (Value at Risk)
************************
The **negative** of the :code:`p`-quantile of portfolio returns. Note that in the background it solved via
:code:`torch.kthvalue`.

SharpeRatio
***********
The **negative** of the Sharpe ratio of portfolio returns.

.. math::

    \frac{{\mu}^{\textbf{w}} - r_{\text{rf}}}{{\sigma}^{\textbf{w}} + \epsilon}

SortinoRatio
************
The **negative** of the Sortino ratio of portfolio returns.

.. math::

    \frac{{\mu}^{\textbf{w}} - r_{\text{rf}}}{\sqrt{\frac{\sum_{i}^{\text{horizon}} \max({\mu}^{\textbf{w}} - r^{\textbf{w}}_{i} , 0)^{2}}{\text{horizon}}} + \epsilon}


SquaredWeights
**************
Loss function independent of :code:`y`, only taking into account the :code:`weights`.

.. math::

    \sum_{i=1}^{N} w_i^2


The lower this loss is, the more diversified our portfolio is. If we focus on two extremes,
for the equally weighted it is :math:`\frac{1}{N}`. For a single asset portfolio it is :math:`1`.

StandardDeviation
*****************

.. math::

    {\sigma}^{\textbf{w}} = \sqrt{\frac{\sum_{i}^{\text{horizon}} (r^{\textbf{w}}_{i} - {\mu}^{\textbf{w}})^{2}}{\text{horizon}}}

Downside Risk
*************

.. math::

    \sqrt{\frac{\sum_{i}^{\text{horizon}} \max({\mu}^{\textbf{w}} - r^{\textbf{w}}_{i} , 0)^{\beta}}{\text{horizon}}}


WorstReturn
***********
The **negative** of the minimum returns

.. math::

    min(\textbf{r}^{\textbf{w}})




Arithmetic operations
----------------------
:code:`deepdow` offers a powerful feature of performing arithmetic operations between loss instances. In other words,
one can obtain new losses by performing **unary** and **binary** operations on existing losses.

Lets assume we have a loss instance, then the available operations are

**Unary**

- addition of a constant
- multiplication by a constant
- division by a constant
- exponentiation

**Binary**

- addition of another loss
- multiplication by another loss
- division by another loss

.. warning::

    Currently, the :code:`__repr__` of a loss that is a result of an arithmetic operation is just a naive
    string concatenation of :code:`__repr__` of the constituent losses. No symbolic mathematics and expression reduction
    is utilized.


