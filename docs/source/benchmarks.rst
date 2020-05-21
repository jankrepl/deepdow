.. _benchmarks:

Benchmarks
==========
The goal of this section is to introduce the concept of a benchmark and demonstrate some specific
examples of it that are implemented within :code:`deepdow`.

In the broad sense, a benchmark is any algorithm that takes in the input feature tensor **x** and
outputs the **weights** tensor. We can divide benchmarks into two categories

- **with** learnable parameters - we call these **networks** and discuss in detail in :ref:`networks`
- **without** learnable parameters - we call these **simple benchmarks** and discuss them in this section


Let us stress one important implication of the above distinction.
The allocation algorithm of simple benchmarks does not need to be differentiable and nothing prevents the
user from casting the input **x** (:code:`torch.Tensor`) to a :code:`numpy` array and using external libraries
(:code:`scipy`, etc).
On the other hand, the allocation algorithm of networks needs to be a forward pass implemented via :code:`torch`
functions, modules or :code:`deepdow.layers` (which is built on top of :code:`torch`).


Benchmark class
---------------
To capture the above general definition we provide an abstract class :code:`deepdow.benchmarks.Benchmark` that
requires its children to implement the :code:`__call__` and optionally
also :code:`hparams` property.


- :code:`__call__` - the weight allocation algorithm
- :code:`hparams` - optional property that is a dictionary of hyperparameters

Simple benchmarks
-----------------
The simple benchmarks are supposed to be allocation schemes that provide a baseline for the trainable networks. By
definition, these simple benchmarks do not change their predictions over different epochs so one can just run them once and
see how they fare against the networks. :code:`deepdow` implements multiple simple benchmarks and we are going to
discuss them in what follows. For usage details see :ref:`benchmarks_API`.


InverseVolatility
*****************
The user needs to specify which channel represents returns via the :code:`returns_channel`
The weight allocation is equal inverse standard deviation of returns if `use_std=True` otherwise it is the inverse
variance.


MaximumReturn
*************
After specifying which channel represents returns via the :code:`returns_channel` a standard maximum return
optimization is performed. One can additionally choose :code:`max_weight` per asset.

MinimumVariance
***************
After specifying which channel represents returns via the :code:`returns_channel` a standard minimum variance
optimization is performed. One can additionally choose :code:`max_weight` per asset.


OneOverN
********
Equally weighted portfolio - each asset has the weight `1/n_assets`.

Random
******
The weights are sampled randomly.

Singleton
*********
Sometimes also called one asset portfolio. The user can chose the single asset via the :code:`asset_ix`.


