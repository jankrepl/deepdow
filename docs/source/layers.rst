Layers
======

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

Collapse layers
---------------

Allocation layers
-----------------





