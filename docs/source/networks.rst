.. _networks:

Networks
========

.. testsetup::

   import torch

   torch.manual_seed(2)



The main goal of :code:`deepdow` is to provide easy access to building end-to-end differentiable portfolio allocation
neural networks. This section proposes multiple different networks and shows how to create new ones. To better understand
this section, we encourage the user to read previous sections. In particular, the section :ref:`basics` discusses
the overall pipeline. Additionally, in :ref:`layers` we describe in detail the
building blocks of the networks.

**Portfolio allocation neural network** :math:`F` is function that inputs a raw feature tensor **x** of shape
:code:`(n_channels, lookback, n_assets)`. The output is the allocation vector **w** of shape :code:`(n_assets,)` such
that :math:`\sum_{i} w_{i} = 1`. The last requirement is that this function is parametrized by a vector :math:`\theta`.

.. image:: https://i.imgur.com/sJ30WFE.png
   :align: center
   :width: 500

Ideally, this network should propose the best portfolio **w** to be held for **horizon** number of time steps given
what happened in the market up until now **x**. The actually meaning of `best` depends on with what loss function the
network was trained.

In this section, we propose multiple architectures. They are by no means optimal and should serve
as an example.


Existing networks
-----------------
:code:`deepdow` offers multiple networks which attempt to demonstrate the versatility of the framework. Note that these
networks are by no means ideal and the users are encouraged to write their custom networks that are better suited for
their use case. See :ref:`writing_custom_networks` for more details. For more details on the exact usage see the
:ref:`networks_API`.


.. warning::

    All :code:`deepdow` networks assume that the input and output tensors have an extra dimension
    in the front—the **sample** dimension. We omit this dimension on purpose to make the examples
    and sketches simpler.


BachelierNet
************
This network relies on RNN to extract features. To find the allocation it uses a convex optimizer.
To determine the input of this convex optimizer (:code:`NumericalMarkowitz`) we make :code:`alpha` and
:code:`gamma` learnable but independent of the sample. On the other hand, :code:`rets` and :code:`covmat`
are going to be some functions of the input sample **x**.

This network has non-trivial branching

**Covariance matrix**

1. input **x** :code:`(n_channels, lookback, n_assets)`
2. normalized :code:`(n_channels, lookback, n_assets)`
3. first channel :code:`(lookback, n_assets)`
4. computed covariance matrix over assets :code:`(n_assets, n_assets)`

**Expected returns**

1. input **x** :code:`(n_channels, lookback, n_assets)`
2. normalized :code:`(n_channels, lookback, n_assets)`
3. 1D RNN hidden states :code:`(hidden_size, lookback, n_assets)`
4. dropped out :code:`(hidden_size, lookback, n_assets)`
5. attention over lookback :code:`(hidden_size, n_assets)`
6. average over channels :code:`(n_assets,)`



.. testcode::

    from deepdow.nn import BachelierNet

    n_input_channels = 2
    n_assets = 10
    max_weight = 0.5
    hidden_size = 32
    network = BachelierNet(n_input_channels, n_assets, hidden_size=hidden_size, max_weight=max_weight)

    print(network)

.. testoutput::

    BachelierNet(
      (norm_layer): InstanceNorm2d(2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
      (transform_layer): RNN(
        (cell): LSTM(2, 16, bidirectional=True)
      )
      (dropout_layer): Dropout(p=0.5, inplace=False)
      (time_collapse_layer): AttentionCollapse(
        (affine): Linear(in_features=32, out_features=32, bias=True)
        (context_vector): Linear(in_features=32, out_features=1, bias=False)
      )
      (covariance_layer): CovarianceMatrix()
      (channel_collapse_layer): AverageCollapse()
      (portfolio_opt_layer): NumericalMarkowitz(
        (cvxpylayer): CvxpyLayer()
      )
    )



KeynesNet
*********
This network connects 1D convolutions (or RNN) with softmax allocation. Note that this network learns the
:code:`temperature` parameter to be used inside the :code:`SoftmaxAllocator`.

The activations have the following shape (omitting the sample dimension).

1. input **x** :code:`(n_channels, lookback, n_assets)`
2. instance normalized :code:`(n_channels, lookback, n_assets)`
3. extracted features (RNN or 1D Conv) :code:`(hidden_size, lookback, n_assets)`
4. group normalized :code:`(hidden_size, lookback, n_assets)`
5. relu :code:`(hidden_size, lookback, n_assets)`
6. average over lookback :code:`(hidden_size, n_assets)`
7. average over channels :code:`(n_assets,)`
8. softmax allocation :code:`(n_assets,)`

.. testcode::

    from deepdow.nn import KeynesNet

    n_input_channels = 2
    hidden_size = 32
    n_groups = 4
    transform_type = 'Conv'

    network = KeynesNet(n_input_channels,
                        hidden_size=hidden_size,
                        transform_type=transform_type,
                        n_groups=n_groups)

    print(network)

.. testoutput::

    KeynesNet(
      (transform_layer): Conv(
        (conv): Conv1d(2, 32, kernel_size=(3,), stride=(1,), padding=(1,))
      )
      (norm_layer_1): InstanceNorm2d(2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
      (norm_layer_2): GroupNorm(4, 32, eps=1e-05, affine=True)
      (time_collapse_layer): AverageCollapse()
      (channel_collapse_layer): AverageCollapse()
      (portfolio_opt_layer): SoftmaxAllocator(
        (layer): Softmax(dim=1)
      )
    )


LinearNet
*********
This network is very particular, since it uses no structural information contained in the input **x**. In other words,
if we randomly shuffle all our inputs along any dimension and retrain this network, it will yield the same predictions.

Note that his network learns the :code:`temperature` parameter to be used inside the :code:`SoftmaxAllocator`.

The activations have the following shape (omitting the sample dimension).

1. input **x** :code:`(n_channels, lookback, n_assets)`
2. flattened :code:`(n_channels * lookback * n_assets,)`
3. normalized :code:`(n_channels, lookback, n_assets)`
4. dropped out :code:`(n_channels, lookback, n_assets)`
5. after dense layer (multivariate linear model) :code:`(n_assets,)`
6. after allocation :code:`(n_assets,)`

.. testcode::

    from deepdow.nn import LinearNet

    n_channels, lookback, n_assets = 2, 30, 10
    network = LinearNet(n_channels, lookback, n_assets)

    print(network)

.. testoutput::

    LinearNet(
      (norm_layer): BatchNorm1d(600, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (dropout_layer): Dropout(p=0.5, inplace=False)
      (linear): Linear(in_features=600, out_features=10, bias=True)
      (allocate_layer): SoftmaxAllocator(
        (layer): Softmax(dim=1)
      )
    )


MinimalNet
**********
:code:`MinimalNet` is the simplest network. It does not pay any attention to input features
and only learns a fixed weight vector that is predicted for all samples. It is a wrapper
around the :ref:`weight_norm` layer.

The activations have the following shape (omitting the sample dimension).

1. input **x** :code:`(n_channels, lookback, n_assets)`
2. output **w** :code:`(n_assets,)`


.. note::

    The reason why we still need to feed the feature tensor **x** during the forward is to extract
    the required number of samples (:code:`x.shape[0]`).

.. testcode::

    from deepdow.nn import MinimalNet

    n_assets = 10
    network = MinimalNet(n_assets)

    print(network)

    assert sum(p.numel() for p in network.parameters() if p.requires_grad) == n_assets

.. testoutput::

    MinimalNet(
      (allocate_layer): WeightNorm()
    )



ThorpeNet
*********
The goal of this network is to demonstrate the possibility of using :code:`deepdow` to create a special case of
networks that do not depend on the input tensor **x**. All the important variables for the portfolio allocation are
learned when training. This means that this network learns a single optimal set of parameters for the entire
training set.

Specifically, we use the :code:`NumericalMarkowitz` allocator (see :ref:`layers` for more details). We need to learn
the following parameters

- :code:`matrix` - square root of the covariance matrix, initial value is identity matrix
- :code:`exp_returns` - expected returns, initial value is 1
- :code:`gamma_sqrt` - risk and return trade-off, initial value is 1
- :code:`alpha` - weight regularization, initial value is 1

Note that to avoid numerical issues, one can set :code:`force_symmetric=True` at construction. This way, the
:code:`matrix` is multiplied by its transpose to guarantee that the input to the allocator is symmetric and
semi-definite.



.. testcode::

    from deepdow.nn import ThorpNet

    n_assets = 10
    max_weight = 0.5
    force_symmetric = True
    network = ThorpNet(n_assets, max_weight=max_weight, force_symmetric=force_symmetric)

    print(network)

    n_parameters = 0
    n_parameters += n_assets  # Expected returns
    n_parameters += n_assets * n_assets # Covariance matrix
    n_parameters += 1  # gamma
    n_parameters += 1  # alpha

    true_n_parameters = sum(p.numel() for p in network.parameters() if p.requires_grad)

    assert n_parameters == true_n_parameters

.. testoutput::

    ThorpNet(
      (portfolio_opt_layer): NumericalMarkowitz(
        (cvxpylayer): CvxpyLayer()
      )
    )






.. _writing_custom_networks:

Writing custom networks
-----------------------
One can create infinitely many architectures using :code:`deepdow` and :code:`torch` layers. The bare minimum is to
subclass :code:`torch.nn.Module` and :code:`deepdow.benchmarks.Benchmark` and implement the :code:`forward` method.

See below an example


.. testcode::

    from deepdow.benchmarks import Benchmark

    class AmazingNetwork(torch.nn.Module, Benchmark):
        """Amazing network.

        Parameters
        ----------
        hyper_param : float
            A hyperparameter.


        Attributes
        ----------
        learnable_param : torch.tensor
            A parameter to be learned during training.

        """
        def __init__(self, hyper_param):
            super().__init__()

            self.hyper_param = hyper_param
            self.learnable_param = torch.nn.Parameter(torch.ones(1), requires_grad=True)

        def forward(self, x):
            """Perform forward pass.

            Parameters
            ----------
            x : torch.Tensor
                Tensor of shape `(n_samples, n_channels, lookback, n_assets)` representing the input features.

            Returns
            -------
            weights : torch.Tensor
                Tensor of shape `(n_samples, n_assets)` representing the final allocation.
            """
            x = self.learnable_param * torch.sin(x + self.hyper_param)
            means = abs(x.mean([1, 2])) +  1e-6

            weights = means / means.sum(dim=1, keepdim=True)

            return weights

        def hparams(self):
            return {'hyper_param': self.hyper_param}


    network = AmazingNetwork(2.4)

    n_samples, n_channels, lookback, n_assets = 10, 2, 20, 5
    x = torch.randn(n_samples, n_channels, lookback, n_assets)
    weights = network(x)

    print(weights)

    assert sum(p.numel() for p in network.parameters() if p.requires_grad) == 1


.. testoutput::
    :options: +NORMALIZE_WHITESPACE

    tensor([[0.2186, 0.1135, 0.2441, 0.2321, 0.1917],
                [0.2096, 0.1877, 0.1719, 0.2010, 0.2297],
                [0.1996, 0.2330, 0.1879, 0.1923, 0.1871],
                [0.1911, 0.2407, 0.1675, 0.2020, 0.1986],
                [0.2495, 0.1988, 0.1833, 0.1703, 0.1981],
                [0.2418, 0.1710, 0.1773, 0.1950, 0.2149],
                [0.1715, 0.2285, 0.3046, 0.0921, 0.2034],
                [0.1825, 0.1882, 0.1603, 0.2631, 0.2058],
                [0.2012, 0.1889, 0.1665, 0.2128, 0.2306],
                [0.1924, 0.2749, 0.1898, 0.1486, 0.1942]], grad_fn=<DivBackward0>)




Note that one needs to always implement the :code:`forward` assuming the input shape is
:code:`(n_samples, n_channels, lookback, n_assets)`. The sample dimension should always be independent.
Meaning that shuffling the input **x** along the sample dimension only results in shuffling the output
**weights**.


