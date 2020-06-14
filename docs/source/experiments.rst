.. testsetup::

    import numpy as np
    import torch

    np.random.seed(2)
    torch.manual_seed(2)

.. _experiments:

Experiments
===========
This section focuses on putting all of the previous sections together and proposes a framework for training and
evaluation of networks. The central object is the :code:`deepdow.experiments.Run` class.

To instantiate run, we need to provide multiple parameters:

- :code:`network` - Network to be trained and evaluated. See :ref:`networks` for details.
- :code:`loss` - Loss criterion. See :ref:`losses` for details.
- :code:`train_dataloader` - Dataloader streaming training data. See :ref:`data` for details.
- :code:`val_dataloaders` - Dictionary where keys are names and values are instances of :code:`RigidDataloader`. See :ref:`data` for details.
- :code:`metrics` - Additional metrics to be monitored. See :ref:`losses` for details.
- :code:`benchmarks` - Additional baseline models to be used for comparison. See :ref:`benchmarks` for details.
- :code:`callbacks` - Additional callbacks to be used (on top of the default ones). See :ref:`callbacks` for details.

Once we construct the :code:`Run`, we can start the training and evaluation loop via the :code:`launch` method.

.. testcode::

    from deepdow.benchmarks import OneOverN
    from deepdow.data import InRAMDataset, RigidDataLoader
    from deepdow.experiments import Run
    from deepdow.losses import MaximumDrawdown, SharpeRatio
    from deepdow.nn import LinearNet

    n_samples, n_channels, lookback, n_assets = 200, 2, 20, 6
    horizon = 15

    X = np.random.random((n_samples, n_channels, lookback, n_assets)) - 0.5
    y = np.random.random((n_samples, n_channels, horizon, n_assets)) - 0.5

    dataset = InRAMDataset(X, y)
    train_dataloader = RigidDataLoader(dataset, indices=list(range(100)), batch_size=10)
    val_dataloaders = {'val': RigidDataLoader(dataset, indices=list(range(130, 180)), batch_size=10)}

    network = LinearNet(n_channels, lookback, n_assets)
    loss = SharpeRatio(returns_channel=0)
    benchmarks = {'1overN': OneOverN()}
    metrics = {'drawdown': MaximumDrawdown(returns_channel=0)}

    run = Run(network,
              loss,
              train_dataloader,
              val_dataloaders=val_dataloaders,
              metrics=metrics,
              benchmarks=benchmarks)

    history = run.launch(n_epochs=1)

.. testoutput::

    model   metric    epoch  dataloader
    1overN  drawdown  -1     val           0.283
            loss      -1     val          -0.331

We get results on the benchmarks in the standard output (see above).
Additionally, progress bar is sent to the standard error. It monitors progress of our network. To read more
details on the :code:`Run` class see :ref:`experiments_API`. Last but not least, we also get an
instance of the :code:`History` class. See below section for more information.

History
-------
The :code:`launch` method returns an instance of the :code:`History` class. It captures all the
useful information that was recorded during training. This information can be accessed via the
:code:`metrics` property that is a :code:`pd.DataFrame` with the following columns

- :code:`model` - name of the model
- :code:`metric` - name of the loss
- :code:`value` - value of the loss
- :code:`batch` - batch
- :code:`epoch` - epoch
- :code:`dataloader` - name of the dataloader
- :code:`lookback` - lookback size, by default only using the one from the dataloader
- :code:`timestamp` - it can be used to unique identify a given sample
- :code:`current_time` - time when the entry logged

.. _callbacks:

Callbacks
---------
Callbacks are intended to be run at precise moments of the training loop. All callbacks have a shared interface
:code:`deepdow.callbacks.Callback` that provides the following methods

- :code:`on_batch_begin` - run at the beginning of each **batch**
- :code:`on_batch_end` - run at the end of each **batch**
- :code:`on_epoch_begin` - run at the beginning of each **epoch**
- :code:`on_epoch_end` - run at the end of each **epoch**
- :code:`on_train_begin` - run at the beginning of the **training**
- :code:`on_train_end`- run at the end of the **training**
- :code:`on_train_interrupt` - run in case training interrupted

Each of these methods inputs the :code:`metadata` dictionary. It contains the most recent value of the most
relevant variables.

Note that when constructing a :code:`Run` there are three callbacks inserted by default

- :code:`BenchmarkCallback`
- :code:`ValidationCallback`
- :code:`ProgressBarCallback`

One can chose additional one by defining adding a list of callbacks as the `callbacks` variable.

Lastly, callback instances can access the :code:`Run` instance within under the :code:`run`
attribute. It is always injected when the training is launched.


In what follows, we provide an overview of all available callbacks. For detailed usage instructions
see :ref:`callbacks_API`.


BenchmarkCallback
*****************
Automatically added to `Run` instances. It computes all metrics for all provided benchmarks over all validation
dataloaders.


EarlyStoppingCallback
*********************
This callback monitors a given metric and if there are no improvements over specific number of epochs it stops the
training.

MLFlowCallback
**************
Callback that logs relevant metrics to MLflow.

ModelCheckpointCallback
***********************
Saving a model each epoch it achieves lower than the previous lowest loss.

ProgressBarCallback
*******************
Automatically added to `Run` instances. Displays progress bar with all relevant metrics. One can choose where outputted
with :code:`output` parameter.


TensorBoardCallback
*******************
Callback that logs relevant metrics to MLflow together with images and histograms.


ValidationCallback
******************
Automatically added to `Run` instances. It computes all metrics of the trained network over all validation dataloaders.