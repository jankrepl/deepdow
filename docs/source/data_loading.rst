.. _data:

Data Loading
============

.. testsetup::

    import pandas as pd
    from pandas import Timestamp
    import torch

    from deepdow.data import InRAMDataset
    from deepdow.utils import raw_to_Xy

    dct = {('MSFT', 'Close'): {Timestamp('2016-01-04 00:00:00'): 54.79999923706055,
      Timestamp('2016-01-05 00:00:00'): 55.04999923706055,
      Timestamp('2016-01-06 00:00:00'): 54.04999923706055,
      Timestamp('2016-01-07 00:00:00'): 52.16999816894531,
      Timestamp('2016-01-08 00:00:00'): 52.33000183105469,
      Timestamp('2016-01-11 00:00:00'): 52.29999923706055,
      Timestamp('2016-01-12 00:00:00'): 52.779998779296875,
      Timestamp('2016-01-13 00:00:00'): 51.63999938964844,
      Timestamp('2016-01-14 00:00:00'): 53.11000061035156,
      Timestamp('2016-01-15 00:00:00'): 50.9900016784668,
      Timestamp('2016-01-19 00:00:00'): 50.560001373291016,
      Timestamp('2016-01-20 00:00:00'): 50.790000915527344,
      Timestamp('2016-01-21 00:00:00'): 50.47999954223633,
      Timestamp('2016-01-22 00:00:00'): 52.290000915527344,
      Timestamp('2016-01-25 00:00:00'): 51.790000915527344,
      Timestamp('2016-01-26 00:00:00'): 52.16999816894531,
      Timestamp('2016-01-27 00:00:00'): 51.220001220703125,
      Timestamp('2016-01-28 00:00:00'): 52.060001373291016,
      Timestamp('2016-01-29 00:00:00'): 55.09000015258789,
      Timestamp('2016-02-01 00:00:00'): 54.709999084472656},
     ('MSFT', 'Volume'): {Timestamp('2016-01-04 00:00:00'): 53778000,
      Timestamp('2016-01-05 00:00:00'): 34079700,
      Timestamp('2016-01-06 00:00:00'): 39518900,
      Timestamp('2016-01-07 00:00:00'): 56564900,
      Timestamp('2016-01-08 00:00:00'): 48754000,
      Timestamp('2016-01-11 00:00:00'): 36943800,
      Timestamp('2016-01-12 00:00:00'): 36095500,
      Timestamp('2016-01-13 00:00:00'): 66883600,
      Timestamp('2016-01-14 00:00:00'): 52381900,
      Timestamp('2016-01-15 00:00:00'): 71820700,
      Timestamp('2016-01-19 00:00:00'): 43564500,
      Timestamp('2016-01-20 00:00:00'): 63273000,
      Timestamp('2016-01-21 00:00:00'): 40191200,
      Timestamp('2016-01-22 00:00:00'): 37555800,
      Timestamp('2016-01-25 00:00:00'): 34707700,
      Timestamp('2016-01-26 00:00:00'): 28900800,
      Timestamp('2016-01-27 00:00:00'): 36775200,
      Timestamp('2016-01-28 00:00:00'): 62513800,
      Timestamp('2016-01-29 00:00:00'): 83611700,
      Timestamp('2016-02-01 00:00:00'): 44208500},
     ('AAPL', 'Close'): {Timestamp('2016-01-04 00:00:00'): 105.3499984741211,
      Timestamp('2016-01-05 00:00:00'): 102.70999908447266,
      Timestamp('2016-01-06 00:00:00'): 100.69999694824219,
      Timestamp('2016-01-07 00:00:00'): 96.44999694824219,
      Timestamp('2016-01-08 00:00:00'): 96.95999908447266,
      Timestamp('2016-01-11 00:00:00'): 98.52999877929688,
      Timestamp('2016-01-12 00:00:00'): 99.95999908447266,
      Timestamp('2016-01-13 00:00:00'): 97.38999938964844,
      Timestamp('2016-01-14 00:00:00'): 99.5199966430664,
      Timestamp('2016-01-15 00:00:00'): 97.12999725341797,
      Timestamp('2016-01-19 00:00:00'): 96.66000366210938,
      Timestamp('2016-01-20 00:00:00'): 96.79000091552734,
      Timestamp('2016-01-21 00:00:00'): 96.30000305175781,
      Timestamp('2016-01-22 00:00:00'): 101.41999816894531,
      Timestamp('2016-01-25 00:00:00'): 99.44000244140625,
      Timestamp('2016-01-26 00:00:00'): 99.98999786376953,
      Timestamp('2016-01-27 00:00:00'): 93.41999816894531,
      Timestamp('2016-01-28 00:00:00'): 94.08999633789062,
      Timestamp('2016-01-29 00:00:00'): 97.33999633789062,
      Timestamp('2016-02-01 00:00:00'): 96.43000030517578},
     ('AAPL', 'Volume'): {Timestamp('2016-01-04 00:00:00'): 67649400,
      Timestamp('2016-01-05 00:00:00'): 55791000,
      Timestamp('2016-01-06 00:00:00'): 68457400,
      Timestamp('2016-01-07 00:00:00'): 81094400,
      Timestamp('2016-01-08 00:00:00'): 70798000,
      Timestamp('2016-01-11 00:00:00'): 49739400,
      Timestamp('2016-01-12 00:00:00'): 49154200,
      Timestamp('2016-01-13 00:00:00'): 62439600,
      Timestamp('2016-01-14 00:00:00'): 63170100,
      Timestamp('2016-01-15 00:00:00'): 79833900,
      Timestamp('2016-01-19 00:00:00'): 53087700,
      Timestamp('2016-01-20 00:00:00'): 72334400,
      Timestamp('2016-01-21 00:00:00'): 52161500,
      Timestamp('2016-01-22 00:00:00'): 65800500,
      Timestamp('2016-01-25 00:00:00'): 51794500,
      Timestamp('2016-01-26 00:00:00'): 75077000,
      Timestamp('2016-01-27 00:00:00'): 133369700,
      Timestamp('2016-01-28 00:00:00'): 55678800,
      Timestamp('2016-01-29 00:00:00'): 64416500,
      Timestamp('2016-02-01 00:00:00'): 40943500}}

    raw_df = pd.DataFrame(dct)

    lookback, gap, horizon = 5, 2, 4

    X, timestamps, y, asset_names, indicators = raw_to_Xy(raw_df,
                                                          lookback=lookback,
                                                          gap=gap,
                                                          horizon=horizon)

    dataset = InRAMDataset(X, y, timestamps=timestamps, asset_names=asset_names)


Introduction
------------
:code:`deepdow` offers multiple utility functions and classes that turn raw data into tensors used by :ref:`layers`
and :ref:`losses`.



See below a scheme of the overall **datamodel** (starting at the top)

.. image:: https://i.imgur.com/Q8Tgnb5.png

We dedicate an entire section to each of the elements.

Raw data
--------
Let us assume, that our raw data :code:`raw_df` is stored in a :code:`pd.DataFrame`. There are :code:`n_timesteps` rows
representing different timesteps with the same time frequency but potentially with gaps (due to non-business days etc.).
They are indexed by :code:`pd.DatetimeIndex`. The columns are indexed by :code:`pd.MultiIndex` where the first level
represents the :code:`n_assets` different **assets**. The second level then represents
the :code:`n_channels` **channels** (indicators) like volume or close price. For the rest of the this
page we will be using the below example

+---------------------+---------+-------------+---------+--------------+
| Asset               |    MSFT |        MSFT |    AAPL |         AAPL |
+---------------------+---------+-------------+---------+--------------+
| Channel             |   Close |      Volume |   Close |       Volume |
+=====================+=========+=============+=========+==============+
| 2016-01-04 00:00:00 |   54.80 | 53778000.00 |  105.35 |  67649400.00 |
+---------------------+---------+-------------+---------+--------------+
| 2016-01-05 00:00:00 |   55.05 | 34079700.00 |  102.71 |  55791000.00 |
+---------------------+---------+-------------+---------+--------------+
| 2016-01-06 00:00:00 |   54.05 | 39518900.00 |  100.70 |  68457400.00 |
+---------------------+---------+-------------+---------+--------------+
| 2016-01-07 00:00:00 |   52.17 | 56564900.00 |   96.45 |  81094400.00 |
+---------------------+---------+-------------+---------+--------------+
| 2016-01-08 00:00:00 |   52.33 | 48754000.00 |   96.96 |  70798000.00 |
+---------------------+---------+-------------+---------+--------------+
| 2016-01-11 00:00:00 |   52.30 | 36943800.00 |   98.53 |  49739400.00 |
+---------------------+---------+-------------+---------+--------------+
| 2016-01-12 00:00:00 |   52.78 | 36095500.00 |   99.96 |  49154200.00 |
+---------------------+---------+-------------+---------+--------------+
| 2016-01-13 00:00:00 |   51.64 | 66883600.00 |   97.39 |  62439600.00 |
+---------------------+---------+-------------+---------+--------------+
| 2016-01-14 00:00:00 |   53.11 | 52381900.00 |   99.52 |  63170100.00 |
+---------------------+---------+-------------+---------+--------------+
| 2016-01-15 00:00:00 |   50.99 | 71820700.00 |   97.13 |  79833900.00 |
+---------------------+---------+-------------+---------+--------------+
| 2016-01-19 00:00:00 |   50.56 | 43564500.00 |   96.66 |  53087700.00 |
+---------------------+---------+-------------+---------+--------------+
| 2016-01-20 00:00:00 |   50.79 | 63273000.00 |   96.79 |  72334400.00 |
+---------------------+---------+-------------+---------+--------------+
| 2016-01-21 00:00:00 |   50.48 | 40191200.00 |   96.30 |  52161500.00 |
+---------------------+---------+-------------+---------+--------------+
| 2016-01-22 00:00:00 |   52.29 | 37555800.00 |  101.42 |  65800500.00 |
+---------------------+---------+-------------+---------+--------------+
| 2016-01-25 00:00:00 |   51.79 | 34707700.00 |   99.44 |  51794500.00 |
+---------------------+---------+-------------+---------+--------------+
| 2016-01-26 00:00:00 |   52.17 | 28900800.00 |   99.99 |  75077000.00 |
+---------------------+---------+-------------+---------+--------------+
| 2016-01-27 00:00:00 |   51.22 | 36775200.00 |   93.42 | 133369700.00 |
+---------------------+---------+-------------+---------+--------------+
| 2016-01-28 00:00:00 |   52.06 | 62513800.00 |   94.09 |  55678800.00 |
+---------------------+---------+-------------+---------+--------------+
| 2016-01-29 00:00:00 |   55.09 | 83611700.00 |   97.34 |  64416500.00 |
+---------------------+---------+-------------+---------+--------------+
| 2016-02-01 00:00:00 |   54.71 | 44208500.00 |   96.43 |  40943500.00 |
+---------------------+---------+-------------+---------+--------------+

.. testcode::

    assert isinstance(raw_df, pd.DataFrame)
    assert isinstance(raw_df.index, pd.DatetimeIndex)
    assert isinstance(raw_df.columns, pd.MultiIndex)
    assert raw_df.shape == (20, 4)

raw_to_Xy
---------
The quickest way to get going given :code:`raw_df` is to use the :code:`deepdow.utils.raw_to_Xy` function.
It performs the following steps

1. exclusion of undesired assets and channels (see :code:`included_assets` and :code:`included_indicators`)
2. adding missing rows - timestamps implied by the specified frequency :code:`freq`
3. filling missing values (forward fill followed by backward fill)
4. computation of returns (if :code:`use_log` then logarithmic else simple) - the first timestep is automatically deleted
5. running the rolling window (see :ref:`basics`) given :code:`lookback`, :code:`gap` and :code:`horizon`

We get the following outputs

- :code:`X` - numpy array of shape :code:`(n_samples, n_channels, lookback, n_assets)` representing **features**
- :code:`timestamps`- list of length :code:`n_samples` representing timestamp of each sample
- :code:`y` - numpy array of shape :code:`(n_samples, n_channels, horizon, n_assets)` representing **targets**
- :code:`asset_names` - list of length :code:`n_assets` representing asset names
- :code:`indicators` - list of length :code:`n_channels` representing channel / indicator names

Note that in our example :code:`n_samples = n_timesteps - lookback - horizon - gap + 1` since there is a single
missing day (`2016-01-18`) w.r.t. the default :code:`B` frequency that is going to be forward filled.

.. testcode::

    from deepdow.utils import raw_to_Xy


    n_timesteps = len(raw_df)  # 20
    n_channels = len(raw_df.columns.levels[0])  # 2
    n_assets = len(raw_df.columns.levels[1])  # 2

    lookback, gap, horizon = 5, 2, 4

    X, timestamps, y, asset_names, indicators = raw_to_Xy(raw_df,
                                                          lookback=lookback,
                                                          gap=gap,
                                                          freq="B",
                                                          horizon=horizon)

    n_samples =  n_timesteps - lookback - horizon - gap + 1  # 10

    assert X.shape == (n_samples, n_channels, lookback, n_assets)
    assert timestamps[0] == raw_df.index[lookback]
    assert asset_names == ['AAPL', 'MSFT']
    assert indicators == ['Close', 'Volume']


.. _inramdataset:

InRAMDataset
------------
The next step is to start migrating our custom lists and numpy arrays to native PyTorch classes. For more details see
`Official tutorial <https://pytorch.org/tutorials/beginner/data_loading_tutorial.html>`_. First of all,
:code:`deepdow` implements its own subclass of :code:`torch.utils.data.Dataset` called :code:`InRAMDataset`. Its goal
is to encapsulate the above generated :code:`X`, :code:`y`, :code:`timestamps` and  :code:`asset_names` and define
per sample loading.

.. testcode::

    from deepdow.data import InRAMDataset

    dataset = InRAMDataset(X, y, timestamps=timestamps, asset_names=asset_names)

    X_sample, y_sample, timestamp_sample, asset_names = dataset[0]

    assert isinstance(dataset, torch.utils.data.Dataset)
    assert len(dataset) == 10

    assert torch.is_tensor(X_sample)
    assert X_sample.shape == (2, 5, 2)  # (n_channels, lookback, n_assets)

    assert torch.is_tensor(y_sample)
    assert y_sample.shape == (2, 4, 2)  # (n_channels, horizon, n_assets)

    assert timestamp_sample == timestamps[0]


Additionally, one can pass a transformation :code:`transform` that can serve as preprocessing or data augmentation.
Currently implemented transforms under :code:`deepdow.data` are

- :code:`Compose` - basically a copy of `Compose` from Torch Vision
- :code:`Dropout` - randomly setting elements to zero
- :code:`Multiply` - multiplying all elements by a constant
- :code:`Noise` - add Gaussian noise
- :code:`Scale` - centering and scaling (similar to scikit-learn :code:`StandardScaler` and :code:`RobustScaler`)

All of the transforms are not in place.

.. _dataloaders:

Dataloaders
-----------
The last ingredient in the data pipeline are dataloaders. Their goal is to stream batches of samples for training and
validation. :code:`deepdow` provides two options

- **RigidDataLoader** - lookback, horizon and assets **are constant** over different batches
- **FlexibleDataLoader** - lookback, horizon and assets **can change** over different batches

Both of them are subclassing :code:`torch.utils.data.DataLoader` and therefore inherit its functionality. One important
example is the :code:`batch_size` parameter. However, they also add new functionality. Notably one can use the
parameter :code:`indices` to specify which samples of the original dataset are going to be streamed. The
**train, validation and test split** can be performed via this parameter. Last but not least they both have its
specific parameters that we describe in the following subsections.

RigidDataLoader
****************
This dataloader streams batches without making fundamental changes to :code:`X_batch` or :code:`y_batch`.

    - The samples are shuffled
    - The shapes are

        - :code:`X_batch.shape = (batch_size, n_channels, lookback, n_assets)`
        - :code:`y_batch.shape = (batch_size, n_channels, horizon, n_assets)`
        - :code:`len(timestamps_batch) = batch_size`
        - :code:`len(asset_names_batch) = n_assets`


    - at construction one can redefine :code:`lookback`, :code:`horizon` and :code:`asset_ixs` to create a new subset



.. testcode::

    from deepdow.data import RigidDataLoader

    torch.manual_seed(1)
    batch_size = 4

    dataloader = RigidDataLoader(dataset, batch_size=batch_size)

    for X_batch, y_batch, timestamps_batch, asset_names_batch in dataloader:
        print(X_batch.shape)
        print(y_batch.shape)
        print(asset_names_batch)
        print(list(map(str, timestamps_batch)))
        print()


.. testoutput::
    :options: +NORMALIZE_WHITESPACE


    torch.Size([4, 2, 5, 2])
    torch.Size([4, 2, 4, 2])
    ['AAPL', 'MSFT']
    ['2016-01-15 00:00:00', '2016-01-19 00:00:00', '2016-01-22 00:00:00', '2016-01-13 00:00:00']

    torch.Size([4, 2, 5, 2])
    torch.Size([4, 2, 4, 2])
    ['AAPL', 'MSFT']
    ['2016-01-14 00:00:00', '2016-01-12 00:00:00', '2016-01-11 00:00:00', '2016-01-20 00:00:00']

    torch.Size([2, 2, 5, 2])
    torch.Size([2, 2, 4, 2])
    ['AAPL', 'MSFT']
    ['2016-01-21 00:00:00', '2016-01-18 00:00:00']

The big advantage of :code:`RigidDataloader` is that the one can use it easily for evaluation purposes since
the shape of batches is always the same. For example, we can be sure the :code:`horizon` in the :code:`y_batch`
is going to be identical and therefore the predicted portfolio will be always held for the :code:`horizon` number
of timesteps.

FlexibleDataLoader
******************
The goal of this dataloader is to introduce major structural changes to the streamed batches :code:`X_batch` or
:code:`y_batch`. The goal is to randomly create subtensors of them. See below important features

    - :code:`lookback_range` tuple specifies the min and max lookback a :code:`X_batch` can have. The actual lookback is sampled **uniformly** for every batch.
    - :code:`horizon_range` tuple specifies the min and max horizon a :code:`y_batch` can have. Sampled **uniformly**.
    - If :code:`asset_ixs` not specified then :code:`n_assets_range` tuple is the min and max number of assets in :code:`X_batch` and :code:`y_batch`. The actual assets sampled randomly.


.. testcode::
    :skipif: True

    from deepdow.data import FlexibleDataLoader

    torch.manual_seed(3)
    batch_size = 4

    dataloader = FlexibleDataLoader(dataset,
                                    batch_size=batch_size,
                                    n_assets_range=(2, 3),  # keep n_assets = 2 but shuffle randomly
                                    lookback_range=(2, 6),  # sampled uniformly from [2, 6)
                                    horizon_range=(2, 5))   # sampled uniformly from [2, 5)

    for X_batch, y_batch, timestamps_batch, asset_names_batch in dataloader:
        print(X_batch.shape)
        print(y_batch.shape)
        print(asset_names_batch)
        print(list(map(str, timestamps_batch)))
        print()

.. testoutput::
    :skipif: True
    :options: +NORMALIZE_WHITESPACE

    torch.Size([4, 2, 5, 2])
    torch.Size([4, 2, 2, 2])
    ['AAPL', 'MSFT']
    ['2016-01-20 00:00:00', '2016-01-15 00:00:00', '2016-01-13 00:00:00', '2016-01-22 00:00:00']

    torch.Size([4, 2, 4, 2])
    torch.Size([4, 2, 2, 2])
    ['MSFT', 'AAPL']
    ['2016-01-12 00:00:00', '2016-01-18 00:00:00', '2016-01-11 00:00:00', '2016-01-21 00:00:00']

    torch.Size([2, 2, 4, 2])
    torch.Size([2, 2, 3, 2])
    ['AAPL', 'MSFT']
    ['2016-01-19 00:00:00', '2016-01-14 00:00:00']

The main purpose of this dataloader is to use it for training. One can design networks that can perform
a forward pass of an input :code:`X` with variable shapes (i.e. RNN over the time dimension). This is where
:code:`FlexibleDataLoader` comes in handy because it can stream these variable inputs.

.. warning::

    As an example when **not** to use :code:`FlexibleDataLoader` let us consider a dummy network. This
    network flattens the input tensor into a 1D vector of length :code:`n_channels * lookback * n_assets`. Afterwards,
    it applies a linear layer and finally uses some allocation layer (softmax). In this case, one cannot just
    stream tensors of different sizes. Additionally, if we randomly shuffle the order of assets (while keeping the overall
    number equal to :code:`n_assets`) the linear model will have no way of learning asset specific features.
