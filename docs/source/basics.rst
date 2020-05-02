Basics
======
This page introduces all the important concepts used within :code:`deepdow`.

Data
----
Financial timeseries can be seen a 3D tensor with the following dimensions

- **time**
- **asset**
- **indicator/channel**

To give a specific example, one can investigate daily (**time** dimension) open prices, close prices and volumes (**channel**
dimension) of multiple NASDAQ stocks (**asset** dimension). Graphically, one can imagine



.. image:: https://i.imgur.com/9Y7uRzE.png
   :align: center
   :width: 300


Let us denote the shape of our tensor :code:`(n_channels, n_timesteps, n_assets) = (3, 10, 6)`. By fixing a time step
(representing **now**), we can split our tensor into 3 disjoint subtensors:

- **x**  - :code:`(n_channels, lookback, n_assets) = (3, 5, 6)`
- **g** - :code:`(n_channels, gap, n_assets) = (3, 1, 6)`
- **y**  - :code:`(n_channels, horizon, n_assets) = (3, 4, 6)`


.. image:: https://i.imgur.com/rsttnxn.png
   :align: center
   :width: 300

Firstly, **x** represents all the knowledge about the past and present. The second tensor **g** represents information
contained in the immediate future that we cannot use to make investment decisions. Finally, **y** is the future
evolution of the market.

One can now move along the time dimension and apply the same decomposition at every time step. This method
of generating a dataset is called the **rolling window**. To illustrate this idea, let us take a slightly
bigger starting tensor (:code:`n_timesteps = 12`) while keeping :code:`loookback = 5`, :code:`gap = 1` and
:code:`horizon = 4`. Let's roll it!


.. image:: https://i.imgur.com/D9HlKpZ.png
   :align: center
   :width: 550

We now possess a collection of 3 **feature** tensors (**x1**, **x2**, **x3**) and 3 **label** tensors (**y1**, **y2**, **y3**).
And that is all we need!

Predictions AKA weights
-----------------------
