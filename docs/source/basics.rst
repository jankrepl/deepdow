.. _basics:

Basics
======
This page introduces all the important concepts used within :code:`deepdow`.

Data
----
Financial timeseries can be seen as a 3D tensor with the following dimensions

- **time**
- **asset**
- **indicator/channel**

To give a specific example, one can investigate daily (**time** dimension) open price returns, close price returns and
volumes (**channel** dimension) of multiple NASDAQ stocks (**asset** dimension). Graphically, one can imagine

.. image:: https://i.imgur.com/RYcdN6y.png
   :align: center
   :width: 450


Let us denote the shape of our tensor :code:`(n_channels, n_timesteps, n_assets) = (3, 10, 6)`. By fixing a time step
(representing **now**), we can split our tensor into 3 disjoint subtensors:

- **x**  - :code:`(n_channels, lookback, n_assets) = (3, 5, 6)`
- **g** - :code:`(n_channels, gap, n_assets) = (3, 1, 6)`
- **y**  - :code:`(n_channels, horizon, n_assets) = (3, 4, 6)`


.. image:: https://i.imgur.com/rsttnxn.png
   :align: center
   :width: 400

Firstly, **x** represents all the knowledge about the past and present. The second tensor **g** represents information
contained in the immediate future that we cannot use to make investment decisions. Finally, **y** is the future
evolution of the market.

One can now move along the time dimension and apply the same decomposition at every time step. This method
of generating a dataset is called the **rolling window**. To illustrate this idea, let us take a slightly
bigger starting tensor (:code:`n_timesteps = 12`) while keeping :code:`lookback = 5`, :code:`gap = 1` and
:code:`horizon = 4`. Let's roll it!


.. image:: https://i.imgur.com/okSUzOk.pngc
   :align: center
   :width: 550

We now possess a collection of 3 **feature** tensors (**x1**, **x2**, **x3**) and 3 **label** tensors (**y1**, **y2**, **y3**).
And that is all we need!

Predictions AKA weights
-----------------------
In the :code:`deepdow` framework, we study networks that input **x** and return a single weight allocation **w** of
shape :code:`(n_assets,)` such that :math:`\sum_{i} w_{i} = 1`. In other words, given our past knowledge **x** we
construct a portfolio **w** that we buy right away and hold for :code:`horizon` time steps. Let **F** some neural network
with parameters :math:`\theta`, the below image represents the high level prediction pipeline:

.. image:: https://i.imgur.com/sJ30WFE.png
   :align: center
   :width: 500

.. _basics_loss:

Loss
----
The last piece of the puzzle is definition of the loss function. In the most general terms, the per sample loss **L**
is any function that inputs **w** and **y** and outputs a real number. However, in most cases we first compute the
portfolio returns **r** over each time step in the :code:`horizon` and then apply some summarization function
**S** like mean, standard deviation, etc.


.. image:: https://i.imgur.com/L0A2bRS.png
   :align: center
   :width: 700



Assumptions
-----------
Before finishing this chapter, let us summarize the important assumptions :code:`deepdow` is making

- The time dimension is **contiguous** with a single frequency (i.e. daily)
- The predicted weights **w** are turned into an actual investment that is **held** over :code:`horizon` time steps


