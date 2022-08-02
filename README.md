![final](https://user-images.githubusercontent.com/18519371/79003829-afca6380-7b53-11ea-8322-f05577536957.png)

[![codecov](https://codecov.io/gh/jankrepl/deepdow/branch/master/graph/badge.svg)](https://codecov.io/gh/jankrepl/deepdow)
[![Documentation Status](https://readthedocs.org/projects/deepdow/badge/?version=latest)](https://deepdow.readthedocs.io/en/latest/?badge=latest)
[![PyPI version](https://badge.fury.io/py/deepdow.svg)](https://badge.fury.io/py/deepdow)
[![DOI](https://zenodo.org/badge/237742797.svg)](https://zenodo.org/badge/latestdoi/237742797)

`deepdow` (read as "wow") is a Python package connecting portfolio optimization and deep learning. Its goal is to
facilitate research of networks that perform weight allocation in **one forward pass**.


# Installation
```bash
pip install deepdow
```
# Resources
- [**Getting started**](https://deepdow.readthedocs.io/en/latest/auto_examples/end_to_end/getting_started.html)
- [**Detailed documentation**](https://deepdow.readthedocs.io/en/latest)
- [**More examples**](https://deepdow.readthedocs.io/en/latest/auto_examples/index.html)

# Description
`deepdow` attempts to **merge** two very common steps in portfolio optimization
1. Forecasting of future evolution of the market (LSTM, GARCH,...)
2. Optimization problem design and solution (convex optimization, ...)

It does so by constructing a pipeline of layers. The last layer performs the allocation and all the previous ones serve
as feature extractors. The overall network is **fully differentiable** and one can optimize its parameters by gradient
descent algorithms.

# `deepdow` is not ...
- focused on active trading strategies, it only finds allocations to be held over some horizon (**buy and hold**)
    - one implication is that transaction costs associated with frequent, short-term trades, will not be a primary concern 
- a reinforcement learning framework, however, one might easily reuse `deepdow` layers in other deep learning applications
- a single algorithm, instead, it is a framework that allows for easy experimentation with powerful building blocks


# Some features
- all layers built on `torch` and fully differentiable
- integrates differentiable convex optimization (`cvxpylayers`)
- implements clustering based portfolio allocation algorithms
- multiple dataloading strategies (`RigidDataLoader`, `FlexibleDataLoader`)
- integration with `mlflow` and `tensorboard` via callbacks
- provides variety of losses like sharpe ratio, maximum drawdown, ...
- simple to extend and customize
- CPU and GPU support

# Citing
If you use `deepdow` (including ideas proposed in the documentation, examples and tests) in your research please **make sure to cite it**.
To obtain all the necessary citing information, click on the **DOI badge** at the beginning of this README and you will be automatically redirected to an external website.
Note that we are currently using [Zenodo](https://zenodo.org/).
