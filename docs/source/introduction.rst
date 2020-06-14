Introduction
============
:code:`deepdow` is a framework that focuses on portfolio optimization via end-to-end deep learning. Its goal is to
facilitate research of networks that perform weight allocation in **one forward pass**.

Name
----
The name of the packages is inspired by the father of technical analysis—**Charles Dow**.


Traditional portfolio optimization
----------------------------------
As described by Markowitz [MARK1952]_, portfolio optimization is commonly divided into 2 separate stages:

1. **Creation of beliefs about the future performances of securities**
2. **Finding optimal portfolio given these beliefs**

One extremely popular example of this two stage paradigm is:

.. _traditional:

1. **Estimation of expected returns** :math:`\boldsymbol{\mu}` **and covariance matrix** :math:`\boldsymbol{\Sigma}`
2. **Solving a convex optimization problem, e.g.** :math:`\boldsymbol{\mu}^T \textbf{w} - \gamma \textbf{w}^T  \boldsymbol{\Sigma} \textbf{w}` **such that** :math:`\textbf{w} > 0` **and** :math:`{\bf 1}^T \textbf{w}=1`

Commonly, these two steps are absolutely separated since they require different approaches

1. **Predictive modeling (statistics + machine learning)**
2. **Objective function and constraints design**

Not surprisingly, one needs to use totally different tools. Below are some examples from the Python ecosystem.

1. :code:`numpy`, :code:`pandas`, :code:`scikit-learn`, :code:`statsmodels`, :code:`tensorflow`, :code:`pytorch`, ...
2. :code:`cvxpy`, :code:`cvxopt`, :code:`scipy`, ...


Why DeepDow different?
----------------------
:code:`deepdow` strives to merge the above mentioned two steps into **one**. The fundamental idea is to construct
end-to-end deep networks that input the rawest features (returns,
volumes, ...) and output asset allocation. This approach has multiple benefits:

- Hyperparameters can be turned into **trainable weights** (i.e. :math:`\gamma` in :ref:`2nd stage <traditional>`)
- Leveraging deep learning to extract useful features for **allocation** (rather than just prediction)
- **Single** loss function


References
----------
.. [MARK1952]
    Markowitz, H. (1952), PORTFOLIO SELECTION. The Journal of Finance, 7: 77-91.
    doi:10.1111/j.1540-6261.1952.tb01525.x