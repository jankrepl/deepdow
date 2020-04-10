Introduction
============

Name
----
The name of the packages is inspired by the father of the Technical Analysis—**Charles Dow**.


Why?
----
As described by Markowitz [MARK1952]_, portfolio optimization is commonly divided into 2 separate stages:

1. **Creation of beliefs about the future performances of securities**
2. **Finding optimal portfolio given these beliefs**

One extremely popular example of this two stage paradigm is

1. Estimation of expected returns :math:`\boldsymbol{\mu}` and covariance matrix :math:`\boldsymbol{\Sigma}`
2. Solving a convex optimization problem, where the objective is e.g. :math:`\boldsymbol{\mu}^T \textbf{w} -  \textbf{w}^T  \boldsymbol{\Sigma} \textbf{w}` with constraints :math:`\textbf{w} > 0` and :math:`{\bf 1}^T \textbf{w}=1`

Commonly, these two steps are absolutely separated since they require different approaches

1. Predictive modeling (statistics + machine learning)
2. Objective function and constraints design

Not suprisingly, one needs to use totally different tools. Below are some examples from the Python ecosystem.

1. :code:`numpy`, :code:`pandas`, :code:`scikit-learn`, :code:`statsmodels`, :code:`tensorflow`, :code:`pytorch`, ...
2. :code:`cvxpy`, :code:`cvxopt`, :code:`scipy`, ...


Conceptually, this separation is also problematic because prediction errors in the first step might have a
devastating effect on the optimization step.



References
----------
.. [MARK1952]
    Markowitz, H. (1952), PORTFOLIO SELECTION*. The Journal of Finance, 7: 77-91.
    doi:10.1111/j.1540-6261.1952.tb01525.x