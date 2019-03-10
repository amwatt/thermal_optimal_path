
About Thermal Optimal Path
==========================

The Thermal Optimal Path methods originates from `Non-parametric Determination of Real-Time Lag Structure between Two Time Series: the “Optimal Thermal Causal Path” Method, D.Sornette and  W.-X. Zhou (2004) <https://arxiv.org/abs/cond-mat/0408166)>`_.

It enables dynamic lead/lag analysis between two time series. Borrowed from the statistical physics literature on directed polymers, it is suited for short and noisy time series.


Main features
=============

* Implementation of the original Thermal Optimal Path method:

  - Partition function computed with Numba's JIT to achieve 1000 times speedup over a pure Python implementation
  - Partition function's average path
  - Mean squared error model for correlated and anti-correlated time series

Notebooks
=========

Jupyter notebooks provide example uses cases.


How to get it
=============

The master branch on GitHub is the most up to date code.


License
=======

Modified BSD (3-clause), as found in the LICENSE file.


Discussion and Development
==========================

We welcome feedback about usability and suggestions for improvements.
