lens
====

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.2593336.svg
   :target: https://doi.org/10.5281/zenodo.2593336

``lens`` is a library for exploring data in Pandas DataFrames. It computes
single column summary statistics and estimates the correlation between columns.
We wrote ``lens`` when we realised that the initial steps of acquiring a new
data set were almost formulaic: What data type is in this column? How many null
values are there? Which columns are correlated? What's the distribution of this
value? ``lens`` calculates all this for you.

See the documentation_ for more details.

.. _documentation: https://lens.readthedocs.io/en/latest

Installation
------------

``lens`` can be installed from PyPI with ``pip``:

.. code-block:: bash

    pip install lens

Testing
-------

Tests can be run using [`tox`](https://tox.readthedocs.io) (replace `py37` with
the version of python you wish to use to run the tests):

.. code-block:: bash

    pip install tox
    tox -e py37

License
-------

``lens`` is licensed under the Apache License, see LICENSE.txt for details.
