.. _installing:

.. currentmodule:: crycocat

==========
Installing
==========

Official releases of cryoCAT can be installed from `PyPI <https://pypi.org/project/cryocat>`_::

    pip install cryocat

Since cryoCAT is being developed on daily basis, it is recommended to clone the `github repository <https://github.com/turonova/cryocat>`_
and install it with::

    pip install -e /path/to/your/local/copy/of/cryocat/.

The '-e' will install editable version of the package - this is important because you do not have to upgrade 
it every time something is changed. The only thing you have to do is to pull the repo to have the latest version, all 
changes will be visible without running pip install upgrade.

Dependencies
~~~~~~~~~~~~

Supported Python versions
^^^^^^^^^^^^^^^^^^^^^^^^^

- Python 3.9+

Mandatory dependencies
^^^^^^^^^^^^^^^^^^^^^^

- `numpy <https://numpy.org/>`__

- `pandas <https://pandas.pydata.org/>`__

- `matplotlib <https://matplotlib.org>`__

- `scipy <https://www.scipy.org/>`__

- `scikit-image <https://scikit-image.org/>`__

- `starfile <https://github.com/teamtomo/starfile>`__

- `emfile <https://github.com/teamtomo/emfile>`__

- `mrcfile <https://github.com/ccpem/mrcfile>`__

- `seaborn <https://seaborn.pydata.org/>`__

- `scikit-learn <https://scikit-learn.org>`__

- `lmfit <https://lmfit.github.io/lmfit-py>`__


Getting help
~~~~~~~~~~~~

If you think you've encountered a bug in cryoCAT, please report it on the
`GitHub issue tracker <https://github.com/turonova/cryocat/issues>`_.
To be useful, bug reports must include the following information:

- A reproducible code example that demonstrates the problem
- The output that you are seeing
- The specific versions of cryoCAT that you are working with

It is preferable that your example generate synthetic data to
reproduce the problem. If you can only demonstrate the issue with your
actual data, you will need to share it.

If you've encountered an error, searching the specific text of the message
before opening a new issue can often help you solve the problem quickly and
avoid making a duplicate report.
