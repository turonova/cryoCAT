.. _installing:

.. currentmodule:: crycocat

==========
Installing
==========

Official releases of cryoCAT can be installed from `PyPI <https://pypi.org/project/cryocat>`_::

    pip install cryocat

Since cryoCAT is being developed on daily basis, it is recommended to clone the `github repository <https://github.com/turonova-lab/cryocat>`_
and install it with::

    pip install -e /path/to/your/local/copy/of/cryocat/.

The '-e' will install editable version of the package - this is important because you do not have to upgrade 
it every time something is changed. The only thing you have to do is to pull the repo to have the latest version, all 
changes will be visible without running pip install upgrade.

Dependencies
~~~~~~~~~~~~

Supported Python versions
^^^^^^^^^^^^^^^^^^^^^^^^^

- Python 3.12+

Optional: isotropic remeshing (gpytoolbox)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Isotropic remeshing (:meth:`cryocat.core.surface.Mesh.remesh_isotropic`, also used by default in
:meth:`cryocat.core.surface.Mesh.from_mrc_screened_poisson`) needs
`gpytoolbox <https://github.com/sgsellan/gpytoolbox>`_. Install it in a second step, without its
dependencies::

    pip install cryocat                          # or, from a clone: pip install -e .
    pip install --no-deps "gpytoolbox==0.3.8"

gpytoolbox is installed with ``--no-deps`` because its package metadata caps scipy and numpy below
the versions cryoCAT uses; the remeshing itself works with the newer versions. Its remaining
runtime dependencies (numpy, scipy, scikit-image, scs) are installed with cryoCAT. Afterwards
``pip check`` reports that gpytoolbox requires older scipy and numpy versions; this warning is
expected and harmless.

gpytoolbox has no wheels for Linux ARM (aarch64), so the second step fails there. Everything else
in cryoCAT works without it; screened Poisson meshes can still be made with
``Mesh.from_mrc_screened_poisson(..., remesh_edge_length=None)``.


Getting help
~~~~~~~~~~~~

If you think you've encountered a bug in cryoCAT, please report it on the
`GitHub issue tracker <https://github.com/turonova-lab/cryocat/issues>`_.
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
