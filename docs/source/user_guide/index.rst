.. _user_guide:

.. currentmodule:: cryocat

.. toctree::
   :maxdepth: 2
   :hidden:

   Tiltstack and Mdoc modules: Functions for preprocessing tilt-series <preprocessing_basics>
   Geometric conventions <geometric_conventions>
   Motl basics <motl_basics>
   Motl advanced <motl_advanced>
   Maps basics <maps_basics>
   Masks basics <masks_basics>
   Symmetries basics <symmetries_basics>
   Wedge list basics <wedgelist_basics>

==========
User guide
==========

The user guide explains the concepts behind cryoCAT and how its modules are meant to be used together. Where the API reference lists what each function does, the user guide shows how the pieces fit: which conventions the code follows, what the main data structures hold, and how a typical step looks in practice.

The topics follow the order in which data usually moves through an analysis:

* :doc:`preprocessing_basics` covers the ``tiltstack`` and ``mdoc`` modules, for operations on tilt series and their metadata before reconstruction.
* :doc:`geometric_conventions` describes the coordinate systems, rotation conventions and Euler angles used throughout cryoCAT. It is worth reading first, since every other module relies on it.
* :doc:`motl_basics` and :doc:`motl_advanced` introduce the motive list (motl), the particle table at the center of cryoCAT: how it is read and written in the supported formats, what its columns mean, and how particles are selected, cleaned and transformed.
* :doc:`maps_basics` and :doc:`masks_basics` cover reading, writing and processing volumes, and creating masks for subtomogram averaging and analysis.
* :doc:`symmetries_basics` explains how symmetry is handled, including splitting particles into their asymmetric subunits.
* :doc:`wedgelist_basics` describes the wedge lists that record the missing-wedge geometry of each tomogram.

All coordinates in cryoCAT are kept in voxels, and a pixel size is used only to report values in physical units. Each page notes where this matters.

The guide is still growing. If a topic you need is not covered yet, the API reference and the tutorials are the best places to look.