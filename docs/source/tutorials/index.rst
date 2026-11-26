.. _tutorials:

.. currentmodule:: cryocat

.. toctree::
   :maxdepth: 2
   :hidden:

   Peak analysis <peak_analysis/peak_analysis>
   Surface sampling <surface_sampling/oversample_capsid>
   Compute curvatures <compute_curvatures/compute_curvatures>
   Mesh-points intersections <mesh_points_intersections/mesh_points_intersections>
   Lamellae analysis <lamellae_analysis/lamellae_analysis>
   Nearest neighbor analysis <nearest_neighbor_analysis/nearest_neighbor_analysis>
   Membrane thickness: preprocessing <membrane_thickness/memthick_tomo_preprocessing>
   Membrane thickness: run <membrane_thickness/memthick_run>
   Membrane thickness: analysis <membrane_thickness/memthick_analysis>
   Structure of tutorial video for TANGO GUI <tango_tutorials/TANGO_tutorial_video_structure>
   TANGO on nuclear pore complexes <tango_tutorials/TANGO_NPC>
   TANGO on virus-like particles <tango_tutorials/TANGO_VLP>

=========
Tutorials
=========

The tutorials are worked examples that take one analysis from start to finish. Each one starts from the input data and shows every step needed to get to a result, so it can be followed as written and then adapted to your own data. For the concepts behind the steps, see the :ref:`user_guide`.

The tutorials are grouped by the kind of question they answer:

* **Particle picking:** :doc:`peak_analysis/peak_analysis` shows how to inspect and analyze the peaks of a template matching scores map.
* **Surfaces and particles:** :doc:`surface_sampling/oversample_capsid` oversamples a surface to generate new candidate positions on it, :doc:`compute_curvatures/compute_curvatures` computes curvature on membrane meshes, and :doc:`mesh_points_intersections/mesh_points_intersections` relates particles to a mesh by intersecting rays with it.
* **Sample geometry:** :doc:`lamellae_analysis/lamellae_analysis` describes the lamella that a tomogram was recorded from.
* **Spatial organization:** :doc:`nearest_neighbor_analysis/nearest_neighbor_analysis` measures distances and orientations between neighboring particles.
* **Membrane thickness:** three tutorials, meant to be followed in order: :doc:`membrane_thickness/memthick_tomo_preprocessing` prepares the tomograms, :doc:`membrane_thickness/memthick_run` runs the measurement, and :doc:`membrane_thickness/memthick_analysis` analyzes the results.
* **TANGO:** analyses in the TANGO graphical interface. :doc:`tango_tutorials/TANGO_tutorial_video_structure` outlines the tutorial video, and :doc:`tango_tutorials/TANGO_NPC` and :doc:`tango_tutorials/TANGO_VLP` apply TANGO to nuclear pore complexes and virus-like particles.

More tutorials will be added over time.