# Changelog

All notable changes to cryoCAT are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [1.0.0rc] — unreleased

> **Backward compatibility is intentionally broken.**
> This release completes a large-scale refactor started on the `Refactoring`
> branch.  Code that ran against any previous release will require updates.
> The migration notes in each section below describe the exact substitutions.

---

### Breaking changes

#### `cryocat.analysis.structure` renamed to `cryocat.analysis.assembly`

Every import of the old module path must be updated:

```python
# before
from cryocat.analysis import structure
from cryocat.analysis.structure import NPC, CnComplex

# after
from cryocat.analysis import assembly
from cryocat.analysis.assembly import NPC, CnComplex
```

The file `pstructure.py` (GUI Surfaces page) is likewise renamed
`passembly.py`; string-based registry entries and test imports are updated
throughout.

#### `ribana` module removed

`cryocat.analysis.ribana` is deleted.  All functionality migrated to
`cryocat.analysis.nnana` and the new `nnana.Chain` class.

| Old | New |
|---|---|
| `ribana.trace_chains(entry, exit, ...)` | `nnana.trace_chains(entry, exit, ...)` |
| `ribana.add_occupancy(motl)` | `Chain(motl).get_occupancy()` |
| `ribana.add_traced_info(traced, input)` | `Chain(traced).add_traced_info(input)` |
| `ribana.get_polysome_stats(entry, exit)` | `Chain.from_motls(entry, exit).get_chain_stats()` |
| `ribana.get_class_polysome_occupancies_mp(motl)` | `Chain(motl).get_class_chain_occupancies(mode='mp')` |
| `ribana.assign_class(...)` | `nnana.assign_class_by_nn(...)` |
| `ribana.get_feature_nn(a, b)` | `nnana.find_nn_indices(a.get_coordinates(), b.get_coordinates(), k=1)` |

#### `surfsamp.py` and `quadric.py` removed

Both legacy surface modules are deleted and replaced by the new
`cryocat.core.surface` module.  The `PleomorphicSurface` and `ParametricSurface`
classes now live in `cryocat.analysis.assembly`.

#### `wedgeutils` moved to `cryocat.core.cryowedge`

```python
# before
from cryocat.utils import wedgeutils

# after
from cryocat.core import cryowedge
```

#### `structure.Icosahedron` replaced by `PolyhedralComplex`

| Old | New |
|---|---|
| `structure.Icosahedron(motl)` | `PolyhedralComplex(motl, "I")` |
| `ico.get_icosahedral_feature_vectors(mode, radius)` | `pc.feature_vectors(mode, radius)` |
| `ico.recover_icosahedral_features(...)` | `PolyhedralComplex.recover_features(...)` |

#### `visplot` is now Plotly-only

All Matplotlib-based functions in `cryocat.analysis.visplot` have been
removed.  Every call site must switch to the Plotly equivalents.

#### Renamed parameters (global)

| Old parameter | New parameter | Affected modules |
|---|---|---|
| `output_file` | `output_path` | `cryomap`, `cryomask`, `tiltstack`, and others |
| `map_name` | `input_map` | `cryomap` |

#### `pana` — template-list CSV column renames and path changes

| Old column | New column |
|---|---|
| `Tomo map` | `Target map` |
| `Tomo created` | `Target created` |

Map filename columns must now include the file extension (`.em`, `.mrc`).
`create_summary_pdf` is removed; use `create_summary_html` instead.
Parameters `wedge_path` and `angle_list_path` removed from batch functions
(paths are now resolved automatically).

#### `RelionMotl` extra-columns interface

The `add_object_id` and `add_subunit_id` arguments to `write_out` are
removed.  Use the new `extra_columns` keyword instead.

---

### Added

#### New module: `cryocat.core.surface`

Complete mesh and point-cloud processing stack:

- `Mesh` — triangle mesh with curvature, smoothing, remeshing, shape-index
- `OrientedPointCloud` — oriented point cloud with normal estimation
- `DiscreteSurface` — base class for both
- Backends: Open3D and PyVista (new dependencies)

#### New module: `cryocat.utils.symmetry`

Point-group rotation library for all symmetries used in cryo-ET:

- `CyclicGroup(n)`, `DihedralGroup(n)`, `TetrahedralGroup()`,
  `OctahedralGroup()`, `IcosahedralGroup()`
- `get_symmetry_rotations(symmetry)` — `(M, 3, 3)` rotation matrix array
- `get_symmetry_angles(symmetry)` — ZXZ Euler angles, optional CSV export
- `compute_conjugation_matrix(in_axis, out_axis)` — axis reorientation

#### New module: `cryocat.utils.imageutils`

Image-level utilities extracted from `cryomap` and `cryomask`:
slice extraction, normalization, profile sampling, volume overlap masking.

#### New module: `cryocat.analysis.clustering`

Standalone clustering utilities (DBSCAN-based, topology-aware) factored out
from `nnana` and `structure`.

#### New type system: `cryocat._types`

`EulerAngles`, `Symmetry`, `RelionVersion`, `ThresholdType`, `PathOrStr`, and
further domain types used for annotation throughout the library.

#### `assembly` — new and extended classes

- **`SymmetricComplex`** — abstract base class for all symmetric complexes;
  centralises motl loading, pixel-size handling, and the `@gui_exposed`
  decorator infrastructure.
- **`CnComplex`** — cyclic-symmetry complex with:
  - `assign_subunit_order()`, `get_subunit_stats()`, `ring_summary()`
  - `get_pore_frame_coords()` — in-plane coordinates with phase alignment
  - `ring_ellipticity()`, `diameter()`
- **`NPC`** — NPC-specific subclass of `CnComplex` with:
  - `classify_pores()` — vacancy / shear / regular classification with
    coherence correlation and argmax-agreement metrics
  - `merge_rings()`, `merge_subunits()`, `unify_nn_orientations()`
- **`PolyhedralComplex`** — T/O/I symmetric particles:
  - `assign_subunit_order()`, `feature_vectors(mode)`, `expand()`,
    `recover_features()`, `angular_dissimilarity()`
- **`PleomorphicSurface`** — membrane / pleomorphic surface complex with
  block-based lattice assignment and ring recovery
- **`BlockDefinition`** / lattice evaluation utilities
- **`Chain`** — generic traced-chain analysis (polysomes, chromatin, etc.)

#### `nnana` — three-layer architecture

Layer 1 pure-array helpers, Layer 2 `NearestNeighbors` class, Layer 3
motl-accepting wrappers (all preserved, backward-compatible).

New in `NearestNeighbors`:
- `add_motl_columns(column_names, *, sides, add_to_df)` — pull motl columns
  into the result dataframe
- `exclude_column_name` parameter — filter out same-value candidates
- `write_out` / `load` — CSV round-trip persistence

#### `pana` — three-layer architecture and new helpers

Layer 1 pure-compute, Layer 2 single-case orchestrators callable without a
template list, Layer 3 batch wrappers.

New: `gather_case_results`, `build_params_record`, `params_to_template_row`,
`create_summary_html_from_folder`, `params.csv` provenance recording.

#### `geom` — Platonic solid classes

`Tetrahedron`, `Octahedron`, `Cube`, `Icosahedron`, `Dodecahedron` (all
subclass `Polyhedron`) with `.vertices`, `.edges`, `.faces` and
`from_vectors(v1, v2)` classmethod.

#### `geom` — extended angular scoring

Full T/O/I/D symmetry support in `angular_distance`, `angular_score`,
`compare_rotations`, and `generate_angles`.  `angular_score_for_c_symmetry`
and improved `process_tomo_twist` handling of C-symmetry edge cases.

#### `tango` / `TwistDescriptor` — T/O/I symmetry

`TwistDescriptor` gains a `kind` parameter (`"cube"` / `"octahedron"` for O;
`"icosahedron"` / `"dodecahedron"` for I) and full non-cyclic angular-score
support.  Tango GUI updated with the new symmetry dropdown.

#### `tmana` — extended template matching utilities

- `extract_peak_orientations()` — Euler angles from TM peak coordinates
- `scores_extract_particles_around_positions()` — max-score peaks near particles
- Symmetry-aware random orientation assignment during peak extraction
- `motl_converter_kwargs()` helper for output type dispatch

#### `RelionMotl` — Relion 5.1 and robustness

- Relion 5.1 (`RelionMotlv5_1`) support
- Robust NaN / missing-column / unparsable-value handling
- `assign_cc_extra_columns()` shared between v4 and v5 for `cc`+column mapping
- `rlnOpticsGroup` synced from the optics table on write
- `add_column_from_star()` — transfer columns across STAR files by proximity

#### `cryomotl` — `output_kwargs` and write improvements

`Motl.write_out`, `CryoMask.write_out`, `TiltStack.write_out` gain
`output_kwargs` for forwarding format-specific options.  `check_df_type` now
accepts `kwargs` for subclass constructors.

#### `ioutils` — new types and angle loading

`euler_angles_load` (replaces `rot_angles_load`) with corrected angle-order
handling; `RelionVersion`, `ThresholdType` types; marker-file I/O (`load_cmm`,
`write_cmm`).

#### GUI — complete component-based redesign

The application shell and all pages have been rewritten:

- **Architecture**: pool / event / datapool / formgen / discovery / registry
  system replaces ad-hoc callbacks
- **New components** (40+): `filebrowser`, `consoleui`, `tablegrid`,
  `tableeditor`, `tablecluster`, `tableplot`, `surfaceview`, `volumeview`,
  `tomoview`, `orientpicker`, `ploteditor`, `poolpicker`, `savedialog`,
  `motlinput`, `motlio`, `anglesbuilder`, and more
- **Console subsystem** (`cryocat.app.console`): parse / execute / help / vars
- **New pages**: Surfaces, Complexes, Console; Tango merged into the suite
- **Graph editor**: `ploteditor` with live Plotly restyle
- **Performance**: lazy page build, server-side row model for the AG Grid,
  pool-aware undo

#### New tutorials and documentation

Surface sampling, curvature computation, mesh–particle intersections,
symmetry basics, and geometric conventions tutorials.  All tutorials updated
to reflect new APIs.

---

### Changed

- `cryomap.get_metadata` added; duplicate read function removed
- `cryomap.sample_line_profiles` — fixed x/z axis swap
- `memthick` — refactored with asymmetry analysis restored; x/z axis swap fixed
- `classutils` — extended with new utilities
- `starfileio` — improved parsing robustness
- `visplot` — new color palettes; mesh and oriented-point-cloud visualization;
  Matplotlib dependency fully removed
- `cryowedge` — moved from `utils/wedgeutils.py` to `core/cryowedge.py`;
  per-tomogram wedge mask building refactored

---

### Removed

- `cryocat.analysis.ribana` — see migration table above
- `cryocat.analysis.surfsamp` — replaced by `cryocat.core.surface`
- `cryocat.analysis.quadric` — replaced by `cryocat.core.surface`
- `cryocat.utils.wedgeutils` — moved to `cryocat.core.cryowedge`
- `cryocat.analysis.structure` — renamed to `cryocat.analysis.assembly`
- `visplot` Matplotlib functions — Plotly equivalents cover all use cases
- `pana.create_summary_pdf` — replaced by `create_summary_html`
- `pana.build_summary_figure` — internal dead code
- `RelionMotl.write_out` parameters `add_object_id`, `add_subunit_id`

---

### Fixed

- **C-symmetry angular score** — particles now aligned on the same z-axis
  before computing the remaining spin; `process_tomo_twist` phi angle derived
  from the particle rotation (fixes edge cases at θ = 0°, 180°, outside 0–180°)
- **`geom.angular_distance`** — Cn symmetry and in-plane distance bugs fixed;
  `inplane_distance` corrected consistently
- **`symmetry_volume`** — fixed; extended for T/O/I canonical frames
- **`RelionMotl` class=0 handling** — fixed zero-value detection on STAR load
- **`RelionMotl` rlnOpticsGroup** — synced from optics table (was always 0.0)
- **`NearestNeighbors.get_nn_subset`** — `feature_id` attribute bug; scalar
  argument wrapping; `motls` propagation to subset
- **`pana._angle_development_figure`** — x-axis was hardcoded to [0, 359];
  now spans [0, n_rows − 1]
- **`cryomask.generate_mask`** — icosahedral mask support added
- **Tango / NN GUI** — loading and display bugs fixed

---

### Dependencies

New required packages:
- `open3d` — mesh processing in `cryocat.core.surface`
- `pyvista` — mesh I/O and visualization in `cryocat.core.surface`
- `ipywidgets` —  in `cryocat.analysis.visplot` for motl browsing
- `IPython` — in `cryocat.analysis.visplot` for motl browsing

---

### Tests

Test suite expanded from ~350 to 725+ tests across all modules.
New test files: `test_assembly.py`, `test_clustering.py`, `test_cryowedge.py`,
`test_geom.py`, `test_imageutils.py`, `test_memthick.py`, `test_nnana.py`,
`test_surface.py`, `test_symmetry.py`, `test_tmana.py`; full `tests/app/`
suite (architecture, design rules, registry, pool, form, grid).

---

[1.0.0rc]: https://github.com/turonova/cryoCAT/compare/main...Refactoring
