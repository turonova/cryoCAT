"""Peak analysis (pana) — batch orchestration for template-matching peak analysis.

Three-layer design
------------------
Layer 1  Pure-compute functions (no I/O): ``mask_stats``, ``sharp_mask_overlap``,
         ``find_matching_overlap_row``, ``dist_map_stats``, ``peak_stats_and_profiles``,
         ``peak_shapes``, ``shape_stats``, ``filter_template_df``,
         ``build_summary_figure``, ``_run_analysis_args_from_row``.

Layer 2  Single-case orchestrators that run one (tomogram, template, angles) triple
         and optionally write their own artifacts: ``analyze_rotations``,
         ``compute_distance_map``, ``compute_peak_stats``, ``visualize_results``,
         ``run_single_case``.

Layer 3  Batch wrappers that iterate over CSV/folder collections and delegate to
         Layer 1/2: ``get_mask_stats``, ``compute_sharp_mask_overlap``,
         ``check_existing_tight_mask_values``, ``compute_dist_maps_voxels``,
         ``compute_center_peak_stats_and_profiles``, ``compute_peak_shapes``,
         ``get_shape_stats``, ``get_indices``, ``create_summary_pdf``,
         ``run_analysis``, ``run_angle_analysis``.
"""

from __future__ import annotations

import datetime
import json
import numpy as np
import pandas as pd
from typing import Literal
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from cryocat.core import cryomap
from cryocat.utils import geom
from cryocat.utils import ioutils
from cryocat.core import cryomotl
from cryocat.analysis import tmana
from cryocat.analysis import visplot
from cryocat.utils import wedgeutils
from cryocat.utils import imageutils
from cryocat.utils.classutils import gui_exposed
from scipy.spatial.transform import Rotation as srot
import re
from pathlib import Path
from cryocat._types import DataSource, EulerAngles, ListLike, MapSource, PathOrStr, Symmetry, TripletLike
from cryocat.core.cryomotl import MotlSource
from cryocat.core import cryomask
import os
from skimage import measure
from skimage import morphology


def create_structure_path(folder_path: PathOrStr, structure_name: str) -> str:
    """Put together a path for the structure folder by combining a base folder path
    and the name of the structure.

    Parameters
    ----------
    folder_path : PathOrStr
        The base directory path where the structure folder should be created.
        It should include a trailing slash if needed, otherwise the function
        will not insert a separator between `folder_path` and `structure_name`.
    structure_name : str
        The name of the structure of interest.

    Returns
    -------
    structure_folder : str
        Full path to the structure folder.
    """

    structure_folder = folder_path + structure_name + "/"
    return structure_folder


def create_em_path(folder_path: PathOrStr, structure_name: str, em_filename: str) -> str:
    """Constructs the full path to an `.em` file within a specific structure folder.

    Parameters
    ----------
    folder_path : PathOrStr
        The base directory path.
    structure_name : str
        The name of the structure, used to create a subdirectory under `folder_path`.
    em_filename : str
        The name of the `.em` file, without the file extension.

    Returns
    -------
    em_path : str
        The full path to the `.em` file, including the `.em` extension.
    """

    structure_folder_path = create_structure_path(folder_path, structure_name)
    em_path = structure_folder_path + em_filename + ".em"
    return em_path


def create_subtomo_name(structure_name: str, motl_name: str, tomo_id: str, boxsize: int) -> str:
    """Generate a standardized filename for a subtomogram.
    The generated file name is
    :code:`subtomo_<structure_name>_m<motl_name>_t<tomo_id>_s<boxsize>.em`

    Parameters
    ----------
    structure_name : str
        Name of the structure.
    motl_name : str
        Name of the motive list file containing particle information.
    tomo_id : str
        Tomogram id/number from which the subtomogram is extracted.
    boxsize : int
        Size of the subtomogram box in voxels.

    Returns
    -------
    subtomo_name : str
        The constructed filename for the subtomogram.
    """

    subtomo_name = "subtomo_" + structure_name + "_m" + motl_name + "_t" + tomo_id + "_s" + str(boxsize) + ".em"

    return subtomo_name


def create_tomo_name(
    folder_path: PathOrStr,
    tomo: str,
) -> str:
    """Generate a full file path for a tomogram with an .mrc extension.

    Parameters
    ----------
    folder_path : PathOrStr
        Path to the directory containing the tomogram.
    tomo : str
        Base name of the tomogram file (without extension).

    Returns
    -------
    tomo_name : str
        Full path to the tomogram file with .mrc extension.
    """

    tomo_name = folder_path + tomo + ".mrc"
    return tomo_name


def create_wedge_names(
    wedge_path: PathOrStr, tomo_number: int, boxsize: int, binning: int, filter: int | None = None
) -> tuple[str, str]:
    """Generate filenames for tomogram and template wedge masks with filtering info.

    If no filter size is provided, it defaults to half of the box size.

    Parameters
    ----------
    wedge_path : PathOrStr
        Directory path where the wedge files will be stored.
    tomo_number : int
        Number of the tomogram.
    boxsize : int
        Size of the subtomogram box in voxels.
    binning : int
        Binning level applied to the tomogram.
    filter : int, optional
        Size of the filter applied during processing.

    Returns
    -------
    tomo_wedge : str
        Filename for the filtered tomogram wedge mask.
    tmpl_wedge : str
        Filename for the filtered template wedge mask.
    """

    if filter is None:
        filter = boxsize // 2

    file_ending = str(boxsize) + "_t" + str(tomo_number) + "_b" + str(binning) + "_f" + str(filter) + ".em"
    tomo_wedge = wedge_path + "tile_filt_" + file_ending
    tmpl_wedge = wedge_path + "tmpl_filt_" + file_ending

    return tomo_wedge, tmpl_wedge


def create_output_base_name(tmpl_index: int) -> str:
    """Generates the base name for peak analysis output folders / files.

    Includes the index of the row analyzed from the template list csv.

    Parameters
    ----------
    tmpl_index : int
        The index of the row analyzed in the templatel list csv.

    Returns
    -------
    output_base : str
        Output file based name. It should be "id_<tmpl_index>".
    """

    output_base = "id_" + str(tmpl_index)
    return output_base


def create_output_folder_name(tmpl_index: int) -> str:
    """Generates the name of the folder (not the full path) where the peak analysis
    results will be stored, given the index of the row from the template list csv.

    Parameters
    ----------
    tmpl_index : int
        The index of the row analyzed in the templatel list csv.

    Returns
    -------
    str
        The name of result folder. It should be 'id_<tmpl_index>_results'.
    """

    return create_output_base_name(tmpl_index) + "_results"


def create_output_folder_path(folder_path: PathOrStr, structure_name: str, folder_spec: int | str) -> str:
    """Constructs the full path of the output folder.

    Parameters
    ----------
    folder_path : PathOrStr
        The path to the peak analysis base folder.
    structure_name : str
        The name of the structure.
    folder_spec : int or str
        When an int (a row index from the template list CSV), the folder name
        is ``id_<folder_spec>_results``.  When a string, it is used directly
        as the folder name.

    Returns
    -------
    output_path : str
        Full path to the output folder, with a trailing slash.
        Pattern: ``<folder_path>/<structure_name>/id_<folder_spec>_results/``
        when `folder_spec` is an int, or ``<folder_path>/<structure_name>/<folder_spec>/``
        otherwise.
    """

    if isinstance(folder_spec, int):
        output_path = create_structure_path(folder_path, structure_name) + create_output_folder_name(folder_spec) + "/"

    else:
        output_path = create_structure_path(folder_path, structure_name) + folder_spec + "/"

    return output_path


def filter_template_df(df: pd.DataFrame, conditions: dict, sort_by: str | None = None) -> pd.Index:
    """Filter an already-loaded template DataFrame and return matching indices.

    Parameters
    ----------
    df : pandas.DataFrame
        Template list DataFrame (indexed by row id).
    conditions : dict
        Keys are column names; values are the required equality values.
        All conditions must match (logical AND).
    sort_by : str, optional
        Column name to sort the filtered result by ascending value.

    Returns
    -------
    pandas.Index
        Index of rows satisfying all conditions, optionally sorted.
    """
    for key, value in conditions.items():
        df = df.loc[df[key] == value, :]
    if sort_by is not None:
        df = df.sort_values(by=sort_by, ascending=True)
    return df.index


def get_indices(template_list: PathOrStr, conditions: dict, sort_by: str | None = None) -> pd.Index:
    """Get the indices of a filtered and optionally sorted template list csv file.

    Parameters
    ----------
    template_list : PathOrStr
        Path to the template list csv file.
    conditions : dict
        Dictionary where keys are template list column names and values are the values to
        filter by. Only rows matching all conditions are retained.
    sort_by : str, optional
        Column name to sort the filtered DataFrame by. If None, no sorting
        is applied.

    Returns
    -------
    pandas.Index
        Index of the filtered (and optionally sorted) rows in the template list DataFrame.
    """
    return filter_template_df(pd.read_csv(template_list, index_col=0), conditions, sort_by=sort_by)


def cut_the_best_subtomo(
    tomogram: PathOrStr, motl_path: PathOrStr, subtomo_shape: TripletLike, output_path: PathOrStr | None
) -> tuple:
    """Extract the highest-scoring subtomogram from a tomogram.

    Loads a tomogram and its corresponding particle motive list, identifies the entry
    with the highest score, extracts the subtomogram at that position, and applies
    the stored shift correction.  Rotation is not applied.
    Optionally writes the result to a file.

    Parameters
    ----------
    tomogram : PathOrStr
        Path to the tomogram file to extract from.
    motl_path : PathOrStr
        Path to the motive list with extracted particle information (.csv or
        compatible format).
    subtomo_shape : TripletLike
        Shape of the subtomogram to extract, in (x, y, z) order.  A
        ``TripletLike`` is a scalar or a 3-element sequence.
    output_path : PathOrStr or None
        Path to save the extracted subtomogram. If None, the file is not saved.

    Returns
    -------
    subvolume_sh : numpy.ndarray
        The extracted and shifted subtomogram.
    angles : numpy.ndarray
        The Euler angles (phi, theta, psi) rotation associated with the best subtomogram.
    """

    tomo = cryomap.read(tomogram)
    m = cryomotl.Motl.load(motl_path)
    m.update_coordinates()

    max_idx = m.df["score"].idxmax()  # get the dataframe idx where ccc is max

    coord = m.df.loc[m.df.index[max_idx], ["x", "y", "z"]].to_numpy() - 1
    shifts = -m.df.loc[m.df.index[max_idx], ["shift_x", "shift_y", "shift_z"]].to_numpy()
    angles = m.df.loc[m.df.index[max_idx], ["phi", "theta", "psi"]].to_numpy()

    subvolume = cryomap.extract_subvolume(tomo, coord, subtomo_shape)
    subvolume_sh = cryomap.shift(subvolume, shifts)
    # subvolume_rot = cryomap.rotate(subvolume_sh,rotation_angles=angles)

    if output_path is not None:
        cryomap.write(subvolume_sh, output_path, data_type=np.single)

    return subvolume_sh, angles


@gui_exposed(label="Extract best subtomogram", category="Peak Analysis", output="map", hide=("output_path",))
def extract_best_subtomogram(
    tomogram: MapSource,
    motl: MotlSource,
    box_size: TripletLike | int,
    *,
    output_path: PathOrStr | None = None,
) -> dict:
    """Extract the highest-scoring subtomogram from a tomogram.

    Parameters
    ----------
    tomogram : MapSource
        Target tomogram to extract from.  A ``MapSource`` is an ndarray or a
        path to a map file (.mrc, .em, …).
    motl : MotlSource
        Motive list with particle coordinates and scores.  A ``MotlSource`` is
        a :class:`~cryocat.core.cryomotl.Motl`, a :class:`pandas.DataFrame`,
        or a path to a compatible file.
    box_size : TripletLike or int
        Box size of the extracted subtomogram.  A scalar is treated as cubic.
    output_path : PathOrStr, optional
        Path to write the extracted subtomogram.  Skipped when *None*.

    Returns
    -------
    dict
        ``subtomogram`` - extracted subtomogram ndarray;
        ``rotation`` - Euler angles ``(phi, theta, psi)`` of the best particle;
        ``output_path`` - path the subtomogram was written to, or *None*.
    """
    box_size = geom.as_triplet(box_size)
    subtomogram, rotation = cut_the_best_subtomo(tomogram, motl, box_size, output_path)
    return {"subtomogram": subtomogram, "rotation": rotation, "output_path": output_path}


def create_subtomograms_for_tm(template_list: PathOrStr, parent_folder_path: PathOrStr) -> pd.DataFrame:
    """Generates subtomograms with highest ccc score from tomograms for each
    entry in template list csv.

    Updates the template list with orientation and status info, and saves
    the updated list.

    Parameters
    ----------
    template_list : PathOrStr
        Path to the template list file with motl path info.
    parent_folder_path : PathOrStr
        Path to the base directory for peak analysis.

    Returns
    -------
    temp_df : pandas.DataFrame
        The updated DataFrame containing subtomogram metadata, including
        creation status, orientation angles, and filenames.
    """

    temp_df = pd.read_csv(template_list, index_col=0)
    target_col = "Target map" if "Target map" in temp_df.columns else "Tomo map"
    created_col = "Target created" if "Target created" in temp_df.columns else "Tomo created"
    temp_df[target_col] = temp_df[target_col].astype(object)
    unique_entries = temp_df.groupby(["Structure", "Motl", "Tomogram", "Boxsize"]).groups
    entry_indices = list(unique_entries.values())

    for i, entry in enumerate(unique_entries):
        if np.all(temp_df.loc[entry_indices[i], created_col]):
            continue  # skip if subtomograms have been created
        else:
            motl = create_em_path(parent_folder_path, entry[0], entry[1])
            boxsize = entry[3]

            # find out which entries from template list have not had subtomos created
            not_created = temp_df.loc[temp_df[created_col] == False, created_col].index
            create_idx = np.intersect1d(not_created, entry_indices[i])

            # cut the subtomos with best ccc scores and save them
            subtomo_name = create_subtomo_name(entry[0], entry[1], entry[2], boxsize)
            _, subtomo_rotation = cut_the_best_subtomo(
                create_tomo_name(parent_folder_path, entry[2]),
                motl,
                (boxsize, boxsize, boxsize),
                create_structure_path(parent_folder_path, entry[0]) + subtomo_name,
            )

            # updates the template list df after creating best subtomos
            temp_df.loc[create_idx, ["Phi", "Theta", "Psi"]] = np.tile(subtomo_rotation, (create_idx.shape[0], 1))
            temp_df.loc[create_idx, created_col] = True
            temp_df.loc[create_idx, target_col] = subtomo_name[0:-3]

    temp_df.to_csv(template_list)

    return temp_df


# ── Layer 1: pure-compute functions ─────────────────────────────────────────


def mask_stats(soft_mask: np.ndarray, sharp_mask: np.ndarray) -> dict:
    """Compute volume statistics for a soft/sharp mask pair.

    Parameters
    ----------
    soft_mask : numpy.ndarray
        Soft (smooth-boundary) mask volume.
    sharp_mask : numpy.ndarray
        Tight (binary) mask volume.

    Returns
    -------
    dict
        ``voxels_soft`` - nonzero voxels in *soft_mask* above 0.5;
        ``voxels_sharp`` - nonzero voxels in *sharp_mask*;
        ``bbox`` - bounding-box dimensions (x, y, z) of *sharp_mask*;
        ``solidity`` - solidity of *sharp_mask*.
    """
    solidity = cryomask.compute_solidity(sharp_mask)
    voxels, bbox = imageutils.mask_voxel_count_and_bbox(sharp_mask)
    voxels_soft, _ = imageutils.mask_voxel_count_and_bbox(soft_mask, threshold=0.5)
    return {"voxels_soft": voxels_soft, "voxels_sharp": voxels, "bbox": bbox, "solidity": solidity}


def sharp_mask_overlap(mask: np.ndarray, rotations: list) -> np.ndarray:
    """Compute the overlap between a mask and its rotated versions.

    Parameters
    ----------
    mask : numpy.ndarray
        Binary or near-binary mask volume.
    rotations : sequence of scipy.spatial.transform.Rotation
        Per-rotation objects to apply to *mask*.

    Returns
    -------
    numpy.ndarray of int
        Voxel count of the intersection ``mask_rotated * mask`` for each rotation.
    """
    voxel_count = []
    for rot in rotations:
        mask_rot = cryomap.rotate(mask, rotation=rot, transpose_rotation=True)
        mask_rot = np.where(mask_rot > 0.1, 1.0, 0.0)
        voxel_count.append(np.count_nonzero(mask_rot * mask))
    return np.asarray(voxel_count)


@gui_exposed(
    label="Compute sharp mask overlap (single case)",
    category="Peak Analysis",
    output="dataframe",
)
def compute_sharp_mask_overlap_single(
    template_mask: MapSource,
    input_angles: DataSource,
    angles_order: str = "zxz",
) -> dict:
    """Compute how much the mask self-overlaps under each search angle.

    For every angle in *input_angles*, the mask is rotated by that angle and
    the voxel-count intersection with the original mask is computed via
    :func:`sharp_mask_overlap`.

    Parameters
    ----------
    template_mask : MapSource
        Binary or near-binary mask volume.  A ``MapSource`` is an ndarray or a
        path to a map file.
    input_angles : DataSource
        ``(N, 3)`` Euler-angle array or path to an angles file.  Each row is
        used as a rotation applied to the mask.  A ``DataSource`` is a path to
        a file or an ndarray.
    angles_order : str, default='zxz'
        Euler-angle convention of *input_angles*.

    Returns
    -------
    dict
        ``"overlap"`` – 1-D integer array of voxel-count overlaps, one per
        angle; ``"angles"`` – the ``(N, 3)`` Euler-angle array used.
    """
    mask_arr = cryomap.read(template_mask) if not isinstance(template_mask, np.ndarray) else template_mask
    angles_arr = (
        ioutils.rot_angles_load(input_angles, angles_order)
        if not isinstance(input_angles, np.ndarray)
        else input_angles
    )

    rotations = [srot.from_euler(angles_order, row, degrees=True) for row in angles_arr]
    overlap = sharp_mask_overlap(mask_arr, rotations)

    return {"overlap": overlap, "angles": angles_arr}


@gui_exposed(
    label="Compute shape stats (single case)",
    category="Peak Analysis",
    output="dataframe",
)
def compute_shape_stats_single(
    scores_map: MapSource,
    shape_type: str = "tight",
) -> dict:
    """Compute peak-shape statistics for a single scores map.

    Wraps :func:`peak_shapes` and returns the principal-axis dimensions of the
    peak ellipsoid under three threshold methods (triangle, Gaussian, half-peak).

    Parameters
    ----------
    scores_map : MapSource
        3-D CC scores volume.  A ``MapSource`` is an ndarray or a path to a
        map file.
    shape_type : str, default='tight'
        Informational label attached to the result (e.g. ``'tight'`` or
        ``'loose'``).

    Returns
    -------
    dict
        ``"tp_shape"``, ``"gp_shape"``, ``"hp_shape"`` – principal dimensions
        ``(x, y, z)`` of the peak ellipsoid under each threshold;
        ``"peak_value"`` – maximum CC score; ``"shape_type"`` – echoed label.
    """
    sc_arr = cryomap.read(scores_map) if not isinstance(scores_map, np.ndarray) else scores_map
    res = peak_shapes(sc_arr)
    return {
        "tp_shape": list(res["tp_shape"]),
        "gp_shape": list(res["gp_shape"]),
        "hp_shape": list(res["hp_shape"]),
        "peak_value": float(res["peak_value"]),
        "shape_type": shape_type,
    }


def find_matching_overlap_row(template_df: pd.DataFrame, i: int) -> pd.Index:
    """Return indices of done rows with the same tight mask and degrees as row *i*.

    Parameters
    ----------
    template_df : pandas.DataFrame
        Already-loaded template list (indexed by row id).
    i : int
        Row index whose ``Tight mask`` and ``Degrees`` values are used as criteria.

    Returns
    -------
    pandas.Index
        Indices of rows where ``Done`` is truthy and both ``Tight mask`` and
        ``Degrees`` match those of row *i*.
    """
    tm = template_df.at[i, "Tight mask"]
    deg = template_df.at[i, "Degrees"]
    match = template_df["Done"].astype(bool) & (template_df["Tight mask"] == tm) & (template_df["Degrees"] == deg)
    return template_df.index[match]


def dist_map_stats(
    dist_map: np.ndarray,
    peak_center: tuple[int, ...],
    degrees: float,
    is_all: bool = False,
    morph_footprint: tuple[int, ...] = (2, 2, 2),
) -> dict:
    """Compute morphological statistics for a distance map around a peak.

    Parameters
    ----------
    dist_map : numpy.ndarray
        Angular distance map volume.
    peak_center : tuple of int
        ``(z, y, x)`` voxel coordinates of the highest-CC peak.
    degrees : float
        Search-angle increment used to threshold the distance map.
    is_all : bool, default False
        When True the threshold is ``2 * degrees`` (for the combined
        ``dist_all`` map); otherwise ``degrees`` is used directly.
    morph_footprint : tuple of int, default (2, 2, 2)
        Structuring element size for morphological opening.

    Returns
    -------
    dict
        ``vc`` – voxel count of the peak component;
        ``solidity`` – solidity of the peak component;
        ``label`` – binary volume isolating the peak component;
        ``vco`` – voxel count after morphological opening;
        ``open_label`` – binary volume after opening;
        ``dim`` – bounding-box dimensions (x, y, z) of the opened component.
    """
    threshold = 2.0 * degrees if is_all else degrees

    dist_map = dist_map.copy()
    dist_map[peak_center[0], peak_center[1], peak_center[2]] = degrees
    dist_map = np.where(dist_map <= threshold, 1.0, 0.0)

    dist_label = measure.label(dist_map, connectivity=1)
    dist_props = pd.DataFrame(measure.regionprops_table(dist_label, properties=("label", "area", "solidity")))

    peak_label = dist_label[peak_center[0], peak_center[1], peak_center[2]]
    vc = dist_props.loc[dist_props["label"] == peak_label, "area"].values
    sol = dist_props.loc[dist_props["label"] == peak_label, "solidity"].values
    label_binary = np.where(dist_label == peak_label, 1.0, 0.0)

    open_label = morphology.binary_opening(label_binary, footprint=np.ones(morph_footprint), out=None)
    open_label = measure.label(open_label, connectivity=1)
    peak_label_open = open_label[peak_center[0], peak_center[1], peak_center[2]]
    open_label_binary = np.where(open_label == peak_label_open, 1.0, 0.0)
    vco = np.count_nonzero(open_label_binary)
    dim = cryomask.get_mass_dimensions(open_label_binary)

    return {
        "vc": vc,
        "solidity": sol,
        "label": label_binary,
        "vco": vco,
        "open_label": open_label_binary,
        "dim": dim,
    }


def peak_stats_and_profiles(scores_map: np.ndarray, peak_center: tuple[int, ...], peak_value: float) -> dict:
    """Compute line profiles and spherical-neighborhood statistics for a scores map peak.

    Parameters
    ----------
    scores_map : numpy.ndarray
        Raw 3-D cross-correlation scores volume.
    peak_center : tuple of int
        ``(z, y, x)`` voxel coordinates of the highest-CC peak.
    peak_value : float
        CC score at *peak_center*.

    Returns
    -------
    dict
        ``line_profiles`` – 2-D array of shape ``(max_dim, 3)`` (x/y/z profiles);
        ``drop_x``, ``drop_y``, ``drop_z`` – score drop from peak to neighbours;
        ``peak_x``, ``peak_y``, ``peak_z`` – peak voxel coordinates;
        ``mean_r``, ``median_r``, ``var_r`` for *r* = 1..5 – statistics over
        spheres of increasing radius centred on *peak_center*.
    """
    _, _, line_profiles = tmana.create_starting_parameters_1D(scores_map)

    stats = {"peak_value": peak_value, "line_profiles": line_profiles}

    for j, dim in enumerate(["x", "y", "z"]):
        drop = peak_value - (line_profiles[peak_center[j] - 1, j] + line_profiles[peak_center[j] + 1, j]) / 2.0
        stats["drop_" + dim] = drop
        stats["peak_" + dim] = peak_center[j]

    for r in range(1, 6):
        cc_mask = cryomask.spherical_mask(np.asarray(scores_map.shape), radius=r, center=peak_center)
        masked_map = scores_map[np.nonzero(scores_map * cc_mask)]
        stats[f"mean_{r}"] = np.mean(masked_map)
        stats[f"median_{r}"] = np.median(masked_map)
        stats[f"var_{r}"] = np.var(masked_map)

    return stats


def peak_shapes(scores_map: np.ndarray) -> dict:
    """Evaluate peak shapes using triangle, Gaussian, and hard thresholding.

    Parameters
    ----------
    scores_map : numpy.ndarray
        3-D cross-correlation scores volume.

    Returns
    -------
    dict
        ``tp_shape``, ``gp_shape``, ``hp_shape`` - principal dimensions (x, y, z)
        of the peak ellipsoid under each threshold method;
        ``peak_value`` - maximum CC score;
        ``t_map``, ``t_th_map``, ``t_surf`` - triangle-threshold outputs;
        ``g_map``, ``g_th_map``, ``g_surf`` - Gaussian-threshold outputs;
        ``h_map``, ``h_th_map``, ``h_surf`` - hard-threshold outputs.
    """
    t_map, tp_shape, peak_value, t_th_map, t_surf = tmana.evaluate_scores_map(
        scores_map, label_type="ellipsoid", threshold_type="triangle"
    )
    g_map, gp_shape, _, g_th_map, g_surf = tmana.evaluate_scores_map(
        scores_map, label_type="ellipsoid", threshold_type="gauss"
    )
    h_map, hp_shape, _, h_th_map, h_surf = tmana.evaluate_scores_map(
        scores_map, label_type="ellipsoid", threshold_type="hard"
    )
    return {
        "tp_shape": tp_shape,
        "gp_shape": gp_shape,
        "hp_shape": hp_shape,
        "peak_value": peak_value,
        "t_map": t_map,
        "t_th_map": t_th_map,
        "t_surf": t_surf,
        "g_map": g_map,
        "g_th_map": g_th_map,
        "g_surf": g_surf,
        "h_map": h_map,
        "h_th_map": h_th_map,
        "h_surf": h_surf,
    }


def shape_stats(sharp_mask: np.ndarray) -> pd.DataFrame:
    """Compute region properties for connected components of a sharp mask.

    Parameters
    ----------
    sharp_mask : numpy.ndarray
        Binary tight mask volume.

    Returns
    -------
    pandas.DataFrame
        One row per labeled region with columns: ``label``, ``area``,
        ``area_bbox``, ``area_convex``, ``equivalent_diameter_area``,
        ``euler_number``, ``feret_diameter_max``, ``inertia_tensor``,
        ``solidity``.
    """
    mask_label = measure.label(sharp_mask, connectivity=1)
    return pd.DataFrame(
        measure.regionprops_table(
            mask_label,
            properties=(
                "label",
                "area",
                "area_bbox",
                "area_convex",
                "equivalent_diameter_area",
                "euler_number",
                "feret_diameter_max",
                "inertia_tensor",
                "solidity",
            ),
        )
    )


# ── Layer 2: batch wrappers ───────────────────────────────────────────────────


def get_mask_stats(template_list: PathOrStr, indices: list[int], parent_folder_path: PathOrStr) -> None:
    """Compute and update mask statistics for specified rows in a template list.

    Loads info about soft and tight (sharp) masks to computes volume-related statistics,
    including nonzero voxel counts, mask element bounding box dimensions, and solidity.
    Updates the CSV with these values: specifically:

        - 'Voxels': Number of nonzero voxels in the soft mask.

        - 'Voxels TM': Number of nonzero voxels in the tight mask.

        - 'Dim x', 'Dim y', 'Dim z': Dimensions of the bounding box enclosing the tight mask.

        - 'Solidity': Solidity metric of the tight mask (volume / convex hull volume).

    Parameters
    ----------
    template_list : PathOrStr
        Path to the CSV template list file.
    indices : list of int
        List of row indices in the CSV to process.
    parent_folder_path : PathOrStr
        Base directory where folders for all structures are.
    """

    temp_df = pd.read_csv(template_list, index_col=0)

    for i in indices:
        structure_name = temp_df.at[i, "Structure"]
        soft_mask = cryomap.read(create_em_path(parent_folder_path, structure_name, temp_df.at[i, "Mask"]))
        sharp_mask = cryomap.read(create_em_path(parent_folder_path, structure_name, temp_df.at[i, "Tight mask"]))

        stats = mask_stats(soft_mask, sharp_mask)

        temp_df.at[i, "Voxels"] = stats["voxels_soft"]
        temp_df.at[i, "Voxels TM"] = stats["voxels_sharp"]
        temp_df.at[i, "Dim x"] = stats["bbox"][0]
        temp_df.at[i, "Dim y"] = stats["bbox"][1]
        temp_df.at[i, "Dim z"] = stats["bbox"][2]
        temp_df.at[i, "Solidity"] = stats["solidity"]

        temp_df.to_csv(template_list)  # save new results back to template list


def compute_sharp_mask_overlap(
    template_list: PathOrStr,
    indices: list[int],
    angle_list_path: PathOrStr,
    parent_folder_path: PathOrStr,
    angles_order: str = "zxz",
) -> None:
    """Compute the overlap between the original tight mask and rotated versions of it.

    For each template specified by its index in the template list, this function loads the
    corresponding tight mask and set of rotation angles, rotates the mask accordingly,
    computes the overlap (intersection) with the original mask, and writes the results
    to a CSV file.

    Parameters
    ----------
    template_list : PathOrStr
        Path to the CSV template list file.
    indices : list of int
        List of row indices in the CSV to process.
    angle_list_path : PathOrStr
        Path to the directory containing angle list files used for rotation.
    parent_folder_path : PathOrStr
        Base directory where folders for all structures are.
    angles_order : str, default='zxz'
        The rotation order used to interpret the Euler angles.
    """

    temp_df = pd.read_csv(template_list, index_col=0)

    # only proceed for rows that haven't been analyzed
    for i in indices:
        if not temp_df.at[i, "Done"]:
            continue

        structure_name = temp_df.at[i, "Structure"]
        mask_name = create_em_path(parent_folder_path, structure_name, temp_df.at[i, "Tight mask"])
        mask = cryomap.read(mask_name)
        angle_list = angle_list_path + temp_df.at[i, "Angles"]
        angles = ioutils.rot_angles_load(angle_list, angles_order)
        rotations = srot.from_euler("zxz", angles, degrees=True)

        voxel_count = sharp_mask_overlap(mask, rotations)

        output_base = create_output_base_name(i)
        output_folder = create_output_folder_path(parent_folder_path, structure_name, temp_df.at[i, "Output folder"])

        csv_name = output_folder + output_base + ".csv"
        info_df = pd.read_csv(csv_name, index_col=0)
        info_df["Tight mask overlap"] = voxel_count
        info_df.to_csv(csv_name)


def check_existing_tight_mask_values(
    template_list: PathOrStr,
    indices: list[int],
    parent_folder_path: PathOrStr,
    angle_list_path: PathOrStr,
    angles_order: str = "zxz",
) -> None:
    """Check and populate "Tight mask overlap" values for given rows in template list.

    This function verifies whether specified rows have "Tight mask overlap" values.
    If the values are missing, it attempts to find an analyzed row (with the same tight
    mask and degrees) from which the overlap values can be copied. If no such data is
    found, the function computes the overlap and saves it back to the output csvs.

    Parameters
    ----------
    template_list : PathOrStr
        Path to the CSV template list file.
    indices : list of int
        List of row indices in `template_list` to check.
    parent_folder_path : PathOrStr
        Base directory where structure folders are located.
    angle_list_path : PathOrStr
        Path to the directory containing rotation angle list files.
    angles_order : str, default='zxz'
        Rotation order to interpret the Euler angles when computing overlaps.
    """

    temp_df = pd.read_csv(template_list, index_col=0)

    for i in indices:
        if not temp_df.at[i, "Done"]:
            continue

        structure_name = temp_df.at[i, "Structure"]
        output_base = create_output_base_name(i)
        output_folder = create_output_folder_path(parent_folder_path, structure_name, temp_df.at[i, "Output folder"])

        rot_info = pd.read_csv(output_folder + output_base + ".csv", index_col=0)

        data_found = False

        # if the value does not exist, find and copy from other "same" analysis outputs
        if "Tight mask overlap" not in rot_info.columns:
            done_idx = find_matching_overlap_row(temp_df, i)

            for j in done_idx:
                csv_file = (
                    create_output_folder_path(
                        parent_folder_path, temp_df.at[j, "Structure"], temp_df.at[j, "Output folder"]
                    )
                    + create_output_base_name(j)
                    + ".csv"
                )
                diff_info = pd.read_csv(csv_file, index_col=0)
                if "Tight mask overlap" in diff_info.columns:
                    rot_info["Tight mask overlap"] = diff_info["Tight mask overlap"].values
                    data_found = True
                    rot_info.to_csv(output_folder + output_base + ".csv")
                    break
        else:
            data_found = True

        # if the value is not found in "same" analysis, computes it
        if not data_found:
            print(f"Computing sharp mask overlap for index {i}")
            compute_sharp_mask_overlap(
                template_list, [i], angle_list_path, parent_folder_path, angles_order=angles_order
            )


def compute_dist_maps_voxels(
    template_list: PathOrStr,
    indices: list[int],
    parent_folder_path: PathOrStr,
    morph_footprint: TripletLike = (2, 2, 2),
) -> None:
    """Compute a few morphology related measurements for areas with highest cc score.

    For each specified rows in the template list, this function processes angular distance
    maps to find patches (i.e. connected components) within the search angle and
    have highest cross correlation score. Morphological properties computed include
    voxel count, solidity, and bounding box dimensions. The results are stored back
    into the template list CSV file.

    Parameters
    ----------
    template_list : PathOrStr
        Path to the CSV template list file.
    indices : list of int
        Row indices in `template_list` to process.
    parent_folder_path : PathOrStr
        Base directory where structure folders are located.
    morph_footprint : TripletLike, default=(2, 2, 2)
        Size of the structuring element used for binary opening during morphological
        processing of labeled regions.  A ``TripletLike`` is a scalar or a
        3-element sequence.

    Notes
    -----
    For "dist_all", the threshold is set to '2.0 * degrees'; for the other maps,
      the threshold is 'degrees'.
        * degrees here is the search increment angle used.

        * the max angular distance (i.e. "dist_all") given angular combinations within
          the "degrees" value is just slightly above 2*degrees.

        * cc score decreases with increasing angular distance (given no symmetry),
          therefore only looking at places where the dist is no larger than search angles

    Labeled connected components are analyzed to extract:
        * Voxel count (VC) of components with highest cc score

        * Solidity

        * Morphologically opened voxel count (VCO)

        * Bounding box dimensions (O x, O y, O z)

    Labeled masks and morphologically opened masks are saved as `.em` files with `_label` and `_label_open` suffixes, respectively.

    """

    temp_df = pd.read_csv(template_list, index_col=0)

    for i in indices:
        if not temp_df.at[i, "Done"]:
            continue

        structure_name = temp_df.at[i, "Structure"]

        degrees = temp_df.at[i, "Degrees"]
        output_base = create_output_base_name(i)
        output_folder = create_output_folder_path(parent_folder_path, structure_name, temp_df.at[i, "Output folder"])

        # only focus on central part of scores map (size of a template or subtomo)
        scores_map = cryomap.read(output_folder + output_base + "_scores.em")
        cc_mask = cryomask.spherical_mask(np.asarray(scores_map.shape), radius=10)
        scores_map *= cc_mask

        # find coordinates of centers of highest cc scores (peaks)
        peak_center, _, _ = tmana.create_starting_parameters_2D(scores_map)

        dist_names = ["dist_all", "dist_normals", "dist_inplane"]

        for j, value in enumerate(dist_names):
            dist_map = cryomap.read(output_folder + output_base + "_angles_" + value + ".em")
            stats = dist_map_stats(dist_map, peak_center, degrees, is_all=(j == 0), morph_footprint=morph_footprint)

            temp_df.at[i, "VC " + value] = stats["vc"]
            temp_df.at[i, "Solidity " + value] = stats["solidity"]
            cryomap.write(
                stats["label"], output_folder + output_base + "_angles_" + value + "_label.em", data_type=np.single
            )
            temp_df.at[i, "VCO " + value] = stats["vco"]
            cryomap.write(
                stats["open_label"],
                output_folder + output_base + "_angles_" + value + "_label_open.em",
                data_type=np.single,
            )
            for d, dim in enumerate(["x", "y", "z"]):
                temp_df.at[i, "O " + value + " " + dim] = stats["dim"][d]

        temp_df.to_csv(template_list)  # to save what was finished in case of a crush


def compute_center_peak_stats_and_profiles(
    template_list: PathOrStr, indices: list[int], parent_folder_path: PathOrStr
) -> None:
    """Compute statistics and line profiles for the cc peaks in a score map.

    For each specified template index, this function:
    1. Identifies the peak location and value.

    2. Saves the 1D line profiles through the peak along x, y, and z axes. Saved as
        '<output_base>_peak_line_profiles.csv' in the output folder.

    3. Computes the drop in score from the peak to its immediate neighbors along
        each axis.

    4. Calculates mean, median, and variance of scores in small areas where the peaks
        are centered (spherical of radius from 1 to 5 px).

    5. Updates the template list CSV file with the computed statistics:
        * "Peak value"
        * "Drop x", "Drop y", "Drop z"
        * "Peak x", "Peak y", "Peak z"
        * "Mean r", "Median r", "Var r" for r = 1..5

    Parameters
    ----------
    template_list : PathOrStr
        Path to the CSV template list file.
    indices : list of int
        List of row indices in `template_list` to process.
    parent_folder_path : PathOrStr
        Base directory containing structure folders.
    """

    temp_df = pd.read_csv(template_list, index_col=0)

    for i in indices:
        if not temp_df.at[i, "Done"]:
            continue

        structure_name = temp_df.at[i, "Structure"]

        output_base = create_output_base_name(i)
        output_folder = create_output_folder_path(parent_folder_path, structure_name, temp_df.at[i, "Output folder"])
        scores_map = cryomap.read(output_folder + output_base + "_scores.em")

        # only look at central area in scores map
        cc_mask = cryomask.spherical_mask(np.asarray(scores_map.shape), radius=10)
        masked_map = scores_map * cc_mask

        # find the coords and values of highest cc values
        peak_center, peak_value, _ = tmana.create_starting_parameters_2D(masked_map)

        stats = peak_stats_and_profiles(scores_map, peak_center, peak_value)

        temp_df.at[i, "Peak value"] = stats["peak_value"]

        line_pd = pd.DataFrame(data=stats["line_profiles"], columns=["x", "y", "z"])
        line_pd.to_csv(output_folder + output_base + "_peak_line_profiles.csv")

        for dim in ["x", "y", "z"]:
            temp_df.at[i, "Drop " + dim] = stats["drop_" + dim]
            temp_df.at[i, "Peak " + dim] = stats["peak_" + dim]

        for r in range(1, 6):
            temp_df.at[i, "Mean " + str(r)] = stats[f"mean_{r}"]
            temp_df.at[i, "Median " + str(r)] = stats[f"median_{r}"]
            temp_df.at[i, "Var " + str(r)] = stats[f"var_{r}"]

        temp_df.to_csv(template_list)


def analyze_rotations(
    target_map: MapSource,
    template: MapSource,
    template_mask: MapSource,
    input_angles: DataSource,
    wedge_mask_target: MapSource | None = None,
    wedge_mask_tmpl: MapSource | None = None,
    output_path: PathOrStr | None = None,
    cc_radius: int = 3,
    angular_offset: EulerAngles | None = None,
    starting_angle: EulerAngles | None = None,
    cyclic_symmetry: Symmetry = 1,
    angles_order: str = "zxz",
) -> tuple:
    """Perform template matching against a fixed target map.

    Rotates ``template`` through ``input_angles`` and computes the normalized
    cross-correlation against the fixed ``target_map`` for each rotation.
    Wedge masks can be applied to account for missing-wedge artifacts, and a
    spherical correlation mask limits measurements to a central region.

    Parameters
    ----------
    target_map : MapSource
        Fixed volume against which the rotated template is matched.  A
        ``MapSource`` is an ndarray or a path to a map file (.mrc, .em, …).
    template : MapSource
        Reference template map (the rotated side).
    template_mask : MapSource
        Binary mask for the template.
    input_angles : DataSource
        Angle list for the rotation search.  A ``DataSource`` is a path to a
        file or an ndarray.
    wedge_mask_target : MapSource, optional
        Wedge mask for the target map.
    wedge_mask_tmpl : MapSource, optional
        Wedge mask for the template.
    output_path : PathOrStr, optional
        Base path for saving output CSV and EM maps.  ``None`` skips disk
        output.  A ``PathOrStr`` is a :class:`str` or :class:`pathlib.Path`.
    cc_radius : int, default=3
        Radius (in voxels) of the spherical mask applied to compute masked
        cross-correlation.
    angular_offset : EulerAngles, optional
        Euler angles (degrees) applied as an offset to all input angles before
        matching.  An ``EulerAngles`` is a ``(3,)`` triple or ``(N, 3)``
        ndarray.
    starting_angle : EulerAngles, optional
        Reference orientation of the template (Euler angles, degrees).
        Applied as a base rotation right-multiplied onto every input angle.
        Defaults to ``(0, 0, 0)``.
    cyclic_symmetry : Symmetry, default=1
        C symmetry of the structure.  A ``Symmetry`` is an int or a string
        like ``"C5"``.
    angles_order : str, default='zxz'
        Euler-angle convention used for rotations.

    Returns
    -------
    res_table : pandas.DataFrame
        Table containing per-rotation statistics, including:
        - ang_dist: Angular distance from starting_angle (degrees)
        - cone_dist: Cone angle difference (degrees)
        - inplane_dist: In-plane rotation difference (degrees)
        - common_voxels: Overlap between mask and rotated mask
        - ccc: Maximum cross-correlation coefficient
        - ccc_masked: Maximum masked CCC within cc_radius
        - z_score: Maximum z-score across the full cc map
        - z_score_masked: Maximum z-score within the spherical mask
    final_ccc_map : ndarray
        3D array of the maximum CCC values observed across all rotations.
    final_angles_map : ndarray
        3D array of 0-based rotation indices into the angle list, recording
        which rotation produced the highest CCC at each voxel.
        Voxels that were never updated retain their initial value of ``-1``.
    final_ccc_map_masked : ndarray
        Masked CCC map showing only the central area of final_ccc_map.

    Notes
    -----
    - If the target_map and template sizes differ, the smaller map is padded to match.
    - The function keeps track of the highest CCC per voxel across all rotations.
    """

    angles = geom.apply_starting_and_offset(
        ioutils.rot_angles_load(input_angles, angles_order),
        starting_angle,
        angular_offset,
        angles_order,
    )

    tomo = cryomap.read(target_map)
    tmpl = cryomap.read(template)
    mask = cryomap.read(template_mask)

    # pad the maps so they are the same size
    if np.any(tomo.shape < tmpl.shape):
        tomo = cryomap.pad(tomo, tmpl.shape)
        output_size = tmpl.shape
    elif np.any(tomo.shape > tmpl.shape):
        tmpl = cryomap.pad(tmpl, tomo.shape)
        mask = cryomap.pad(mask, tomo.shape)
        output_size = tomo.shape
    else:
        output_size = tomo.shape

    # a small central area where the ccc is relevant
    cc_mask = cryomask.spherical_mask(np.array(output_size), radius=cc_radius).astype(np.single)

    # calculates the complex conjugate of fourier transformed target
    if wedge_mask_target is not None:
        wedge_target = cryomap.read(wedge_mask_target)
        conj_target, conj_target_sq = imageutils.calculate_conjugates(tomo, wedge_target)
    else:
        conj_target, conj_target_sq = imageutils.calculate_conjugates(tomo)

    if wedge_mask_tmpl is not None:
        wedge_tmpl = cryomap.read(wedge_mask_tmpl)

    # make an array of starting angles the same shape as angles
    _sa = np.asarray(starting_angle, dtype=float) if starting_angle is not None else np.zeros(3)
    starting_angles = np.tile(_sa, (angles.shape[0], 1))

    # calculates angular/cone/inplane distances
    ang_dist, cone, inplane = geom.compare_rotations(starting_angles, angles, cyclic_symmetry=cyclic_symmetry)

    res_table = pd.DataFrame(
        columns=[
            "ang_dist",
            "cone_dist",
            "inplane_dist",
            "common_voxels",
            "ccc",
            "ccc_masked",
            "z_score",
            "z_score_masked",
        ],
        dtype=float,
    )

    # init output maps
    final_ccc_map = np.full(output_size, -1)
    final_angles_map = np.full(output_size, -1)

    for i, a in enumerate(angles):

        # rotate the template and the mask
        rot_ref = cryomap.rotate(tmpl, rotation_angles=a, spline_order=1).astype(np.single)
        rot_mask = cryomap.rotate(mask, rotation_angles=a, spline_order=1).astype(np.single)

        rot_mask[rot_mask < 0.001] = 0.0  # Cutoff values for weird interpolated values
        rot_mask[rot_mask > 1.000] = 1.0  # Cutoff values

        # mask out missing wedge for template
        if wedge_mask_tmpl is not None:
            rot_ref = np.fft.ifftn(np.fft.fftn(rot_ref) * wedge_tmpl).real

        norm_ref = cryomap.normalize_under_mask(rot_ref, rot_mask)
        masked_ref = norm_ref * rot_mask

        # calculates fast local correlation coefficient
        cc_map = imageutils.calculate_flcf(masked_ref, rot_mask, conj_target=conj_target, conj_target_sq=conj_target_sq)
        z_score = (cc_map - np.mean(cc_map)) / np.std(cc_map)

        # find the indices where the current ccc is bigger than ccc in init/saved cc map
        max_idx = np.argmax((final_ccc_map, cc_map), 0).astype(bool)

        # overwrite in the same position with the bigger ccc (corresponding ang dist)
        final_ccc_map = np.maximum(final_ccc_map, cc_map)
        final_angles_map[max_idx] = i

        masked_map = cc_map * cc_mask
        z_score_masked = z_score * cc_mask

        # update the table with a new row
        res_table.loc[len(res_table)] = [
            ang_dist[i],
            cone[i],
            inplane[i],
            cryomask.mask_overlap(mask, rot_mask),
            np.max(cc_map),
            np.max(masked_map),
            np.max(z_score),
            np.max(z_score_masked),
        ]

    # make sure the cc map value stays between 0 and 1
    final_ccc_map = np.clip(final_ccc_map, 0.0, 1.0)
    final_ccc_map_masked = final_ccc_map * cc_mask

    if output_path is not None:
        res_table.to_csv(output_path + ".csv", index=False)
        cryomap.write(output_path=output_path + "_scores.em", data_to_write=final_ccc_map, data_type=np.single)
        # cryomap.write(output_path=output_path + '_scores_masked.em',
        #               data_to_write = final_ccc_map_masked,
        #               data_type=np.single)
        cryomap.write(output_path=output_path + "_angles.em", data_to_write=final_angles_map, data_type=np.single)

    return res_table, final_ccc_map, final_angles_map, final_ccc_map_masked


# ---------------------------------------------------------------------------
# Layer 2 — single-case orchestrators
# ---------------------------------------------------------------------------

_ARTIFACT_FILES = (
    "scores.em",
    "angles.em",
    "angles.csv",
    "stats.json",
    "peak_line_profiles.csv",
    "distance_map_all.em",
    "distance_map_normals.em",
    "distance_map_inplane.em",
    "distance_map_all_label.em",
    "distance_map_normals_label.em",
    "distance_map_inplane_label.em",
    "distance_map_all_label_open.em",
    "distance_map_normals_label_open.em",
    "distance_map_inplane_label_open.em",
)


def _resolve_write_dir(
    output_dir: PathOrStr,
    case_name: str,
    if_exists: Literal["overwrite", "error", "timestamp"],
) -> Path:
    case_dir = Path(output_dir) / case_name
    if if_exists == "overwrite":
        case_dir.mkdir(parents=True, exist_ok=True)
        return case_dir
    if if_exists == "error":
        if case_dir.exists() and any((case_dir / f).exists() for f in _ARTIFACT_FILES):
            raise FileExistsError(f"Output artifacts already exist in {case_dir}")
        case_dir.mkdir(parents=True, exist_ok=True)
        return case_dir
    # "timestamp"
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = case_dir / f"run_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


@gui_exposed(
    label="Compute angular distance maps",
    category="Peak Analysis",
    output="dataframe",
    hide=("output_dir", "scores_map", "degrees", "morph_footprint"),
)
def compute_distance_map(
    angles_map: MapSource,
    angles_list: DataSource,
    starting_angle: EulerAngles | None = None,
    cyclic_symmetry: Symmetry = 1,
    angles_order: str = "zxz",
    *,
    scores_map: MapSource | None = None,
    degrees: float | None = None,
    morph_footprint: TripletLike = (2, 2, 2),
    output_dir: PathOrStr | None = None,
) -> dict:
    """Compute angular distance maps relative to ``starting_angle``.

    For each voxel in ``angles_map`` (a 0-based index into ``angles_list``), the
    three distances ``dist_all``, ``dist_normals``, ``dist_inplane`` are computed
    between that voxel's Euler triple and ``starting_angle`` using
    :func:`cryocat.utils.geom.compare_rotations`.  Unset voxels (sentinel ``-1``)
    receive distance 0.  ``starting_angle=None`` defaults to ``(0, 0, 0)``.

    Parameters
    ----------
    angles_map : MapSource
        3-D integer map of 0-based angle indices produced by
        :func:`analyze_rotations`.  Unset voxels carry value ``-1``.  A
        ``MapSource`` is an ndarray or a path to a map file.
    angles_list : DataSource
        ``(N, 3)`` Euler-angle array or path to an angles file.  A
        ``DataSource`` is a path to a file or an ndarray.
    starting_angle : EulerAngles, optional
        Reference rotation from which distances are measured.  ``None`` is
        treated as ``(0, 0, 0)``.  An ``EulerAngles`` is a ``(3,)`` triple or
        ``(1, 3)`` ndarray.
    cyclic_symmetry : Symmetry, default=1
        Cyclic symmetry order passed to :func:`cryocat.utils.geom.compare_rotations`.
        A ``Symmetry`` is an int or a string like ``"C5"``.
    angles_order : str, default='zxz'
        Euler-angle convention of *angles_list*.
    scores_map : MapSource, optional
        3-D CC scores volume.  When provided together with *degrees*, the peak
        voxel is located and label maps are computed via :func:`dist_map_stats`.
        A ``MapSource`` is an ndarray or a path to a map file.
    degrees : float, optional
        Search-angle increment used to threshold the distance maps for labeling.
        Required together with *scores_map* to produce label volumes.
    morph_footprint : TripletLike, default=(2, 2, 2)
        Structuring element size for binary opening in :func:`dist_map_stats`.
        A ``TripletLike`` is a scalar or a 3-element sequence.
    output_dir : PathOrStr, optional
        Directory where output ``.em`` files are written.  Three distance maps
        are always written; if labels were computed, six additional label
        volumes are written as well.  A ``PathOrStr`` is a :class:`str` or
        :class:`pathlib.Path`.

    Returns
    -------
    dict
        ``dist_all`` – 3-D map of total angular distances;
        ``dist_normals`` – 3-D map of rotation-axis distances;
        ``dist_inplane`` – 3-D map of in-plane rotation distances;
        ``labels_all``, ``labels_normals``, ``labels_inplane`` – binary peak
        component volumes (``None`` when labels not computed);
        ``labels_all_open``, ``labels_normals_open``, ``labels_inplane_open``
        – morphologically opened label volumes (``None`` when not computed);
        ``output_dir`` – resolved output directory path or ``None``.
    """
    angles_map_arr = cryomap.read(angles_map) if not isinstance(angles_map, np.ndarray) else angles_map
    angles_arr = (
        ioutils.rot_angles_load(angles_list, angles_order) if not isinstance(angles_list, np.ndarray) else angles_list
    )

    # Reference rotation: None or (0, 0, 0) → measure from the identity.
    if starting_angle is None:
        ref_triple = np.zeros(3)
    else:
        ref_triple = np.asarray(starting_angle, dtype=float).reshape(-1)
        if ref_triple.shape != (3,):
            raise ValueError(f"starting_angle must be a length-3 Euler triple, got shape {ref_triple.shape}.")

    ref_angles = np.tile(ref_triple, (len(angles_arr), 1))
    dist_all, dist_normals, dist_inplane = geom.compare_rotations(
        ref_angles, angles_arr, cyclic_symmetry=cyclic_symmetry
    )

    map_shape = angles_map_arr.shape
    idx_flat = angles_map_arr.flatten().astype(int)  # 0-based; unset voxels are -1
    valid = idx_flat >= 0

    dist_all_flat = np.zeros(len(idx_flat))
    dist_normals_flat = np.zeros(len(idx_flat))
    dist_inplane_flat = np.zeros(len(idx_flat))
    dist_all_flat[valid] = dist_all[idx_flat[valid]]
    dist_normals_flat[valid] = dist_normals[idx_flat[valid]]
    dist_inplane_flat[valid] = dist_inplane[idx_flat[valid]]

    dist_all_map = dist_all_flat.reshape(map_shape).astype(np.single)
    dist_normals_map = dist_normals_flat.reshape(map_shape).astype(np.single)
    dist_inplane_map = dist_inplane_flat.reshape(map_shape).astype(np.single)

    # Compute label maps when scores_map and degrees are both available.
    labels_all = labels_normals = labels_inplane = None
    labels_all_open = labels_normals_open = labels_inplane_open = None

    if scores_map is not None and degrees is not None:
        sc_arr = cryomap.read(scores_map) if not isinstance(scores_map, np.ndarray) else scores_map
        sc_sphere = cryomask.spherical_mask(np.array(sc_arr.shape), radius=10)
        peak_center, _, _ = tmana.create_starting_parameters_2D(sc_arr * sc_sphere)

        st_all = dist_map_stats(dist_all_map, peak_center, degrees, is_all=True, morph_footprint=morph_footprint)
        st_nrm = dist_map_stats(dist_normals_map, peak_center, degrees, is_all=False, morph_footprint=morph_footprint)
        st_inp = dist_map_stats(dist_inplane_map, peak_center, degrees, is_all=False, morph_footprint=morph_footprint)

        labels_all = st_all["label"].astype(np.single)
        labels_normals = st_nrm["label"].astype(np.single)
        labels_inplane = st_inp["label"].astype(np.single)
        labels_all_open = st_all["open_label"].astype(np.single)
        labels_normals_open = st_nrm["open_label"].astype(np.single)
        labels_inplane_open = st_inp["open_label"].astype(np.single)

    result = {
        "dist_all": dist_all_map,
        "dist_normals": dist_normals_map,
        "dist_inplane": dist_inplane_map,
        "labels_all": labels_all,
        "labels_normals": labels_normals,
        "labels_inplane": labels_inplane,
        "labels_all_open": labels_all_open,
        "labels_normals_open": labels_normals_open,
        "labels_inplane_open": labels_inplane_open,
        "output_dir": None,
    }

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        cryomap.write(dist_all_map, str(out / "distance_map_all.em"), data_type=np.single)
        cryomap.write(dist_normals_map, str(out / "distance_map_normals.em"), data_type=np.single)
        cryomap.write(dist_inplane_map, str(out / "distance_map_inplane.em"), data_type=np.single)
        if labels_all is not None:
            cryomap.write(labels_all, str(out / "distance_map_all_label.em"), data_type=np.single)
            cryomap.write(labels_normals, str(out / "distance_map_normals_label.em"), data_type=np.single)
            cryomap.write(labels_inplane, str(out / "distance_map_inplane_label.em"), data_type=np.single)
            cryomap.write(labels_all_open, str(out / "distance_map_all_label_open.em"), data_type=np.single)
            cryomap.write(labels_normals_open, str(out / "distance_map_normals_label_open.em"), data_type=np.single)
            cryomap.write(labels_inplane_open, str(out / "distance_map_inplane_label_open.em"), data_type=np.single)
        result["output_dir"] = out

    return result


@gui_exposed(
    label="Compute peak statistics",
    category="Peak Analysis",
    output="dataframe",
    hide=("output_dir",),
)
def compute_peak_stats(
    scores_map: MapSource,
    dist_all_map: MapSource,
    dist_normals_map: MapSource,
    dist_inplane_map: MapSource,
    degrees: float,
    cc_radius: int = 10,
    morph_footprint: TripletLike = (2, 2, 2),
    output_dir: PathOrStr | None = None,
) -> dict:
    """Compute morphological and profile statistics for the peak in a scores map.

    Finds the highest-scoring voxel within a central sphere, then computes
    connected-component morphology for each angular distance map (voxel count,
    solidity, opened voxel count, bounding-box dimensions) and line-profile /
    neighbourhood statistics for the scores-map peak.

    Parameters
    ----------
    scores_map : MapSource
        3-D CC scores volume.  A ``MapSource`` is an ndarray or a path to a
        map file (.mrc, .em, …).
    dist_all_map : MapSource
        3-D total angular distance map.
    dist_normals_map : MapSource
        3-D rotation-axis distance map.
    dist_inplane_map : MapSource
        3-D in-plane rotation distance map.
    degrees : float
        Search-angle increment (degrees) used to threshold the distance maps.
    cc_radius : int, default=10
        Radius (voxels) of the central sphere used for peak detection.
    morph_footprint : TripletLike, default=(2, 2, 2)
        Structuring element size for binary opening in :func:`dist_map_stats`.
        A ``TripletLike`` is a scalar or a 3-element sequence.
    output_dir : PathOrStr, optional
        If provided, ``stats.json`` and ``peak_line_profiles.csv`` are written
        into this directory.  A ``PathOrStr`` is a :class:`str` or
        :class:`pathlib.Path`.

    Returns
    -------
    dict
        ``"peak_stats"`` – peak value, coordinates, score drops, and
        spherical-neighbourhood means / medians / variances for radii 1–5;
        ``"dist_maps"`` – keyed by ``"dist_all"``, ``"dist_normals"``,
        ``"dist_inplane"``, each containing ``vc``, ``solidity``, ``vco``, ``dim``;
        ``"peak_line_profiles"`` – :class:`pandas.DataFrame` with columns
        ``"x"``, ``"y"``, ``"z"``.
    """
    scores_map = cryomap.read(scores_map) if not isinstance(scores_map, np.ndarray) else scores_map
    dist_all_map = cryomap.read(dist_all_map) if not isinstance(dist_all_map, np.ndarray) else dist_all_map
    dist_normals_map = (
        cryomap.read(dist_normals_map) if not isinstance(dist_normals_map, np.ndarray) else dist_normals_map
    )
    dist_inplane_map = (
        cryomap.read(dist_inplane_map) if not isinstance(dist_inplane_map, np.ndarray) else dist_inplane_map
    )

    sphere_mask = cryomask.spherical_mask(np.array(scores_map.shape), radius=cc_radius)
    masked_scores = scores_map * sphere_mask
    peak_center, peak_value, _ = tmana.create_starting_parameters_2D(masked_scores)

    dist_map_names = ["dist_all", "dist_normals", "dist_inplane"]
    dist_map_arrays = [dist_all_map, dist_normals_map, dist_inplane_map]
    dist_results: dict[str, dict] = {}
    for j, (name, dm) in enumerate(zip(dist_map_names, dist_map_arrays)):
        st = dist_map_stats(dm, peak_center, degrees, is_all=(j == 0), morph_footprint=morph_footprint)
        dist_results[name] = {
            "vc": float(st["vc"][0]) if len(st["vc"]) else 0.0,
            "solidity": float(st["solidity"][0]) if len(st["solidity"]) else 0.0,
            "vco": int(st["vco"]),
            "dim": [float(v) for v in st["dim"]],
        }

    ps = peak_stats_and_profiles(scores_map, peak_center, peak_value)
    peak_result: dict = {
        "peak_value": float(ps["peak_value"]),
        "peak_x": int(ps["peak_x"]),
        "peak_y": int(ps["peak_y"]),
        "peak_z": int(ps["peak_z"]),
        "drop_x": float(ps["drop_x"]),
        "drop_y": float(ps["drop_y"]),
        "drop_z": float(ps["drop_z"]),
        "line_profiles": ps["line_profiles"].tolist(),
    }
    for r in range(1, 6):
        peak_result[f"mean_{r}"] = float(ps[f"mean_{r}"])
        peak_result[f"median_{r}"] = float(ps[f"median_{r}"])
        peak_result[f"var_{r}"] = float(ps[f"var_{r}"])

    lp_df = pd.DataFrame(peak_result.pop("line_profiles"), columns=["x", "y", "z"])
    result = {"peak_stats": peak_result, "dist_maps": dist_results, "peak_line_profiles": lp_df}

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        with open(str(out / "stats.json"), "w") as fh:
            json.dump({"peak_stats": peak_result, "dist_maps": dist_results}, fh, indent=2)
        lp_df.to_csv(str(out / "peak_line_profiles.csv"), index=False)

    return result


# Module-level aliases so run_single_case can reference these functions even
# when its bool parameters shadow the names in its local scope.
_compute_distance_map = compute_distance_map
_compute_peak_stats = compute_peak_stats


def _make_slice_figure(
    arr: np.ndarray,
    colorscale: str,
    title: str,
    peak_center: tuple | None = None,
    zoom_half: int = 10,
    zmin: float | None = None,
    zmax: float | None = None,
) -> go.Figure:
    """Return a 2-row subplot: orthogonal slices (row 1) + zoom patches with value labels (row 2).

    Each column corresponds to one principal axis (0, 1, 2).  The zoom window
    is ``2*zoom_half × 2*zoom_half`` voxels centred on *peak_center*, clipped
    to array bounds.  Pixel values are rendered as text inside each zoom cell.
    All subplot cells have a fixed 1:1 aspect ratio so a cubic volume appears
    as squares.
    """
    slices = cryomap.get_cross_slices(
        arr,
        slice_numbers=list(peak_center) if peak_center is not None else None,
        axis=[0, 1, 2],
    )
    if peak_center is not None:
        px = int(round(peak_center[0]))
        py = int(round(peak_center[1]))
        pz = int(round(peak_center[2]))
        zoom_centers = [(py, pz), (px, pz), (px, py)]
    else:
        zoom_centers = [(sl.shape[0] // 2, sl.shape[1] // 2) for sl in slices]

    _zmin = float(arr.min()) if zmin is None else zmin
    _zmax = float(arr.max()) if zmax is None else zmax

    fig = make_subplots(
        rows=2,
        cols=3,
        subplot_titles=["Cross-section XY", "Cross-section YZ", "Cross-section XZ", "Zoom XY", "Zoom YZ", "Zoom XZ"],
        row_heights=[0.6, 0.4],
        vertical_spacing=0.10,
    )

    for k, (sl, (rc, cc)) in enumerate(zip(slices, zoom_centers)):
        fig.add_trace(
            go.Heatmap(
                z=sl,
                colorscale=colorscale,
                zmin=_zmin,
                zmax=_zmax,
                showscale=(k == 0),
            ),
            row=1,
            col=k + 1,
        )
        r0 = max(0, rc - zoom_half)
        r1 = min(sl.shape[0], rc + zoom_half)
        c0 = max(0, cc - zoom_half)
        c1 = min(sl.shape[1], cc + zoom_half)
        crop = sl[r0:r1, c0:c1]
        abs_max = float(np.abs(crop).max()) if crop.size else 1.0
        fmt = ".2f" if abs_max < 10 else ".1f"
        text_vals = [[f"{v:{fmt}}" for v in row] for row in crop.tolist()]
        fig.add_trace(
            go.Heatmap(
                z=crop,
                text=text_vals,
                texttemplate="%{text}",
                textfont={"size": 7},
                colorscale=colorscale,
                zmin=_zmin,
                zmax=_zmax,
                showscale=False,
            ),
            row=2,
            col=k + 1,
        )

    # fixed 1:1 aspect ratio for every subplot cell (6 cells in a 2×3 grid)
    for i in range(6):
        sfx = "" if i == 0 else str(i + 1)
        fig.update_layout(**{f"yaxis{sfx}": {"scaleanchor": f"x{sfx}", "scaleratio": 1}})

    fig.update_layout(title_text=title, height=650)
    return fig


@gui_exposed(
    label="Visualize peak-analysis results",
    category="Peak Analysis",
    output="figures",
)
def visualize_results(
    *,
    scores: MapSource | None = None,
    angles_map: MapSource | None = None,
    angles_list: DataSource | None = None,
    dist_all_map: MapSource | None = None,
    dist_normals_map: MapSource | None = None,
    dist_inplane_map: MapSource | None = None,
    peak_stats: dict | PathOrStr | None = None,
) -> dict[str, go.Figure]:
    """Build Plotly figures for whichever result artifacts are present.

    Each panel is produced only when its prerequisites are available; missing
    inputs are silently skipped (graceful degradation).

    Parameters
    ----------
    scores : MapSource, optional
        3-D CC scores volume.  Required for ``"score_slices"``,
        ``"line_profiles"``, and ``"peak_shape"`` panels.  A ``MapSource`` is
        an ndarray or a path to a map file.
    angles_map : MapSource, optional
        3-D integer index map (output of :func:`analyze_rotations`).
        Required for the ``"angle_distribution"`` panel.
    angles_list : DataSource, optional
        ``(N, 3)`` Euler-angle array or path to an angles file.  Enriches the
        ``"angle_distribution"`` panel with angle labels.  A ``DataSource`` is
        a path to a file or an ndarray.
    dist_all_map : MapSource, optional
        3-D total angular-distance map.  Required for ``"distance_slices_all"``.
    dist_normals_map : MapSource, optional
        3-D rotation-axis distance map.  Required for ``"distance_slices_normals"``.
    dist_inplane_map : MapSource, optional
        3-D in-plane distance map.  Required for ``"distance_slices_inplane"``.
    peak_stats : dict or PathOrStr, optional
        Stats dict as returned by :func:`compute_peak_stats`, or a
        ``PathOrStr`` path to the JSON file.  When provided, the peak location
        is used to centre the slice and zoom panels.

    Returns
    -------
    dict[str, go.Figure]
        Possible keys: ``"score_slices"``, ``"line_profiles"``,
        ``"peak_shape"``, ``"distance_slices_all"``,
        ``"distance_slices_normals"``, ``"distance_slices_inplane"``,
        ``"angle_distribution"``.
        Only panels whose prerequisites are present are included.
    """

    # ── load all inputs ──────────────────────────────────────────────────────
    def _load_map(src):
        if src is None:
            return None
        return cryomap.read(src) if not isinstance(src, np.ndarray) else src

    scores_arr = _load_map(scores)
    amap_arr = _load_map(angles_map)
    dmap_all_arr = _load_map(dist_all_map)
    dmap_normals_arr = _load_map(dist_normals_map)
    dmap_inplane_arr = _load_map(dist_inplane_map)

    alist_arr: np.ndarray | None = None
    if angles_list is not None:
        alist_arr = (
            ioutils.rot_angles_load(angles_list, "zxz") if not isinstance(angles_list, np.ndarray) else angles_list
        )

    ps_dict: dict | None = None
    if peak_stats is not None:
        if isinstance(peak_stats, dict):
            ps_dict = peak_stats
        else:
            with open(str(peak_stats)) as fh:
                ps_dict = json.load(fh)

    peak_center: tuple | None = None
    if ps_dict is not None:
        _ps = ps_dict.get("peak_stats", ps_dict)
        _pc = (_ps.get("peak_x"), _ps.get("peak_y"), _ps.get("peak_z"))
        if None not in _pc:
            peak_center = _pc

    figs: dict[str, go.Figure] = {}

    # ── score_slices ─────────────────────────────────────────────────────────
    if scores_arr is not None:
        figs["score_slices"] = _make_slice_figure(
            scores_arr,
            colorscale="Viridis",
            title="Score map — orthogonal slices",
            peak_center=peak_center,
            zmin=0.0,
            zmax=float(scores_arr.max()),
        )

    # ── line_profiles ─────────────────────────────────────────────────────────
    if scores_arr is not None:
        _pc_lp, _pv_lp, _profiles = tmana.create_starting_parameters_1D(scores_arr)
        fig_lp = go.Figure()
        for k, dim in enumerate(["x", "y", "z"]):
            fig_lp.add_trace(go.Scatter(y=_profiles[:, k], name=dim, mode="lines"))
        fig_lp.update_layout(
            title_text="Line profiles through peak",
            xaxis_title="Position (voxel)",
            yaxis_title="CC score",
            height=380,
        )
        figs["line_profiles"] = fig_lp

    # ── peak_shape ────────────────────────────────────────────────────────────
    if scores_arr is not None:
        try:
            _ps_result = peak_shapes(scores_arr)
            fig_sh = go.Figure()
            for label, key in [
                ("Triangle", "tp_shape"),
                ("Gaussian", "gp_shape"),
                ("Half-peak", "hp_shape"),
            ]:
                _vals = _ps_result.get(key, [0, 0, 0])
                fig_sh.add_trace(go.Bar(x=["x", "y", "z"], y=list(_vals), name=label))
            fig_sh.update_layout(
                title_text=f"Peak shape (peak = {_ps_result.get('peak_value', 0.0):.4f})",
                barmode="group",
                height=380,
            )
            figs["peak_shape"] = fig_sh
        except Exception:
            pass  # non-fatal if peak_shapes fails (e.g. map too small)

    # ── distance map panels (all three types) ─────────────────────────────────
    _dist_specs = [
        (dmap_all_arr, "distance_slices_all", "Distance map (all) — orthogonal slices"),
        (dmap_normals_arr, "distance_slices_normals", "Distance map (normals) — orthogonal slices"),
        (dmap_inplane_arr, "distance_slices_inplane", "Distance map (in-plane) — orthogonal slices"),
    ]
    for _dmap, _key, _title in _dist_specs:
        if _dmap is not None:
            figs[_key] = _make_slice_figure(
                _dmap,
                colorscale="RdYlGn_r",
                title=_title,
                peak_center=peak_center,
            )

    # ── angle_distribution ────────────────────────────────────────────────────
    if amap_arr is not None:
        _flat = amap_arr.flatten().astype(int)
        _flat = _flat[_flat >= 0]  # drop unset sentinel -1
        if len(_flat) > 0:
            _counts = np.bincount(_flat)
            _idx = np.nonzero(_counts)[0]
            _vals = _counts[_idx]
            _labels = [str(i) for i in _idx]
            if alist_arr is not None:
                _labels = [f"{alist_arr[i, 1]:.1f}°" if 0 <= i < len(alist_arr) else str(i) for i in _idx]
            fig_ad = go.Figure(go.Bar(x=_labels, y=list(_vals)))
            fig_ad.update_layout(
                title_text="Angle index distribution",
                xaxis_title="Angle (theta) / index",
                yaxis_title="Voxel count",
                height=380,
            )
            figs["angle_distribution"] = fig_ad

    return figs


@gui_exposed(
    label="Analyze rotations (single case)",
    category="Peak Analysis",
    output="dataframe",
    hide=("output_path",),
)
def run_single_case(
    target_map: MapSource,
    template: MapSource,
    template_mask: MapSource,
    input_angles: DataSource,
    output_dir: PathOrStr,
    case_name: str,
    *,
    starting_angle: EulerAngles | None = None,
    angular_offset: EulerAngles | None = None,
    cyclic_symmetry: Symmetry = 1,
    wedge_mask_target: MapSource | None = None,
    wedge_mask_tmpl: MapSource | None = None,
    cc_radius: int = 10,
    degrees: float | None = None,
    morph_footprint: TripletLike = (2, 2, 2),
    compute_distance_map: bool = True,
    compute_peak_stats: bool = True,
    angles_order: str = "zxz",
    if_exists: Literal["overwrite", "error", "timestamp"] = "overwrite",
) -> dict:
    """Run a complete single-case peak analysis and return all results.

    Rotates ``template`` through ``input_angles`` and computes the normalized
    cross-correlation against the fixed ``target_map`` for each rotation.
    Orchestrates :func:`analyze_rotations`, :func:`compute_distance_map`, and
    :func:`compute_peak_stats`.  Artifacts are written to
    ``<output_dir>/<case_name>/``.

    Parameters
    ----------
    target_map : MapSource
        Fixed volume against which the rotated template is matched.  A
        ``MapSource`` is an ndarray or a path to a map file (.mrc, .em, …).
    template : MapSource
        Reference template map (the rotated side).
    template_mask : MapSource
        Binary mask for the template.
    input_angles : DataSource
        Angle list for the rotation search.  A ``DataSource`` is a path to a
        file or an ndarray.
    output_dir : PathOrStr
        Root directory under which the case sub-folder is created.  A
        ``PathOrStr`` is a :class:`str` or :class:`pathlib.Path`.
    case_name : str
        Name of the sub-folder for this case.
    starting_angle : EulerAngles, optional
        Reference orientation applied as a base rotation (Euler angles,
        degrees).  An ``EulerAngles`` is a ``(3,)`` triple or ``(N, 3)``
        ndarray.  ``None`` is treated as ``(0, 0, 0)``.
    angular_offset : EulerAngles, optional
        Additional rotation applied to all input angles.
    cyclic_symmetry : Symmetry, default=1
        C symmetry of the structure.  A ``Symmetry`` is an int or a string
        like ``"C5"``.
    wedge_mask_target : MapSource, optional
        Wedge mask for the target map (path or array).  When ``None``, no
        target-side wedge correction is applied.
    wedge_mask_tmpl : MapSource, optional
        Wedge mask for the template (path or array).  When ``None``, no
        template-side wedge correction is applied.
    cc_radius : int, default=10
        Radius (voxels) of the central sphere.
    degrees : float, optional
        Search-angle increment for distance-map thresholding in
        :func:`compute_peak_stats`.  Required when *compute_peak_stats* is True.
    morph_footprint : TripletLike, default=(2, 2, 2)
        Structuring element for binary opening in :func:`compute_peak_stats`.
        A ``TripletLike`` is a scalar or a 3-element sequence.
    compute_distance_map : bool, default=True
        Whether to run :func:`compute_distance_map` after the rotation search.
    compute_peak_stats : bool, default=True
        Whether to run :func:`compute_peak_stats` after distance-map computation.
    angles_order : str, default='zxz'
        Euler-angle convention.
    if_exists : {'overwrite', 'error', 'timestamp'}, default='overwrite'
        Policy when output artifacts already exist.  ``'overwrite'`` writes
        directly; ``'error'`` raises :exc:`FileExistsError`; ``'timestamp'``
        creates a timestamped sub-folder.

    Returns
    -------
    dict
        Always contains ``"res_table"``, ``"scores_map"``, ``"angles_map"``,
        ``"write_dir"``.  When *compute_distance_map* is True, also contains
        ``"dist_all_map"``, ``"dist_normals_map"``, ``"dist_inplane_map"``.
        When *compute_peak_stats* is True and *degrees* is provided, also
        contains ``"peak_stats"``.
    """
    write_dir = _resolve_write_dir(output_dir, case_name, if_exists)

    # Pre-compute transformed angles so angles.csv is consistent with the
    # 0-based indices stored in angles.em by analyze_rotations.
    angles_array = geom.apply_starting_and_offset(
        ioutils.rot_angles_load(input_angles, angles_order),
        starting_angle,
        angular_offset,
        angles_order,
    )
    ioutils.angles_save(angles_array, str(write_dir / "angles.csv"), float_format="%.3f")

    res_table, scores_map, angles_map, _ = analyze_rotations(
        target_map=target_map,
        template=template,
        template_mask=template_mask,
        input_angles=input_angles,
        wedge_mask_target=wedge_mask_target,
        wedge_mask_tmpl=wedge_mask_tmpl,
        cc_radius=cc_radius,
        angular_offset=angular_offset,
        starting_angle=starting_angle,
        cyclic_symmetry=cyclic_symmetry,
        angles_order=angles_order,
    )

    cryomap.write(scores_map, str(write_dir / "scores.em"), data_type=np.single)
    cryomap.write(angles_map, str(write_dir / "angles.em"), data_type=np.single)

    result: dict = {
        "res_table": res_table,
        "scores_map": scores_map,
        "angles_map": angles_map,
        "write_dir": write_dir,
    }

    if compute_distance_map:
        _run_labels = compute_peak_stats and degrees is not None
        dist_result = _compute_distance_map(
            angles_map=angles_map,
            angles_list=angles_array,
            starting_angle=starting_angle,
            cyclic_symmetry=cyclic_symmetry,
            angles_order=angles_order,
            scores_map=scores_map if _run_labels else None,
            degrees=degrees if _run_labels else None,
            morph_footprint=morph_footprint,
            output_dir=str(write_dir),
        )
        da = dist_result["dist_all"]
        dn = dist_result["dist_normals"]
        di = dist_result["dist_inplane"]
        result["dist_all_map"] = da
        result["dist_normals_map"] = dn
        result["dist_inplane_map"] = di

        if _run_labels:
            stats = _compute_peak_stats(
                scores_map=scores_map,
                dist_all_map=da,
                dist_normals_map=dn,
                dist_inplane_map=di,
                degrees=degrees,
                cc_radius=cc_radius,
                morph_footprint=morph_footprint,
                output_dir=str(write_dir),
            )
            result["peak_stats"] = stats

    return result


def _run_analysis_args_from_row(row: pd.Series, parent_folder_path: PathOrStr, wedge_path: PathOrStr) -> dict:
    """Resolve file paths and rotation args for one template-list row.

    Parameters
    ----------
    row : pandas.Series
        A single row of the template-list DataFrame, indexed by column name.
    parent_folder_path : PathOrStr
        Root folder containing structure subfolders, templates, and tomograms.
    wedge_path : PathOrStr
        Directory containing wedge mask files.

    Returns
    -------
    dict
        ``structure_name``, ``template``, ``mask``, ``target_map``,
        ``wedge_target``, ``wedge_tmpl``, ``starting_angle`` (shape ``(1, 3)``),
        ``cyclic_symmetry``.
    """
    structure_name = row["Structure"]
    template = create_em_path(parent_folder_path, structure_name, row["Template"])
    mask = create_em_path(parent_folder_path, structure_name, row["Mask"])

    wedge_target = None
    wedge_tmpl = None

    target_col = "Target map" if "Target map" in row.index else "Tomo map"

    if row["Compare"] == "tmpl":
        target_map = template
    elif row["Compare"] == "subtomo":
        target_map = create_em_path(parent_folder_path, structure_name, row[target_col])
        tomo_number = re.findall(r"\d+", row["Tomogram"])[0]
        if row["Apply wedge"]:
            wedge_target, wedge_tmpl = create_wedge_names(wedge_path, tomo_number, row["Boxsize"], row["Binning"])
    else:
        target_map = create_em_path(parent_folder_path, row["Compare"], row[target_col])

    starting_angle = row[["Phi", "Theta", "Psi"]].to_numpy().reshape(1, 3)
    cyclic_symmetry = row["Symmetry"]

    return {
        "structure_name": structure_name,
        "template": template,
        "mask": mask,
        "target_map": target_map,
        "wedge_target": wedge_target,
        "wedge_tmpl": wedge_tmpl,
        "starting_angle": starting_angle,
        "cyclic_symmetry": cyclic_symmetry,
    }


def run_analysis(
    template_list: PathOrStr,
    indices: list[int],
    angle_list_path: PathOrStr,
    wedge_path: PathOrStr,
    parent_folder_path: PathOrStr,
    cc_radius_tol: int = 10,
) -> None:
    """Run peak analysis based on a list with parameters and save results.

    This function iterates over the provided `indices` of a template list CSV,
    loads the corresponding tomogram, template, and mask files, and then calls
    `analyze_rotations` to perform rotation-based cross-correlation analysis.
    The results (score maps, angle maps, CSV stats) are written to an output
    folder for each index. It also generates angular distance maps for the
    resulting angle maps. The function updates the CSV in-place to record
    progress, ensuring partial results are saved in case of interruption.

    Parameters
    ----------
    template_list : PathOrStr
        Path to a CSV file containing info about peak analysis to perform. The CSV
        must include at least the following columns:
        - Structure
        - Template
        - Mask
        - Angles
        - Compare (compare method; "tmpl": tmpl vs. tmpl, "subtomo": subtomo vs. sutomo, or else)
        - Tomo map (i.e. subtomo)
        - Tomogram
        - Apply wedge (bool)
        - Boxsize
        - Binning
        - Phi, Theta, Psi (starting Euler angles in degrees)
        - Apply angular offset (bool)
        - Degrees (search angle increment / offset magnitude)
        - Symmetry (C symmetry)
    indices : sequence of int
        List or array of row indices (0-based) in `template_list` to process.
    angle_list_path : PathOrStr
        Base directory path where the angle list files are stored.
        The angle file name from the CSV's "Angles" column is appended to this path.
    wedge_path : PathOrStr
        Base directory containing wedge mask files. Used only if "Apply wedge"
        is set for the current row.
    parent_folder_path : PathOrStr
        Root folder containing all structure subfolders, templates, tomograms,
        and masks referenced in `template_list`.
    cc_radius_tol : int, default=10
        Radius (in voxels) of the spherical mask used for computing local
        cross-correlation scores in `analyze_rotations`.

    Notes
    -------
    - Writes output files for each processed index:
        * '<output_base>_scores.em' (cross-correlation coefficient map)
        * '<output_base>_angles.em' (best-angle index map)
        * CSV file with per-angle statistics
    - Writes angular distance maps via 'tmana.create_angular_distance_maps'
    - Updates `template_list` CSV in place:
        * "Output folder" set for each processed index
        * "Done" flag set to True
    - Creates any necessary output directories.
    - For rows with `"Compare" == "tmpl"`, the tomogram is the same as the template.
    - For `"Compare" == "subtomo"`, a tomogram is loaded from the specified file,
      and wedge masks may be applied if `"Apply wedge"` is true.
    - `starting_angle` is read directly from `"Phi"`, `"Theta"`, `"Psi"` columns.
    - If `"Apply angular offset"` is true, `angular_offset` is set to half of
      `Degrees` for all three Euler components; otherwise it is [0, 0, 0].
    - Symmetry (`cyclic_symmetry`) is passed to `analyze_rotations` to account for
      cyclic symmetry in angular distance calculations.
    """

    temp_df = pd.read_csv(template_list, index_col=0)

    for i in indices:
        args = _run_analysis_args_from_row(temp_df.loc[i], parent_folder_path, wedge_path)

        angle_list = angle_list_path + temp_df.at[i, "Angles"]

        if temp_df.at[i, "Apply angular offset"]:
            angular_offset = np.full((3,), temp_df.at[i, "Degrees"] / 2.0)
        else:
            angular_offset = np.asarray([0, 0, 0])

        output_base = create_output_base_name(i)
        output_folder = create_output_folder_path(parent_folder_path, args["structure_name"], i)

        temp_df.at[i, "Output folder"] = create_output_folder_name(i)
        Path(output_folder).mkdir(parents=True, exist_ok=True)

        _ = analyze_rotations(
            target_map=args["target_map"],
            template=args["template"],
            template_mask=args["mask"],
            input_angles=angle_list,
            wedge_mask_target=args["wedge_target"],
            wedge_mask_tmpl=args["wedge_tmpl"],
            output_path=output_folder + "/" + output_base,
            angular_offset=angular_offset,
            starting_angle=args["starting_angle"],
            cc_radius=cc_radius_tol,
            cyclic_symmetry=args["cyclic_symmetry"],
        )[0]

        angles_map = output_folder + "/" + output_base + "_angles.em"
        _, _, _ = tmana.create_angular_distance_maps(angles_map, angle_list, write_out_maps=True)

        temp_df.at[i, "Done"] = True
        temp_df.to_csv(template_list)  # save what was finished in case of a crush


def run_single_gradual_case(
    target_map: MapSource,
    template: MapSource,
    template_mask: MapSource,
    starting_angle: EulerAngles | None = None,
    cyclic_symmetry: Symmetry = 1,
    wedge_mask_target: MapSource | None = None,
    wedge_mask_tmpl: MapSource | None = None,
    angular_range: int = 359,
    cc_radius: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run a gradual rotation sweep for a single (target_map, template, mask) triple.

    Tests all integer angles from 0 to ``angular_range - 1`` degrees for three
    rotation types: combined (both cone and in-plane), cone-only, and in-plane
    only.  For each angle and type, :func:`analyze_rotations` is called and
    the resulting CC metrics are collected.

    Parameters
    ----------
    target_map : MapSource
        Fixed volume against which the rotated template is matched.  A
        ``MapSource`` is an ndarray or a path to a map file.
    template : MapSource
        Reference template map (the rotated side).
    template_mask : MapSource
        Binary mask for the template.
    starting_angle : EulerAngles, optional
        Base orientation applied before each test angle.  An ``EulerAngles`` is
        a ``(3,)`` triple or ``(N, 3)`` ndarray.  ``None`` is treated as
        ``(0, 0, 0)``.
    cyclic_symmetry : Symmetry, default=1
        C symmetry of the structure.  A ``Symmetry`` is an int or a string
        like ``"C5"``.
    wedge_mask_target : MapSource, optional
        Wedge mask for the target map.
    wedge_mask_tmpl : MapSource, optional
        Wedge mask for the template.
    angular_range : int, default=359
        Number of integer degrees to test (0 to ``angular_range - 1``).
    cc_radius : int, default=10
        Radius (voxels) of the central sphere for CC evaluation.

    Returns
    -------
    tuple of (pandas.DataFrame, pandas.DataFrame)
        ``final_df`` – per-angle CC metrics for all three rotation types
        concatenated column-wise;
        ``hist_df`` – cumulative CC-score histograms for each rotation type.
    """
    results = np.zeros((angular_range, 8, 3))
    n_bins = 100
    final_hist = np.zeros((n_bins, 3))

    for a in range(angular_range):
        angles = np.full((3, 3), float(a))
        angles[1, 0] = 0.0  # cone-only: phi=0
        angles[2, 1:] = 0.0  # inplane-only: theta=psi=0

        for j in range(3):
            res_df, cc_map, _, _ = analyze_rotations(
                target_map=target_map,
                template=template,
                template_mask=template_mask,
                input_angles=angles[j, :].reshape(1, 3),
                wedge_mask_target=wedge_mask_target,
                wedge_mask_tmpl=wedge_mask_tmpl,
                output_path=None,
                starting_angle=starting_angle,
                cc_radius=cc_radius,
                cyclic_symmetry=cyclic_symmetry,
            )
            results[a, :, j] = res_df.values
            hist, _ = np.histogram(cc_map, bins=n_bins, range=(0.0, 1.0))
            final_hist[:, j] += hist

    ang_dist = pd.DataFrame(
        data=results[:, :, 0],
        columns=[
            "ang_dist",
            "cone_dist",
            "inplane_dist",
            "common_voxels",
            "ccc",
            "ccc_masked",
            "z_score",
            "z_score_masked",
        ],
    )
    ang_cone = pd.DataFrame(
        data=results[:, :, 1],
        columns=[
            "cone_ang_dist",
            "cone_cone_dist",
            "cone_inplane_dist",
            "cone_common_voxels",
            "cone_ccc",
            "cone_ccc_masked",
            "cone_z_score",
            "cone_z_score_masked",
        ],
    )
    ang_inplane = pd.DataFrame(
        data=results[:, :, 2],
        columns=[
            "inplane_ang_dist",
            "inplane_cone_dist",
            "inplane_inplane_dist",
            "inplane_common_voxels",
            "inplane_ccc",
            "inplane_ccc_masked",
            "inplane_z_score",
            "inplane_z_score_masked",
        ],
    )
    final_df = pd.concat([ang_dist, ang_cone, ang_inplane], axis=1)
    hist_df = pd.DataFrame(data=final_hist, columns=["ang_dist", "cone_dist", "inplane_dist"])

    return final_df, hist_df


def run_angle_analysis(
    template_list: PathOrStr,
    indices: list[int],
    wedge_path: PathOrStr,
    parent_folder_path: PathOrStr,
    angular_range: int = 359,
    write_output: bool = False,
    cc_radius_tol: int = 10,
) -> None:
    """Perform a gradual rotation angular peak analysis.

    This function systematically evaluates the effect of varying Euler angles
    (3 kinds of rotations: full angular distance, cone rotation, and in-plane rotation)
    on the cross-correlation between a tomogram (or template) and a reference template.
    It iterates through a specified angular range 1 deg by 1 deg, computes
    correlation metrics, and optionally saves detailed analysis results and histograms.

    Parameters
    ----------
    template_list : PathOrStr
        Path to a CSV file containing metadata and file paths for templates,
        tomograms, masks, and analysis parameters.
    indices : list of int
        List of row indices in `template_list` to process.
    wedge_path : PathOrStr
        Directory path containing wedge mask files.
    parent_folder_path : PathOrStr
        Root directory containing structure and template data.
    angular_range : int, default=359
        Number of degrees to test in the rotation range. 359 means all integer
        angles from 0 to 358 are analyzed, covering a full circle.
    write_output : bool, default=False
        If True, saves the computed angular analysis results and histograms
        as CSV files in the corresponding output directory.
    cc_radius_tol : int, default=10
        Radius (in voxels) of the spherical mask applied to the cross-correlation
        map when evaluating local correlation.

    Notes
    -----
    - A histogram of CCC values across the angular range is also computed for each
      rotation type.

    - Output files are:

      - '<output_base>_gradual_angles_analysis.csv': Table containing detailed
        rotation metrics for all tested angles and rotation types

      - '<output_base>_gradual_angles_histograms.csv': Histograms of CCC values for
        each rotation type
    """

    temp_df = pd.read_csv(template_list, index_col=0)

    for i in indices:
        args = _run_analysis_args_from_row(temp_df.loc[i], parent_folder_path, wedge_path)

        final_df, hist_df = run_single_gradual_case(
            target_map=args["target_map"],
            template=args["template"],
            template_mask=args["mask"],
            starting_angle=args["starting_angle"],
            cyclic_symmetry=args["cyclic_symmetry"],
            wedge_mask_target=args["wedge_target"],
            wedge_mask_tmpl=args["wedge_tmpl"],
            angular_range=angular_range,
            cc_radius=cc_radius_tol,
        )

        if write_output:
            output_base = create_output_folder_path(
                parent_folder_path, args["structure_name"], temp_df.at[i, "Output folder"]
            ) + create_output_base_name(i)
            final_df.to_csv(output_base + "_gradual_angles_analysis.csv")
            hist_df.to_csv(output_base + "_gradual_angles_histograms.csv")


def build_summary_figure(
    figure_title: str,
    dicts: list,
    rot_info: pd.DataFrame,
    line_profiles: pd.DataFrame,
    cross_slices: list,
    peak_val: float,
    hist_info: pd.DataFrame | None = None,
    hist_info2: pd.DataFrame | None = None,
) -> go.Figure:
    """Build a multi-panel Plotly figure summarising one peak-analysis result.

    Parameters
    ----------
    figure_title : str
        Title string rendered at the top of the figure.
    dicts : list of list of [str, str]
        Three sub-lists, each a sequence of ``[key, value]`` pairs used to
        populate the three summary tables in row 1.
    rot_info : pandas.DataFrame
        Per-rotation CC data with columns ``"Tight mask overlap"``,
        ``"ang_dist"``, and ``"ccc_masked"``.
    line_profiles : pandas.DataFrame
        Line profiles along x/y/z through the peak, one column each.
    cross_slices : list
        Cross-section arrays from ``cryomap.get_cross_slices``.
    peak_val : float
        Maximum CC score; sets the upper bound of the viridis colorbar.
    hist_info : pandas.DataFrame or None, optional
        Histogram CSV data (row 3 is included only when not None).
    hist_info2 : pandas.DataFrame or None, optional
        Gradual-angles analysis CSV data (paired with *hist_info*).

    Returns
    -------
    plotly.graph_objects.Figure
    """
    add_hist = hist_info is not None

    n_hm_rows = len(cross_slices)
    n_top_rows = 2 + (1 if add_hist else 0)
    total_rows = n_top_rows + n_hm_rows

    specs = [
        [{"type": "table"}, {"type": "table"}, {"type": "table"}],
        [{"type": "xy"}, {"type": "xy"}, {"type": "xy"}],
    ]
    if add_hist:
        specs.append([{"type": "xy"}, {"type": "xy"}, None])
    for _ in range(n_hm_rows):
        specs.append([{"type": "xy"}, {"type": "xy"}, {"type": "xy"}])

    row_heights_w = [2.5, 1.2]
    if add_hist:
        row_heights_w.append(0.8)
    row_heights_w.extend([1.0] * n_hm_rows)

    # Calculate colorbar y positions based on row layout
    vs = 0.02
    content_h = 1.0 - (total_rows - 1) * vs
    total_w = sum(row_heights_w)
    row_h_frac = [w / total_w * content_h for w in row_heights_w]
    row_tops, row_bottoms = [], []
    cum = 0.0
    for k, h in enumerate(row_h_frac):
        top = 1.0 - cum - (vs if k > 0 else 0)
        row_tops.append(top)
        row_bottoms.append(top - h)
        cum += h + (vs if k < total_rows - 1 else 0)

    s = n_top_rows  # index of first heatmap row
    ca1_y = (row_tops[s] + row_bottoms[s]) / 2
    ca1_len = row_tops[s] - row_bottoms[s]
    ca2_y = (row_tops[s + 1] + row_bottoms[s + 2]) / 2
    ca2_len = row_tops[s + 1] - row_bottoms[s + 2]
    ca3_y = (row_tops[s + 3] + row_bottoms[s + 5]) / 2
    ca3_len = row_tops[s + 3] - row_bottoms[s + 5]

    fig = make_subplots(
        rows=total_rows,
        cols=3,
        specs=specs,
        row_heights=row_heights_w,
        horizontal_spacing=0.03,
        vertical_spacing=vs,
    )

    # Row 1: tables
    for j, d in enumerate(dicts, start=1):
        keys = [row[0] for row in d]
        values = [row[1] for row in d]
        fig.add_trace(
            go.Table(
                header=dict(values=["Parameter", "Value"], align="left", fill_color="lightgrey", font=dict(size=10)),
                cells=dict(values=[keys, values], align="left", font=dict(size=10)),
            ),
            row=1,
            col=j,
        )

    # Row 2: scatter + line plots
    fig.add_trace(
        go.Scatter(
            x=rot_info["Tight mask overlap"],
            y=rot_info["ccc_masked"],
            mode="markers",
            marker=dict(size=3),
            showlegend=False,
        ),
        row=2,
        col=1,
    )
    fig.update_xaxes(title_text="Tight mask overlap (in voxels)", row=2, col=1)
    fig.update_yaxes(title_text="CCC", row=2, col=1)

    fig.add_trace(
        go.Scatter(
            x=rot_info["ang_dist"], y=rot_info["ccc_masked"], mode="markers", marker=dict(size=3), showlegend=False
        ),
        row=2,
        col=2,
    )
    fig.update_xaxes(title_text="Angular distance (in degrees)", row=2, col=2)

    x_pos = list(range(len(line_profiles)))
    for col_name in ["x", "y", "z"]:
        fig.add_trace(
            go.Scatter(x=x_pos, y=line_profiles[col_name], mode="lines", name=col_name),
            row=2,
            col=3,
        )
    fig.update_xaxes(title_text="Position (in voxels)", row=2, col=3)

    # Row 3 (optional): histogram line plots
    if add_hist:
        x100 = np.linspace(0.0, 1.0, num=100)
        for col_name in ["ang_dist", "cone_dist", "inplane_dist"]:
            fig.add_trace(
                go.Scatter(x=x100, y=hist_info[col_name], mode="lines", name=col_name, showlegend=False),
                row=3,
                col=1,
            )
        fig.update_yaxes(range=[0, 250], title_text="Number of CCC values (bin size 0.1)", row=3, col=1)
        fig.update_xaxes(title_text="CCC", row=3, col=1)

        x359 = np.linspace(0.0, 359.0, num=359)
        for col_name in ["ccc_masked", "cone_ccc_masked", "inplane_ccc_masked"]:
            fig.add_trace(
                go.Scatter(x=x359, y=hist_info2[col_name], mode="lines", name=col_name, showlegend=False),
                row=3,
                col=2,
            )
        fig.update_yaxes(title_text="CCC", row=3, col=2)
        fig.update_xaxes(title_text="Rotation (in degrees)", row=3, col=2)

    # Heatmap rows (coloraxis1=gray/tight mask, coloraxis2=viridis/scores, coloraxis3=cividis/angles)
    for c, slice_group in enumerate(cross_slices):
        if c == 0:
            coloraxis, annot_fmt = "coloraxis1", None
        elif c == 1:
            coloraxis, annot_fmt = "coloraxis2", None
        elif c == 2:
            coloraxis, annot_fmt = "coloraxis2", ".2f"
        else:  # c >= 3
            coloraxis, annot_fmt = "coloraxis3", ".1f"

        hm_row = n_top_rows + c + 1
        visplot.add_xyz_heatmap_row(
            fig,
            slice_group,
            row=hm_row,
            coloraxis=coloraxis,
            annot_format=annot_fmt,
        )

    fig.update_layout(
        title_text=figure_title,
        height=max(900, 280 * total_rows),
        coloraxis1=dict(
            colorscale="gray",
            cmin=0,
            cmax=1.0,
            colorbar=dict(title="Tight mask", thickness=12, len=ca1_len, y=ca1_y, yanchor="middle", x=1.02),
        ),
        coloraxis2=dict(
            colorscale="viridis",
            cmin=0,
            cmax=peak_val,
            colorbar=dict(title="Score", thickness=12, len=ca2_len, y=ca2_y, yanchor="middle", x=1.10),
        ),
        coloraxis3=dict(
            colorscale="cividis",
            cmin=0,
            cmax=180,
            colorbar=dict(title="Angle (°)", thickness=12, len=ca3_len, y=ca3_y, yanchor="middle", x=1.18),
        ),
    )

    return fig


def create_summary_pdf(template_list: PathOrStr, indices: list[int], parent_folder_path: PathOrStr) -> None:
    """Generate a detailed summary PDF for a set of peak analysis results.

    This function reads metadata from a CSV file (`template_list`), retrieves
    volumetric map data, analysis results, and visualization slices for the
    specified `indices`, and compiles them into a structured multi-panel PDF
    report. Each report includes:

    - Template and processing parameters (symmetry, wedge application, voxel size, etc.)
    - Peak detection information (location, value, line profiles)
    - Distance map statistics and solidity/volume coverage measures
    - Scatter plots, histograms, and gradual rotation CCC analysis (if available)
    - Cross-sectional heatmaps of masks, score maps, and angular distance maps

    The output is saved as 'id_<index>_summary.pdf' in the corresponding
    output folder for each index.

    Parameters
    ----------
    template_list : PathOrStr
        Path to the CSV file containing metadata for all templates and analyses.
    indices : list of int
        List of row indices from `template_list` to process.
    parent_folder_path : PathOrStr
        Base directory containing the structure folders, output folders, and map files.

    Notes
    -----
    - If the "Done" column is False for a given index, that entry is skipped.
    - Gradual rotation histogram and CCC analysis are included if the corresponding
      '_gradual_angles_histograms.csv' and '_gradual_angles_analysis.csv' are found.
    """

    temp_df = pd.read_csv(template_list, index_col=0)

    for i in indices:
        if not temp_df.at[i, "Done"]:
            continue

        structure_name = temp_df.at[i, "Structure"]
        output_base = create_output_base_name(i)
        output_folder = create_output_folder_path(parent_folder_path, structure_name, temp_df.at[i, "Output folder"])

        def _p(suffix: str) -> str | None:
            path = output_folder + output_base + suffix
            return path if os.path.isfile(path) else None

        figs = visualize_results(
            scores=_p("_scores.em"),
            dist_all_map=_p("_angles_dist_all.em"),
            dist_normals_map=_p("_angles_dist_normals.em"),
            dist_inplane_map=_p("_angles_dist_inplane.em"),
            peak_stats=_p("_stats.json"),
        )

        for panel_name, fig in figs.items():
            fig.write_html(output_folder + output_base + f"_{panel_name}.html")


##########################################################################################################################
###### Following functions are not used in the current analysis but might come handy later ###############################
##########################################################################################################################


# Check what kind of descriptors skimage can offer
def get_shape_stats(
    template_list: PathOrStr, indices: list[int], shape_type: str, parent_folder_path: PathOrStr
) -> None:
    """Compute and save shape statistics for specific shapes in a template list.

    This function reads path from a CSV file, loads corresponding tight masks, labels
    connected components, computes geometric and morphological
    properties for each labeled region, and saves the results to CSV files.

    Parameters
    ----------
    template_list : PathOrStr
        Path to the CSV file containing metadata for all templates and analyses.
    indices : array_like of int
        Rows in `template_list` for which statistics should be computed.
    shape_type : str
        A descriptive label for the type of shape used for analysis. This string
        is appended to the output CSV filename.
    parent_folder_path : PathOrStr
        Path to the root directory containing the structure and mask files.

    Notes
    -----
    The following region properties are computed for each labeled region:
        - 'label' : integer label ID
        - 'area' : voxel count
        - 'area_bbox' : bounding box volume
        - 'area_convex' : convex hull volume
        - 'equivalent_diameter_area' : diameter of a sphere with same volume
        - 'euler_number' : topological Euler characteristic
        - 'feret_diameter_max' : maximum caliper distance
        - 'inertia_tensor' : 3×3 inertia tensor matrix
        - 'solidity' : ratio of area to convex hull area

    Output files are named in the format:
        '<structure_path>/<output_folder>/id_<index>_shape_stats_<shape_type>.csv'
    """

    temp_df = pd.read_csv(template_list, index_col=0)

    for i in indices:
        structure_name = temp_df.at[i, "Structure"]
        sharp_mask = cryomap.read(create_em_path(parent_folder_path, structure_name, temp_df.at[i, "Tight mask"]))

        stats_df = shape_stats(sharp_mask)

        output_base = (
            create_structure_path(parent_folder_path, structure_name)
            + temp_df.at[i, "Output folder"]
            + "/id_"
            + str(i)
            + "_shape_stats_"
        )
        stats_df.to_csv(output_base + shape_type + ".csv")


def compute_peak_shapes(template_list: PathOrStr, indices: list[int], parent_folder_path: PathOrStr) -> None:
    """
    Compute and record peak shape statistics from scores maps for selected structures.

    This function processes score maps for the given structures, evaluates the peak
    shapes using multiple thresholding methods, stores the results in the template
    list, and generates a visualization of the peaks and thresholds.

    Parameters
    ----------
    template_list : PathOrStr
        Path to the CSV file containing metadata for all templates and analyses.
    indices : list of int
        List of row indices in the template list to process.
    parent_folder_path : PathOrStr
        Path to the root directory containing structure data and score maps.

    Notes
    -----
    - Only rows marked as 'Done == True' in the template list are processed.
    - Structures named ``'membrane'`` are skipped.
    - peak shapes are measured using `tmana.evaluate_scores_map` with
      three thresholding methods:
      triangle, Gaussian, and hard threshold.
    - The three principal dimensions (x, y, z) of each peak shape are stored in
      the template list under the columns:
      'TP x/y/z', 'GP x/y/z', and 'HP x/y/z' (triangle, Gaussian, hard).
    - The maximum peak value is stored under 'Peak value'.
    - A PNG plot visualizing the scores and peaks is saved in the corresponding
      output folder under the name 'peaks.png'.
    """

    temp_df = pd.read_csv(template_list, index_col=0)

    for i in indices:
        if not temp_df.at[i, "Done"]:
            continue

        if temp_df.at[i, "Structure"] == "membrane":
            continue

        structure_name = temp_df.at[i, "Structure"]

        output_base = create_output_base_name(i)
        output_folder = create_output_folder_path(parent_folder_path, structure_name, temp_df.at[i, "Output folder"])
        scores_map = cryomap.read(output_folder + output_base + "_scores.em")

        shapes = peak_shapes(scores_map)

        for t, p in enumerate(["x", "y", "z"]):
            temp_df.at[i, "TP " + p] = np.round(shapes["tp_shape"][t], 3)
            temp_df.at[i, "GP " + p] = np.round(shapes["gp_shape"][t], 3)
            temp_df.at[i, "HP " + p] = np.round(shapes["hp_shape"][t], 3)

        temp_df.at[i, "Peak value"] = shapes["peak_value"]

        temp_df.to_csv(template_list)  # save what was finished in case of a crush

        visplot.plot_scores_and_peaks(
            [
                scores_map,
                shapes["t_th_map"],
                shapes["t_surf"],
                shapes["t_map"],
                shapes["g_th_map"],
                shapes["g_surf"],
                shapes["g_map"],
                shapes["h_th_map"],
                shapes["h_surf"],
                shapes["h_map"],
            ],
            plot_title=structure_name + " id" + str(i),
            output_path=output_folder + "peaks.png",
        )


## Function to change the output folder base name
def rename_folders(template_list: PathOrStr, indices: list[int], parent_folder_path: PathOrStr) -> None:
    """
    Rename output folders for specified dataset entries and update metadata.

    This function updates the output folder names for given indices in a template
    list CSV file. Each folder is renamed to a new standardized name generated
    from its index, and the corresponding entry in the CSV file is updated to
    reflect the change.

    Parameters
    ----------
    template_list : PathOrStr
        Path to the CSV file containing metadata for all templates and analyses.
    indices : iterable of int
        List or array of row indices in `template_list` to process.
    parent_folder_path : PathOrStr
        Base directory containing all structure folders.

    Notes
    -----
    - Uses 'create_structure_path' to locate the parent folder for each structure.
    - Uses 'create_output_folder_name' to generate a new standardized folder name
      based on the index.
    """

    temp_df = pd.read_csv(template_list, index_col=0)

    for i in indices:
        structure_path = create_structure_path(parent_folder_path, temp_df.at[i, "Structure"])
        current_folder_path = structure_path + temp_df.at[i, "Output folder"]
        new_folder_name = create_output_folder_name(i)
        new_folder_path = structure_path + new_folder_name

        os.rename(current_folder_path, new_folder_path)
        temp_df.at[i, "Output folder"] = new_folder_name
        temp_df.to_csv(template_list)  # to save what was finished in case of a crush


# Function to change the names of TM results -> facilitate reading later on
def rename_scores_angles(template_list: PathOrStr, indices: list[int], parent_folder_path: PathOrStr) -> None:
    """
    Rename score and angle-related output files for specified dataset entries.

    This function updates the filenames of score and angular analysis files for
    given indices in a template list CSV file. The files are renamed to a new
    standardized base name derived from the entry's index. File renaming is done
    in the filesystem without altering the CSV metadata.

    Parameters
    ----------
    template_list : PathOrStr
        Path to the CSV file containing metadata for all templates and analyses.
    indices : iterable of int
        List or array of row indices in `template_list` to process.
    parent_folder_path : PathOrStr
        Base directory containing all structure folders.

    Notes
    -----
    - The base name pattern of the old files depends on the value of the
      'Compare' column:

        * `"tmpl"` → `"tt_" + Map type`

        * `"subtomo"` → `"ts_t<tomogram_number>_" + Map type`

        * other → `"td_<Compare>_" + Map type`

    - The following files are renamed with new base name for each entry:

        * '<base>.csv'

        * '<base>_scores.em'

        * '<base>_angles.em'

        * '<base>_angles_dist_all.em'

        * '<base>_angles_dist_normals.em'

        * '<base>_angles_dist_inplane.em'

    """

    temp_df = pd.read_csv(template_list, index_col=0)

    for i in indices:
        structure_name = temp_df.at[i, "Structure"]

        if temp_df.at[i, "Compare"] == "tmpl":
            comp_type = "tt_"
        elif temp_df.at[i, "Compare"] == "subtomo":
            tomo = create_em_path(parent_folder_path, structure_name, temp_df.at[i, "Tomo map"])
            tomo_number = re.findall(r"\d+", temp_df.at[i, "Tomogram"])[0]
            comp_type = "ts_t" + tomo_number + "_"
        else:
            tomo = create_em_path(parent_folder_path, temp_df.at[i, "Compare"], temp_df.at[i, "Tomo map"])
            comp_type = "td_" + temp_df.at[i, "Compare"] + "_"

        output_base = comp_type + temp_df.at[i, "Map type"]
        new_base = create_output_base_name(i)
        output_folder = create_output_folder_path(parent_folder_path, structure_name, temp_df.at[i, "Output folder"])
        csv_file = output_folder + output_base + ".csv"
        scores_map = output_folder + output_base + "_scores.em"
        angles_map1 = output_folder + output_base + "_angles.em"
        angles_map2 = output_folder + output_base + "_angles_dist_all.em"
        angles_map3 = output_folder + output_base + "_angles_dist_normals.em"
        angles_map4 = output_folder + output_base + "_angles_dist_inplane.em"

        new_csv_file = output_folder + new_base + ".csv"
        new_scores_map = output_folder + new_base + "_scores.em"
        new_angles_map1 = output_folder + new_base + "_angles.em"
        new_angles_map2 = output_folder + new_base + "_angles_dist_all.em"
        new_angles_map3 = output_folder + new_base + "_angles_dist_normals.em"
        new_angles_map4 = output_folder + new_base + "_angles_dist_inplane.em"

        os.rename(csv_file, new_csv_file)
        os.rename(scores_map, new_scores_map)
        os.rename(angles_map1, new_angles_map1)
        os.rename(angles_map2, new_angles_map2)
        os.rename(angles_map3, new_angles_map3)
        os.rename(angles_map4, new_angles_map4)


def correct_bbox(template_list: PathOrStr, indices: list[int]) -> None:
    """
    Increment specific bounding box-related columns by 1 for completed entries.

    This function reads a CSV file containing template metadata and, for the specified
    row indices where the "Done" column is True, increments the values in several
    bounding box-related columns by 1 along each spatial dimension ("x", "y", "z").
    The updated DataFrame is saved back to the same CSV file after all corrections.

    Parameters
    ----------
    template_list : PathOrStr
        Path to the CSV file containing metadata for all templates and analyses.
        The file must include the following columns:
        - "Dim x", "Dim y", "Dim z"
        - "O dist_all x", "O dist_all y", "O dist_all z"
        - "O dist_normals x", "O dist_normals y", "O dist_normals z"
        - "O dist_inplane x", "O dist_inplane y", "O dist_inplane z"
    indices : iterable of int
        List or array of row indices to process within the CSV file.
    """

    temp_df = pd.read_csv(template_list, index_col=0)

    for i in indices:
        if not temp_df.at[i, "Done"]:
            continue

        list_to_correct = ["Dim", "O dist_all", "O dist_normals", "O dist_inplane"]

        for l in list_to_correct:
            for d in ["x", "y", "z"]:
                temp_df.at[i, l + " " + d] += 1

        temp_df.to_csv(template_list)


def recompute_dist_maps(
    template_list: PathOrStr, indices: list[int], parent_folder_path: PathOrStr, angle_list_path: PathOrStr
) -> None:
    """
    Recompute angular distance maps for specified entries in a template list.

    This function reads a CSV file containing template metadata, and for each specified
    index where the "Done" flag is True, it recalculates angular distance maps using
    corresponding angle files. The updated maps are written to disk.

    Parameters
    ----------
    template_list : PathOrStr
        Path to the CSV file containing metadata for all templates and analyses.
    indices : iterable of int
        List or array of row indices in the CSV file to process.
    parent_folder_path : PathOrStr
        Base path to the parent folder where structure and output folders are.
    angle_list_path : PathOrStr
        Base path to the directory containing angle list files referenced in the CSV.
    """

    temp_df = pd.read_csv(template_list, index_col=0)

    for i in indices:
        if not temp_df.at[i, "Done"]:
            continue

        structure_name = temp_df.at[i, "Structure"]

        output_base = create_output_base_name(i)
        output_folder = create_output_folder_path(parent_folder_path, structure_name, temp_df.at[i, "Output folder"])
        angles_map = output_folder + output_base + "_angles.em"
        angle_list = angle_list_path + temp_df.at[i, "Angles"]
        cyclic_symmetry = temp_df.at[i, "Symmetry"]
        _, _, _ = tmana.create_angular_distance_maps(
            angles_map, angle_list, write_out_maps=True, cyclic_symmetry=cyclic_symmetry
        )
