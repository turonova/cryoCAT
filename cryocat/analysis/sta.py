import copy
import math
import re
import warnings
from dataclasses import dataclass as _dataclass, field as _field
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Literal, get_args

from scipy.sparse import csr_matrix

from cryocat.utils.exceptions import UserInputError

import numpy as np
import pandas as pd
from cryocat.core import cryomotl
from cryocat.core.cryomotl import MotlSource
from cryocat.utils import geom
from cryocat.utils import mathutils
from cryocat.analysis import visplot
from cryocat.utils import ioutils
from cryocat.utils.starfileio import Starfile
from cryocat._types import ListLike, MotlColumn, MotlType, PathOrStr


def get_stable_particles(
    motl_base_name: str,
    start_it: int,
    end_it: int,
    motl_type: MotlType = "emmotl",
    load_kwargs: dict | None = None,
) -> list:
    """Load and analyze particle data across multiple iterations to identify stable particles, i.e. particles that do
    not change their class.

    Parameters
    ----------
    motl_base_name : str
        Base name for a motl to perform the evaluation on. Base name means without the
        iteration number and extension. For example for name motl_shift_3.em the base name is motl\_shift\_.
    start_it : int
        Starting iteration number.
    end_it : int
        Ending iteration number.
    motl_type : MotlType, default='emmotl'
        Type of the input motl.  One of the standard motl-format identifiers
        (``'emmotl'``, ``'stopgap'``, ``'relion'``, ``'relion5'``, ``'relion5_1'``).
    load_kwargs : dict or None, default=None
        Dictionary of keyword arguments passed to the `Motl.load` method (and subsequently to the underlying
        Motl class constructors like 'RelionMotl' and `RelionMotlv5`). This is useful for providing necessary metadata like
        `pixel_size`, `binning`, `optics_data`, or custom formats (`tomo_format`, `subtomo_format`).

    Returns
    -------
    list
        List of subtomo_ids that have the same class across the specified iterations.

    Notes
    -----
    This function loads motive list files from specified iterations, merges them, and identifies
    subtomo_ids (subtomogram identifiers) that have a consistent class across all iterations.
    The percentage of stable particles relative to the total number of particles in the first
    iteration is printed.
    """

    load_kwargs = load_kwargs or {}
    dfs = []
    for i in np.arange(start_it, end_it + 1):
        filename = get_motl_filename(motl_base_name, i, motl_type)
        m = cryomotl.Motl.load(filename, motl_type=motl_type, **load_kwargs)
        dfs.append(m.df)

    # Merge the dataframes on 's_id' column
    merged_df = pd.concat(dfs, axis=0, ignore_index=True)

    # Check if the 'class' column is the same for all frames
    same_class_mask = merged_df.groupby("subtomo_id")["class"].nunique() == 1

    # Get the s_ids where the class is the same for all frames
    common_subtomo_ids = same_class_mask[same_class_mask].index.tolist()

    # Get percentage
    print(
        f"The number of stable particles is {len(common_subtomo_ids)} which corresponds to {len(common_subtomo_ids)/dfs[0].shape[0]*100:.2f}%"
    )
    return common_subtomo_ids


def evaluate_alignment(
    motl_base_names: ListLike[str],
    start_it: int,
    end_it: int,
    motl_type: MotlType = "stopgap",
    write_out_stats: bool = False,
    plot_values: bool = True,
    filter_rows: ListLike | None = None,
    filter_column_name: ListLike[MotlColumn] = "subtomo_id",
    labels: ListLike[str] | None = None,
    graph_title: str = "Alignment stability",
    graph_output_file: PathOrStr | None = None,
    load_kwargs: dict | None = None,
) -> list:
    """Evaluate alignment stability for specified motls and iterations.

    Parameters
    ----------
    motl_base_names : ListLike of str
        List of MOTL base names or a single motl base name to perform the evaluation on. Base name means without the
        iteration number and extension. For example for name motl_shift_3.em the base name is motl_shift\_.
    start_it : int
        Starting iteration number.
    end_it : int
        Ending iteration number.
    motl_type : MotlType, default='stopgap'
        Type of the input motl.
    write_out_stats : bool, default=False
        Whether to write out stats. If True, the stats will be written to the motl_base_name + _as_motlID.csv where the
        motlID is given by its position in the motl_base_names list. For example, for motl_shift_3.em the final will
        be motl_shift_as_1.em if the motl_shift\_ is the first motl in the motl_base_names.
    plot_values : bool, default=True
        Whether to plot values.
    filter_rows : ListLike or None, default=None
        Rows to filter. Only rows that are within the filter_rows will be kept. ``None`` means no filtering.
    filter_column_name : ListLike of MotlColumn, default='subtomo_id'
        Column name(s) based on which the filtering is performed. Ignored when ``filter_rows`` is ``None``.
    labels : ListLike of str or None, default=None
        Labels for the plot. Should have the same length as the motl_base_names. In case of ``None``, the labels will
        be automatically set as motl_base_names (paths are stripped).  Used only if ``plot_values`` is True.
    graph_title : str, default='Alignment stability'
        Title of the graph. Used only if ``plot_values`` is True.
    graph_output_file : PathOrStr or None, default=None
        Output file for the graph. Used only if ``plot_values`` is True. If ``None`` no file will be written out.
    load_kwargs : dict or None, default=None
        Dictionary of keyword arguments passed to the `Motl.load` method (and subsequently to the underlying
        Motl class constructors like 'RelionMotl' and `RelionMotlv5`). This is useful for providing necessary metadata like
        `pixel_size`, `binning`, `optics_data`, or custom formats (`tomo_format`, `subtomo_format`).

    Returns
    -------
    list of pandas DataFrames
        List of computed alignment stability statistics dataframes.

    Examples
    --------
    >>> # Single motl, no filtering, motls motl_1.star to motl_17.star will be loaded for evaluation. Statistics
    >>> # will be written into /path/to/the/motl_as_1.csv file.
    >>> motl_base_name = "/path/to/the/motl_"
    >>> stats_df = evaluate_alignment(motl_base_name, 1, 17, motl_type="stopgap", plot_values=True,write_out_stats=True)

    >>> # Multiple motls, no filtering, motls motl1_1.star to motl1_17.star and motl3_1.star to motl3_17.star
    >>> # will be loaded for evaluation. Statistics will be written into /path/to/the/motl1_as_1.csv and
    >>> # /path/to/the/motl3_as_2.csv files.
    >>> motl_base_names = ["/path/to/the/motl1_", "/path/to/the/motl3_"]
    >>> stats_df = evaluate_alignment(motl_base_names, 1, 17, motl_type="stopgap", plot_values=True, write_out_stats=True)

    >>> # Multiple motls, motls motl1_1.star to motl1_17.star and motl3_1.star to motl3_17.star will be loaded for
    >>> # evaluation. Statistics will be written into /path/to/the/motl1_as_1.csv and /path/to/the/motl3_as_2.csv files.
    >>> # Filtering will be done based on column geom3 and only particles with values in filter_rows will be evaluated.
    >>> motl_base_names = ["/path/to/the/motl1_", "/path/to/the/motl3_"]
    >>> filter_rows = [values_to_keep_for_motl1, values_to_keep_for_motl3]
    >>> stats_df = evaluate_alignment(
    ...     motl_base_names, 1, 17,
    ...     filter_rows=filter_rows, filter_column_name="geom3",
    ...     motl_type="stopgap", plot_values=True, write_out_stats=True
    ... )

    >>> # Multiple motls, motls motl1_1.star to motl1_17.star and motl3_1.star to motl3_17.star will be loaded for
    >>> # evaluation. Statistics will be written into /path/to/the/motl1_as_1.csv and /path/to/the/motl3_as_2.csv files.
    >>> # Filtering will be done based on column geom3 for motl1 and based on subtomo_id for motl3.
    >>> # Only particles with values in filter_rows will be evaluated.
    >>> motl_base_names = ["/path/to/the/motl1_", "/path/to/the/motl3_"]
    >>> filter_rows = [values_to_keep_for_motl1, values_to_keep_for_motl3]
    >>> filter_column_name = ["geom3", "subtomo_id"]
    >>> stats_df = evaluate_alignment(
    ...     motl_base_names, 1, 17,
    ...     filter_rows=filter_rows, filter_column_name=filter_column_name,
    ...     motl_type="stopgap", plot_values=True, write_out_stats=True
    ... )

    >>> # Multiple motls, motls motl1_1.star to motl1_17.star and motl3_1.star to motl3_17.star will be loaded for
    >>> # evaluation. Statistics will be written into /path/to/the/motl1_as_1.csv and /path/to/the/motl3_as_2.csv files.
    >>> # Filtering will be done based on column geom3 for motl1 and no filtering will be done for motl3.
    >>> # Only particles with values in filter_rows will be evaluated.
    >>> motl_base_names = ["/path/to/the/motl1_", "/path/to/the/motl3_"]
    >>> filter_rows = [values_to_keep_for_motl1, None]
    >>> filter_column_name = ["geom3", None]
    >>> stats_df = evaluate_alignment(
    ...     motl_base_names, 1, 17,
    ...     filter_rows=filter_rows, filter_column_name=filter_column_name,
    ...     motl_type="stopgap", plot_values=True, write_out_stats=True
    ... )
    """

    if not isinstance(motl_base_names, list):
        motl_base_names = [motl_base_names]
    # ensure correct input formats in case there is only one filter_rows and filter_column specified
    if not isinstance(filter_column_name, list):
        filter_column_name = [filter_column_name]
        if len(motl_base_names) > 1 and len(filter_column_name) == 1:
            filter_column_name = filter_column_name * len(motl_base_names)

    if filter_rows is None:
        filter_rows = [None] * len(motl_base_names)
    # filter_rows = np.full((1, len(motl_base_names)), None)
    elif not isinstance(filter_rows, list):
        filter_rows = [filter_rows]
        if len(filter_rows) != len(motl_base_names) and len(filter_rows) == 1:
            filter_rows = filter_rows * len(motl_base_names)

    stats_dfs = []
    load_kwargs = load_kwargs or {}
    for i, m in enumerate(motl_base_names):
        if write_out_stats:
            stats_file_name = m + f"as_{str(i+1)}.csv"
        else:
            stats_file_name = None
        stats_dfs.append(
            compute_alignment_statistics(
                m,
                start_it,
                end_it,
                motl_type=motl_type,
                filter_rows=filter_rows[i],
                filter_column_name=filter_column_name[i],
                output_path=stats_file_name,
                load_kwargs=load_kwargs,
            )
        )

    if plot_values:
        if labels is None:
            labels = [ioutils.get_filename_from_path(m)[0:-1] for m in motl_base_names]
        visplot.plot_alignment_stability(
            stats_dfs, labels=labels, graph_title=graph_title, output_path=graph_output_file
        )

    return stats_dfs


def get_motl_extension(motl_type: MotlType) -> str:
    """Return the file extension for a given motl type.

    Parameters
    ----------
    motl_type : MotlType
        The type of motl file (``'emmotl'``, ``'relion'``, ``'relion5'``,
        ``'relion5_1'``, or ``'stopgap'``).

    Returns
    -------
    str
        The file extension corresponding to the motl type.

    Raises
    ------
    ValueError
        If the motl type is not supported.
    """

    if motl_type in ["stopgap", "relion", "relion5", "relion5_1"]:
        motl_ext = ".star"
    elif motl_type == "emmotl":
        motl_ext = ".em"
    else:
        raise ValueError(f"The motl type {motl_type} is not currently supported.")
    return motl_ext


def compute_alignment_statistics(
    motl_base_name: str,
    start_it: int,
    end_it: int,
    motl_type: MotlType = "stopgap",
    filter_rows: ListLike | None = None,
    filter_column_name: MotlColumn = "subtomo_id",
    output_path: PathOrStr | None = None,
    load_kwargs: dict | None = None,
) -> pd.DataFrame:
    """Compute alignment statistics for specified motls and iterations. Pairs of (current motl, subsequent motl) are
    evaluated for differences in cone angles, in-plane angles, change in positions of particles and root mean square
    errors (RMSE) in x, y, and z directions. The output contains mean, median, std, and variance for cone and in-plane
    angles, the mean distance between the particles and the RMSE of movement in x, y, and z directions.

    Parameters
    ----------
    motl_base_name : str
        Base name for a motl to perform the evaluation on. Base name means without the
        iteration number and extension. For example for name motl_shift_3.em the base name is motl\_shift\_.
    start_it : int
        Starting iteration number.
    end_it : int
        Ending iteration number.
    motl_type : MotlType, default='stopgap'
        Type of the input motl.
    filter_rows : ListLike or None, default=None
        Rows to filter. Only rows that are within the filter_rows will be kept. ``None`` means no filtering.
    filter_column_name : MotlColumn, default='subtomo_id'
        Column based on which the filtering is performed. Ignored when ``filter_rows`` is ``None``.
    output_path : PathOrStr or None, default=None
        Output file for the statistics. If ``None`` no file will be written out.
    load_kwargs : dict or None, default=None
        Dictionary of keyword arguments passed to the `Motl.load` method (and subsequently to the underlying
        Motl class constructors like 'RelionMotl' and `RelionMotlv5`). This is useful for providing necessary metadata like
        `pixel_size`, `binning`, `optics_data`, or custom formats (`tomo_format`, `subtomo_format`).

    Returns
    -------
    pandas DataFrame
        Comptuted statistics of the alignment for the specified iterations.

    Examples
    --------
    >>> # No filtering, motls motl_1.star to motl_17.star will be loaded for evaluation. Statistics
    >>> # will be written into /path/to/the/motl_alignment_stats.csv file.
    >>> stats_df = compute_alignment_statistics(
    ...    "/path/to/the/motl_", 1, 17,
    ...     motl_type="stopgap", output_path="/path/to/the/motl_alignment_stats.csv"
    ... )

    >>> # Motls motl_1.star to motl_17.star will be loaded for evaluation, no file will be written out.
    >>> # Filtering will be done based on column geom3 and only particles with values in filter_rows will be evaluated.
    >>> stats_df = compute_alignment_statistics(
    ...     "/path/to/the/motl_", 1, 17,
    ...     filter_rows=values_to_keep_for_motl, filter_column_name="geom3",
    ...     motl_type="stopgap"
    ... )
    """

    stats_df = pd.DataFrame(
        columns=[
            "cone_mean",
            "cone_median",
            "cone_std",
            "cone_var",
            "plane_mean",
            "plane_median",
            "plane_std",
            "plane_var",
            "position_change",
            "rmse_x",
            "rmse_y",
            "rmse_z",
        ]
    )

    # Repeat the empty DataFrame to the desired length
    stats_df = pd.concat([stats_df] * (end_it - start_it + 1), ignore_index=True)

    # load motls
    motls = []
    load_kwargs = load_kwargs or {}
    for i in np.arange(start_it, end_it + 1):
        filename = get_motl_filename(motl_base_name, i, motl_type)
        m = cryomotl.Motl.load(filename, motl_type=motl_type, **load_kwargs)
        if filter_rows is not None:
            m.df = m.df[m.df[filter_column_name].isin(filter_rows)]
        motls.append(m)

    for i in np.arange(
        0, end_it - start_it
    ):  ## FIXME this fixes 'index out of range' when start_it=!0, but does not account for the correct plot labels in such case (when called by evaluate_alignment)
        current_rot = motls[i].get_rotations()
        next_rot = motls[i + 1].get_rotations()

        current_coord = motls[i].get_coordinates()
        next_coord = motls[i + 1].get_coordinates()
        point_distances = geom.point_pairwise_dist(current_coord, next_coord)

        diff_cone, diff_plane = geom.cone_inplane_distance(current_rot, next_rot)
        stats_df.at[i, "cone_mean"] = np.mean(diff_cone)
        stats_df.at[i, "cone_std"] = np.std(diff_cone)
        stats_df.at[i, "cone_var"] = np.var(diff_cone)
        stats_df.at[i, "cone_median"] = np.median(diff_cone)
        stats_df.at[i, "plane_mean"] = np.mean(diff_plane)
        stats_df.at[i, "plane_std"] = np.std(diff_plane)
        stats_df.at[i, "plane_var"] = np.var(diff_plane)
        stats_df.at[i, "plane_median"] = np.median(diff_plane)
        stats_df.at[i, "position_change"] = np.mean(point_distances)
        stats_df.loc[i, ["rmse_x", "rmse_y", "rmse_z"]] = mathutils.compute_rmse(current_coord, next_coord)

    if output_path is not None:
        stats_df.to_csv(output_path, index=False)

    return stats_df


def write_out_motl(
    input_motl: MotlSource,
    output_file_base: PathOrStr,
    output_motl_type: str,
) -> None:
    """Writes out a given motl file to a specified output format.

    Parameters
    ----------
    input_motl : MotlSource
        Input motl to be written out (a :class:`Motl`, a DataFrame, or a path).
        Only the ``.df`` attribute is read.
    output_file_base : PathOrStr
        Base path for the output file (extension is appended automatically).
    output_motl_type : str
        Type of the output motl file -- one of ``'stopgap'``, ``'relion'``,
        ``'relion5'``, ``'relion5_1'``, ``'emfile'``.  ``'emfile'`` is
        accepted here for historical reasons and maps to an :class:`EmMotl`
        write; for new code prefer the standard :data:`MotlType` enum value
        ``'emmotl'`` via the :class:`cryocat.core.cryomotl.Motl` API.

    Raises
    ------
    ValueError
        If the output_motl_type is not one of the supported types.

    Returns
    -------
    None
    """

    if output_motl_type == "stopgap":
        final_motl = cryomotl.StopgapMotl(input_motl.df)
        final_motl.write_out(output_file_base + ".star", reset_index=True)
    elif output_motl_type == "relion":
        final_motl = cryomotl.RelionMotl(input_motl.df)
        final_motl.write_out(output_file_base + ".star")
    elif output_motl_type == "relion5":
        final_motl = cryomotl.RelionMotlv5(input_motl.df)
        final_motl.write_out(output_file_base + ".star")
    elif output_motl_type == "relion5_1":
        final_motl = cryomotl.RelionMotlv5_1(input_motl.df)
        final_motl.write_out(output_file_base + ".star")
    elif output_motl_type == "emfile":
        final_motl = cryomotl.EmMotl(input_motl.df)
        final_motl.write_out(output_file_base + ".em")
    else:
        raise ValueError(f"The output motl type {output_motl_type} is not supported.")


_MOTL_EXT_MAP: dict[str, str] = {
    "stopgap": ".star",
    "relion": ".star",
    "relion5": ".star",
    "relion5_1": ".star",
    "emmotl": ".em",
    "emfile": ".em",
}


def _motl_file_ext(output_motl_type: str) -> str:
    """Return the file extension written by write_out_motl for *output_motl_type*."""
    return _MOTL_EXT_MAP.get(output_motl_type, ".star")


def create_multiref_run(
    input_motl: MotlSource,
    number_of_classes: int,
    output_motl_base: PathOrStr,
    input_motl_type: MotlType = "emmotl",
    iteration_number: int = 1,
    number_of_runs: int = 1,
    output_motl_type: str = "stopgap",
) -> list[Path]:
    """Creates motls for multiple runs of a multi-reference alignment. In essence, it will randomly assign specified number
    of classes to each motl that will be created. New motls will be written out into files
    output_motl_base_mr#runID_iterationNumber either in stopgap, emmotl or relion format.

    Parameters
    ----------
    input_motl : MotlSource
        Input motl (specified either as a path, DataFrame or Motl object).
    number_of_classes : int
        Number of classes to assign randomly.
    output_motl_base : PathOrStr
        Base path for the output motl files. The final name will be created as output_motl_base_mr#runID_iterationNumber
        where runID is from 1 to number_of_runs and iterationNumber is iteration_number. The extension is determined
        by ``output_motl_type``.
    input_motl_type : MotlType, default='emmotl'
        Type of the input motl file.
    iteration_number : int, default=1
        Iteration number to be used in the output name creation.
    number_of_runs : int, default=1
        Number of motls to create.
    output_motl_type : str, default='stopgap'
        Type of the output motl file (see :func:`write_out_motl` for accepted values).

    Returns
    -------
    list[Path]
        Paths of every motl file written (one per run).

    Examples
    --------
    >>> # Will create two motls in stopgap format with names stopgap_classes_mr1_4.star and stopgap_classes_mr2_4.star
    >>> create_multiref_run(
    ... "/path/to/relion_1.star", number_of_classes=8, output_motl_base="stopgap_classes",
    ... input_motl_type="relion", iteration_number=4, number_of_runs=2,
    ... output_motl_type="stopgap"
    ... )
    """

    motl = cryomotl.Motl.load(input_motl, motl_type=input_motl_type)
    motl.df = motl.df.fillna(0.0)

    ext = _motl_file_ext(output_motl_type)
    created_files: list[Path] = []
    for i in range(1, number_of_runs + 1):
        # create motl with randomly assigned classes
        motl.assign_random_classes(number_of_classes)

        output_path = output_motl_base + "_mr" + str(i) + "_" + str(iteration_number)
        write_out_motl(motl, output_path, output_motl_type=output_motl_type)
        created_files.append(Path(output_path + ext))

    print(
        f"create_multiref_run: wrote {number_of_runs} motl(s) with {number_of_classes} classes:\n"
        + "\n".join(f"  {p}" for p in created_files)
    )
    return created_files


def create_denovo_multiref_run(
    input_motl: MotlSource,
    number_of_classes: int,
    output_motl_base: PathOrStr,
    input_motl_type: MotlType = "emmotl",
    class_occupancy: int | None = None,
    iteration_number: int = 1,
    number_of_runs: int = 1,
    output_motl_type: str = "stopgap",
) -> list[Path]:
    """Creates number_of_runs motls for reference averaging and one motl for alignment. The motls for reference averaging
    are created by random selection of N particles for each class from the input_motl, where N equals to class_occupancy.
    The particles within the classes of each motl can overlap, i.e. each class will have a unique set of particles, but
    some particles can be assigned in mutliple classes. The alignment motl is just input motl where the class was
    randomly assign to be from 1 to number_of_classes. The idea behind this is to run multi-reference alignment
    where different runs will have different starting references while due to simmulated annealing only one motl
    for alignment is needed afterwards.

    Parameters
    ----------
    input_motl : MotlSource
        Input motl (specified either as a path, DataFrame or Motl object).
    number_of_classes : int
        Number of classes to create references for and to assign randomly to the alignment motl.
    output_motl_base : PathOrStr
        Base path for the output motl files. The final name will be created as output_motl_base_ref_mr#runID_iterationNumber
        where runID is from 1 to number_of_runs and iterationNumber is iteration_number. The alignment motl will be named
        output_motl_base_iterationNumber. Extension is determined by ``output_motl_type``.
    input_motl_type : MotlType, default='emmotl'
        Type of the input motl file.
    class_occupancy : int or None, default=None
        Number of particles per class for the reference averaging motls. If ``None``, the number is determined as 1/10
        of total number of particles in the input motl.
    iteration_number : int, default=1
        Iteration number to be used in the output name creation.
    number_of_runs : int, default=1
        Number of motls to create.
    output_motl_type : str, default='stopgap'
        Type of the output motl file (see :func:`write_out_motl` for accepted values).

    Returns
    -------
    list[Path]
        Paths of every motl file written: ``number_of_runs`` reference motls followed by
        the single shared alignment motl.

    Examples
    --------
    >>> # Will create two motls in stopgap format with names stopgap_dn_ref_mr1_4.star and stopgap_dn_ref_mr2_4.star for
    >>> # reference averaging and one alignment motl stopgap_dn_4.star. In each motl, the particles will have 8 classes.
    >>> # The alignment motl will have same number of particles as the input_motl, the reference motls will have
    >>> # number_of_classes * class_occupancy (16 000) particles each.
    >>> create_denovo_multiref_run(
    ... "/path/to/relion_1.star", number_of_classes=8, output_motl_base="stopgap_dn",
    ... input_motl_type="relion", class_occupancy = 2000, iteration_number=4, number_of_runs=2,
    ... output_motl_type="stopgap"
    ... )
    """

    motl = cryomotl.Motl.load(input_motl, motl_type=input_motl_type)
    motl.df = motl.df.fillna(0.0)

    n_particles = motl.df.shape[0]

    # create motl for reference creation
    if class_occupancy is None:
        class_occupancy = int(np.ceil(n_particles / 10))

    ext = _motl_file_ext(output_motl_type)
    created_files: list[Path] = []
    for i in range(1, number_of_runs + 1):
        ref_df = pd.DataFrame()
        for c in range(1, number_of_classes + 1):
            class_motl = motl.get_random_subset(class_occupancy)
            class_motl.df["class"] = c
            ref_df = pd.concat([ref_df, class_motl.df], ignore_index=True)

        ref_df.reset_index(inplace=True, drop=True)
        new_motl = cryomotl.Motl(ref_df)
        output_path = output_motl_base + "_ref_mr" + str(i) + "_" + str(iteration_number)
        write_out_motl(new_motl, output_file_base=output_path, output_motl_type=output_motl_type)
        created_files.append(Path(output_path + ext))

    # create motl with randomly assigned classes
    ali_path = output_motl_base + "_" + str(iteration_number)
    motl.assign_random_classes(number_of_classes)
    write_out_motl(motl, output_file_base=ali_path, output_motl_type=output_motl_type)
    created_files.append(Path(ali_path + ext))

    print(
        f"create_denovo_multiref_run: wrote {number_of_runs} reference motl(s) + 1 alignment motl:\n"
        + "\n".join(f"  {p}" for p in created_files)
    )
    return created_files


def evaluate_multirun_stability(
    input_motls: list[MotlSource],
    input_motl_type: MotlType = "stopgap",
) -> dict:
    """Evaluate how many particles ended up within the same class among all the classification runs. It is meant to be
    used for multiruns with existing references (i.e. not de novo ones) where all runs uses the same references in the
    same order.

    Parameters
    ----------
    input_motls : list of MotlSource
        List of input motls (paths, DataFrames, or Motl objects).  At least two are required.
    input_motl_type : MotlType, default='stopgap'
        Type of the input motl.

    Returns
    -------
    common_occupancies : dict
        A dictionary containing common subtomo_ids for each class of particles.
    """

    dfs = []
    for i in input_motls:
        motl = cryomotl.Motl.load(i, motl_type=input_motl_type)
        dfs.append(motl.df)

    if len(dfs) < 2:
        raise ValueError("At least 2 motls are required.")

    unique_classes = dfs[0]["class"].unique()  # Identify unique classes from the first frame

    common_occupancies = {}

    for class_val in sorted(unique_classes):
        class_dfs = [df.loc[df["class"] == class_val, "subtomo_id"] for df in dfs]
        common_ids = set.intersection(*map(set, class_dfs))  # Find common subtomo_ids

        percentage = []
        for df in class_dfs:
            percentage.append(len(common_ids) / len(df) * 100)

        print(
            f"Class {class_val} has {len(common_ids)} stable particles which corresponds to {[f'{perc:.2f}' for perc in percentage]}% of provided motls."
        )

        common_occupancies[class_val] = sorted(common_ids)

    return common_occupancies


def get_subtomos_class_stability(
    motl_base_name: str,
    start_it: int,
    end_it: int,
    motl_type: MotlType = "stopgap",
    load_kwargs: dict | None = None,
) -> dict:
    """Calculate the class stability of subtomograms over iterations.

    Parameters
    ----------
    motl_base_name : str
        Base name for a motl to perform the evaluation on. Base name means without the
        iteration number and extension. For example for name motl_shift_3.em the base name is motl\_shift\_.
    start_it : int
        Starting iteration number.
    end_it : int
        Ending iteration number.
    motl_type : MotlType, default='stopgap'
        Type of the input motl.
    load_kwargs : dict or None, default=None
        Dictionary of keyword arguments passed to the `Motl.load` method (and subsequently to the underlying
        Motl class constructors like 'RelionMotl' and `RelionMotlv5`). This is useful for providing necessary metadata like
        `pixel_size`, `binning`, `optics_data`, or custom formats (`tomo_format`, `subtomo_format`).

    Returns
    -------
    different_sids : dict
        A dictionary containing the number of different subtomogram IDs for each class over iterations.

    Notes
    -----
    Loading of many motls can take some time. If you also want to compute occupancy of classes it is recommended to
    use :meth:`cryocat.analysis.sta.evaluate_classification` which gives both occupancy and stability and reads in all the motls
    only once.
    """

    dfs = []
    load_kwargs = load_kwargs or {}
    for i in np.arange(start_it, end_it + 1):
        filename = get_motl_filename(motl_base_name, i, motl_type)
        m = cryomotl.Motl.load(filename, motl_type=motl_type, **load_kwargs)
        dfs.append(m.df)

    # Concatenate the list of DataFrames into a single DataFrame
    changing_subtomos = {cls: [] for cls in dfs[0]["class"].unique()}
    for i in range(1, len(dfs)):
        previous_df = dfs[i - 1]
        current_df = dfs[i]
        for cls in changing_subtomos.keys():
            previous_sids = set(previous_df.loc[previous_df["class"] == cls, "subtomo_id"])
            current_sids = set(current_df.loc[current_df["class"] == cls, "subtomo_id"])
            num_different_sids = len(current_sids.difference(previous_sids))
            changing_subtomos[cls].append(num_different_sids)

    return changing_subtomos


def evaluate_classification(
    motl_base_name: str,
    start_it: int,
    end_it: int,
    motl_type: MotlType = "stopgap",
    output_file_stats: PathOrStr | None = None,
    plot_results: bool = False,
    output_file_graphs: PathOrStr | None = None,
    load_kwargs: dict | None = None,
) -> tuple[dict, dict]:
    """Get the occupancy of each class over the iterations and the class stability of subtomograms over iterations.

    Parameters
    ----------
    motl_base_name : str
        Base name for a motl to perform the evaluation on. Base name means without the
        iteration number and extension. For example for name motl_shift_3.em the base name is motl\_shift\_.
    start_it : int
        Starting iteration number.
    end_it : int
        Ending iteration number.
    motl_type : MotlType, default='stopgap'
        Type of the input motl.
    output_file_stats : PathOrStr or None, default=None
        Name of the file into which the results will be written out. If ``None``, no results will be written out.
    plot_results : bool, default=False
        Whether to plot the results.
    output_file_graphs : PathOrStr or None, default=None
        Name of the file into which the plotted graphs will be written out. If ``None``, the graphs will not be written out.
        Ignored when ``plot_results`` is False.
    load_kwargs : dict or None, default=None
        Dictionary of keyword arguments passed to the `Motl.load` method (and subsequently to the underlying
        Motl class constructors like 'RelionMotl' and `RelionMotlv5`). This is useful for providing necessary metadata like
        `pixel_size`, `binning`, `optics_data`, or custom formats (`tomo_format`, `subtomo_format`).

    Returns
    -------
    occupancy : dict
        A dictionary containing the occupancy of each class over the iterations.
    changing_subtomos : dict
        A dictionary containing the number of different subtomogram IDs for each class over iterations.
    """

    dfs = []
    load_kwargs = load_kwargs or {}
    for i in np.arange(start_it, end_it + 1):
        filename = get_motl_filename(motl_base_name, i, motl_type)
        m = cryomotl.Motl.load(filename, motl_type=motl_type, **load_kwargs)
        dfs.append(m.df)

    # Create a dictionary to store the occupancy of each class per dataframe
    occupancy = {}
    for i, df in enumerate(dfs):
        for c in df["class"].unique():
            if c not in occupancy:
                occupancy[c] = [0] * len(dfs)
            occupancy[c][i] = len(df[df["class"] == c])

    changing_subtomos = {cls: [] for cls in dfs[0]["class"].unique()}
    for i in range(1, len(dfs)):
        previous_df = dfs[i - 1]
        current_df = dfs[i]
        for cls in changing_subtomos.keys():
            previous_sids = set(previous_df.loc[previous_df["class"] == cls, "subtomo_id"])
            current_sids = set(current_df.loc[current_df["class"] == cls, "subtomo_id"])
            num_different_sids = len(current_sids.difference(previous_sids))
            changing_subtomos[cls].append(num_different_sids)

    # sort the dictionaries
    occupancy = dict(sorted(occupancy.items()))
    changing_subtomos = dict(sorted(changing_subtomos.items()))

    if plot_results:
        visplot.plot_classification_convergence(
            occupancy, changing_subtomos, graph_title="Classification progress", output_path=output_file_graphs
        )

    if output_file_stats is not None:
        occupancy_df = pd.DataFrame(occupancy)
        subtomos_df = pd.DataFrame(changing_subtomos)
        # Add a row of NaNs for the changes as at iteration one the numbers are no available
        nan_row = pd.Series([np.nan] * len(subtomos_df.columns), index=subtomos_df.columns)
        subtomos_df = pd.concat([pd.DataFrame([nan_row]), subtomos_df], ignore_index=True)

        it = pd.DataFrame({"#": range(1, occupancy_df.shape[0] + 1)})
        merged = pd.concat(
            [it, occupancy_df, subtomos_df], axis=1, keys=["Iteration", "Class occupancy", "Class changes"]
        )
        merged.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0] for col in merged.columns]
        merged.to_csv(output_file_stats, index=False)

    return occupancy, changing_subtomos


def get_class_occupancy(
    motl_base_name: str,
    start_it: int,
    end_it: int,
    motl_type: MotlType = "stopgap",
    load_kwargs: dict | None = None,
) -> dict:
    """Get the occupancy of each class over the iterations.

    Parameters
    ----------
    motl_base_name : str
        Base name for a motl to perform the evaluation on. Base name means without the
        iteration number and extension. For example for name motl_shift_3.em the base name is motl\_shift\_.
    start_it : int
        Starting iteration number.
    end_it : int
        Ending iteration number.
    motl_type : MotlType, default='stopgap'
        Type of the input motl.
    load_kwargs : dict or None, default=None
        Dictionary of keyword arguments passed to the `Motl.load` method (and subsequently to the underlying
        Motl class constructors like 'RelionMotl' and `RelionMotlv5`). This is useful for providing necessary metadata like
        `pixel_size`, `binning`, `optics_data`, or custom formats (`tomo_format`, `subtomo_format`).

    Returns
    -------
    occupancy : dict
        A dictionary containing the occupancy of each class over the iterations.

    Notes
    -----
    Loading of many motls can take some time. If you also want to compute stability of classes it is recommended to
    use :meth:`cryocat.analysis.sta.evaluate_classification` which gives both occupancy and stability and reads in all the motls
    only once.
    """
    load_kwargs = load_kwargs or {}
    dfs = []
    for i in np.arange(start_it, end_it + 1):
        filename = get_motl_filename(motl_base_name, i, motl_type)
        m = cryomotl.Motl.load(filename, motl_type=motl_type, **load_kwargs)
        dfs.append(m.df)

    # Create a dictionary to store the occupancy of each class per dataframe
    occupancy = {}
    for i, df in enumerate(dfs):
        for c in df["class"].unique():
            if c not in occupancy:
                occupancy[c] = [0] * len(dfs)
            occupancy[c][i] = len(df[df["class"] == c])

    return occupancy


def get_motl_filename(
    motl_base_name: str,
    iteration: int,
    motl_type: MotlType,
) -> str:
    """Construct the full filename for a motl file given a base name, iteration, and type.

    For Relion-type motls the iteration number is zero-padded to three digits
    and the suffix ``_data.star`` is appended.  For all other types the
    extension is determined by :func:`get_motl_extension`.

    Parameters
    ----------
    motl_base_name : str
        Base name for the motl file, without the iteration number or extension.
        For example, for ``motl_shift_3.em`` the base name is ``motl_shift_``.
    iteration : int
        Iteration number to embed in the filename.
    motl_type : MotlType
        Type of the motl file (e.g. ``'emmotl'``, ``'stopgap'``, ``'relion'``,
        ``'relion5'``, ``'relion5_1'``).  Any value containing ``'relion'``
        triggers zero-padded three-digit formatting.

    Returns
    -------
    str
        Full filename constructed from ``motl_base_name``, ``iteration``, and
        the appropriate extension or suffix for the given ``motl_type``.
    """
    if "relion" in motl_type:
        return f"{motl_base_name}{str(iteration).zfill(3)}_data.star"
    else:
        motl_ext = get_motl_extension(motl_type)
        return f"{motl_base_name}{iteration}{motl_ext}"


# ── STA parameter file I/O ─────────────────────────────────────────────────────
#
# Public API
# ----------
# Angle conversion:  stopgap_to_nova_angles / nova_to_stopgap_angles
# Log parser:        sta_log_read
# Classes:           StaParameters (base), StopgapParams, NovaStaParams
# Wrappers:          evaluate_alignment_from_params
#                    compute_alignment_statistics_from_params

# Run-level novaSTA keys — not broadcast per iteration, not stored in df
_NOVA_RUN_LEVEL_KEYS: frozenset[str] = frozenset({"iter", "startIndex", "createRef"})

# ── §2: Type aliases ─────────────────────────────────────────────────────────

StaRefFamily = Literal["singleref", "multiref", "multiclass"]
StaSubtomoMode = Literal[
    "ali_singleref", "ali_multiref", "ali_multiclass", "avg_singleref", "avg_multiref", "avg_multiclass"
]
StaSearchMode = Literal["hc", "shc"]
StaConeSearchType = Literal["coarse", "complete"]
StaAvgMode = Literal["full", "partial"]
StaScoringFcn = Literal["flcf", "pearson"]
StaRotMode = Literal["linear", "cubic"]

# ── Sentinels and context ─────────────────────────────────────────────────────


class _Sentinel:
    def __init__(self, name: str) -> None:
        self._name = name

    def __repr__(self) -> str:
        return self._name


MANDATORY: _Sentinel = _Sentinel("MANDATORY")
DERIVED: _Sentinel = _Sentinel("DERIVED")


@_dataclass
class StaParamContext:
    """Context passed to ``mandatory_if`` lambdas during validation and write.

    Parameters
    ----------
    create_ref : bool
        Whether an averaging pre-step is materialised on write.
    ref_family : StaRefFamily
        Reference strategy (``"singleref"``, ``"multiref"``, ``"multiclass"``).
    n_iterations : int
        Total number of alignment iterations.
    is_avg_row : bool
        ``True`` when building the optional averaging row (STOPGAP only).
    sta_type : str
        Target format — ``"stopgap"`` or ``"novasta"``.  Used by format-aware
        ``mandatory_if`` lambdas (e.g. ``rootdir`` is mandatory for STOPGAP
        only; ``fsc mask`` and ``pixel size`` are mandatory for novaSTA only).
    use_euler_search : bool
        When ``True`` and ``sta_type == "stopgap"``, the Euler parameterisation
        replaces the cone search — the four canonical angle extents become
        optional and the seven Euler columns become required.
    _row : dict
        Snapshot of the current row's canonical column values (for
        cross-parameter dependencies in ``mandatory_if``).
    """

    create_ref: bool
    ref_family: StaRefFamily
    n_iterations: int
    is_avg_row: bool
    sta_type: Literal["stopgap", "novasta"]
    use_euler_search: bool
    _row: dict

    def get(self, name: str, default: Any = None) -> Any:
        return self._row.get(name, default)


# Backward-compatible alias (internal name removed from public surface)
_WriteCtx = StaParamContext


# ── StaParamSpec dataclass ────────────────────────────────────────────────────


@_dataclass(frozen=True)
class StaParamSpec:
    """Descriptor for a single STA parameter across formats.

    Parameters
    ----------
    canonical : str or None
        Canonical df column name.  ``None`` means STOPGAP-derived-only (e.g.
        ``angincr``), which has no df column of its own.
    stopgap : str or None
        STOPGAP STAR column name **without** leading ``_``.
        ``None`` = novaSTA only.
    novasta : str or None
        novaSTA camelCase key.  ``None`` = STOPGAP only.
    dtype : type
        Expected Python type (``int``, ``float``, ``str``, ``bool``).
    default : Any
        Default value, or the sentinel ``MANDATORY`` / ``DERIVED``.  Use
        ``None`` for optional parameters that have no meaningful default.
    literals : Any, optional
        ``Literal`` type alias whose ``get_args`` gives allowed string values.
    per_iteration : bool, optional
        True when the parameter may vary per alignment iteration.
    group : str, optional
        Logical group: ``"core"``, ``"filters"``, ``"angles"``, ``"full"``,
        ``"euler"``, ``"spectral"``, ``"cleaning"``, ``"extraction"``.
    mandatory_if : callable or None, optional
        ``(ctx: StaParamContext) -> bool``; parameter is required when this
        returns ``True`` AND ``default is MANDATORY`` (or ``default`` is a
        fall-back value for conditional requirements).
    to_format : callable or None, optional
        ``(value, sta_type: str) -> Any``; converts the canonical value to the
        format-specific representation on write.  Used for parameters whose
        on-disk encoding differs between formats (e.g. ``symmetry`` Schoenflies
        vs. integer, ``split into even odd`` vs. ``ignore_halfsets`` inversion).
    from_format : callable or None, optional
        ``(value, sta_type: str) -> Any``; converts a format-specific value read
        from disk to the canonical representation on load.  Symmetric with
        ``to_format``.
    note : str, optional
        Free-text remark (e.g. open questions marked ``# CONFIRM:``).
    """

    canonical: str | None
    stopgap: str | None
    novasta: str | None
    dtype: type
    default: Any
    literals: Any = None
    per_iteration: bool = False
    group: str = "core"
    mandatory_if: Callable | None = _field(default=None, hash=False, compare=False)
    to_format: Callable | None = _field(default=None, hash=False, compare=False)
    from_format: Callable | None = _field(default=None, hash=False, compare=False)
    note: str = ""


# ── Schema table ──────────────────────────────────────────────────────────────
# The first 34 entries (those with stopgap is not None and group in
# {"core","filters","angles"}) determine STOPGAP write order for param_set="basic".


# ── Format converters (to_format / from_format for StaParamSpec) ──────────────
# _is_none_val is defined later in the module but resolved at call time (fine).

def _sym_to_format(v: Any, sta_type: str) -> Any:
    """Canonical symmetry → format-specific: Schoenflies for STOPGAP, integer for novaSTA."""
    if v is None:
        return v
    s = str(v).strip()
    if sta_type == "stopgap":
        # Ensure Schoenflies form; plain integer → "Cn"
        if re.match(r"^\d+$", s):
            return f"C{int(s)}"
        return s
    # novaSTA expects an integer
    if re.match(r"^[Cc](\d+)$", s):
        return int(s[1:])
    if re.match(r"^\d+$", s):
        return int(s)
    warnings.warn(
        f"Non-cyclic symmetry {s!r} cannot be represented as a novaSTA integer; writing 1.",
        stacklevel=4,
    )
    return 1


def _sym_from_format(v: Any, sta_type: str) -> Any:
    """Load symmetry from disk → canonical Schoenflies string."""
    if v is None:
        return v
    s = str(v).strip()
    if sta_type == "novasta" and re.match(r"^\d+$", s):
        return f"C{int(s)}"
    return s  # STOPGAP already Schoenflies; leave non-cyclic as-is


def _halfset_to_format(v: Any, sta_type: str) -> Any:
    """split into even odd (bool, True=do split) → format integer.

    STOPGAP ``ignore_halfsets=0`` means halfsets ARE used (= DO split), so the
    mapping inverts: True→0, False→1.  novaSTA ``splitIntoEvenOdd=1`` means DO
    split: True→1, False→0.
    """
    b = bool(v)
    if sta_type == "stopgap":
        return 0 if b else 1
    return 1 if b else 0


def _halfset_from_format(v: Any, sta_type: str) -> Any:
    """Format integer → canonical bool (True = do split).

    STOPGAP: ``ignore_halfsets=0`` → True (split); invert.
    novaSTA: ``splitIntoEvenOdd=1`` → True (split); same direction.
    """
    try:
        i = int(v)
    except (TypeError, ValueError):
        return bool(v)
    if sta_type == "stopgap":
        return i == 0  # invert: 0 means "do not ignore" = do split
    return i != 0

_STA_SCHEMA: list[StaParamSpec] = [
    # ── 34 core STOPGAP columns (STOPGAP write order is authoritative) ────────
    StaParamSpec(canonical="completed ali", stopgap="completed_ali", novasta=None, dtype=int, default=DERIVED),
    StaParamSpec(canonical="completed p avg", stopgap="completed_p_avg", novasta=None, dtype=int, default=DERIVED),
    StaParamSpec(canonical="completed f avg", stopgap="completed_f_avg", novasta=None, dtype=int, default=DERIVED),
    StaParamSpec(canonical="iteration", stopgap="iteration", novasta=None, dtype=int, default=DERIVED),
    StaParamSpec(
        canonical="subtomo mode",
        stopgap="subtomo_mode",
        novasta=None,
        dtype=str,
        default=DERIVED,
        literals=StaSubtomoMode,
    ),
    StaParamSpec(canonical="rootdir", stopgap="rootdir", novasta="folder", dtype=str, default=None,
                 mandatory_if=lambda ctx: ctx.sta_type == "stopgap"),
    StaParamSpec(canonical="motl", stopgap="motl_name", novasta="motl", dtype=str, default=MANDATORY),
    StaParamSpec(canonical="wedge list", stopgap="wedgelist_name", novasta="wedgeList", dtype=str, default=MANDATORY),
    StaParamSpec(canonical="binning", stopgap="binning", novasta=None, dtype=int, default=1),
    StaParamSpec(canonical="ref", stopgap="ref_name", novasta="ref", dtype=str, default=MANDATORY),
    StaParamSpec(
        canonical="subtomo name", stopgap="subtomo_name", novasta="subtomograms", dtype=str, default=MANDATORY
    ),
    StaParamSpec(canonical="mask", stopgap="mask_name", novasta="mask", dtype=str, default=MANDATORY),
    StaParamSpec(
        canonical="cc mask",
        stopgap="ccmask_name",
        novasta="ccMask",
        dtype=str,
        default=MANDATORY,
        mandatory_if=lambda ctx: not ctx.is_avg_row,
    ),
    StaParamSpec(
        canonical="search mode", stopgap="search_mode", novasta=None, dtype=str, default="hc", literals=StaSearchMode
    ),
    # STOPGAP angle iteration columns (canonical=None; derived from cone angle/sampling on write)
    StaParamSpec(canonical=None, stopgap="angincr", novasta=None, dtype=float, default=DERIVED, group="angles"),
    StaParamSpec(canonical=None, stopgap="angiter", novasta=None, dtype=int, default=DERIVED, group="angles"),
    StaParamSpec(canonical=None, stopgap="phi_angincr", novasta=None, dtype=float, default=DERIVED, group="angles"),
    StaParamSpec(canonical=None, stopgap="phi_angiter", novasta=None, dtype=int, default=DERIVED, group="angles"),
    StaParamSpec(
        canonical="cone search type",
        stopgap="cone_search_type",
        novasta=None,
        dtype=str,
        default="coarse",
        literals=StaConeSearchType,
    ),
    StaParamSpec(canonical="apply laplacian", stopgap="apply_laplacian", novasta=None, dtype=bool, default=False),
    StaParamSpec(
        canonical="low pass",
        stopgap="lp_rad",
        novasta="lowPass",
        dtype=int,
        default=MANDATORY,
        per_iteration=True,
        group="filters",
    ),
    StaParamSpec(
        canonical="low pass sigma",
        stopgap="lp_sigma",
        novasta="lowPassSigma",
        dtype=float,
        default=3.0,
        group="filters",
    ),
    StaParamSpec(
        canonical="high pass",
        stopgap="hp_rad",
        novasta="highPass",
        dtype=int,
        default=1,
        per_iteration=True,
        group="filters",
    ),
    StaParamSpec(
        canonical="high pass sigma",
        stopgap="hp_sigma",
        novasta="highPassSigma",
        dtype=float,
        default=2.0,
        group="filters",
    ),
    StaParamSpec(canonical="calc exp", stopgap="calc_exp", novasta=None, dtype=bool, default=True),
    StaParamSpec(canonical="calc ctf", stopgap="calc_ctf", novasta=None, dtype=bool, default=True),
    StaParamSpec(
        canonical="cos weight",
        stopgap="cos_weight",
        novasta=None,
        dtype=float,
        default=0.0,
        note="exponent of cosine weighting (0=none, 1=cos, 2=cos²)",
    ),
    StaParamSpec(
        canonical="score weight",
        stopgap="score_weight",
        novasta=None,
        dtype=float,
        default=0.01,
        note="pass-through factor at unbinned Nyquist",
    ),
    StaParamSpec(canonical="symmetry", stopgap="symmetry", novasta="symmetry", dtype=str, default="C1",
                 to_format=_sym_to_format, from_format=_sym_from_format),
    StaParamSpec(canonical="threshold", stopgap="score_thresh", novasta="threshold", dtype=float, default=0.0),
    StaParamSpec(canonical="subset", stopgap="subset", novasta=None, dtype=int, default=100, note="100 = disabled"),
    StaParamSpec(
        canonical="avg mode", stopgap="avg_mode", novasta=None, dtype=str, default="full", literals=StaAvgMode
    ),
    # split into even odd: canonical name; merged with STOPGAP ignore_halfsets (logically inverted).
    # Default True = do split. STOPGAP ignore_halfsets=0 means halfsets ARE used (= split); novaSTA
    # splitIntoEvenOdd=1 means DO split. to_format/from_format handle the inversion per format.
    StaParamSpec(canonical="split into even odd", stopgap="ignore_halfsets", novasta="splitIntoEvenOdd",
                 dtype=bool, default=True, to_format=_halfset_to_format, from_format=_halfset_from_format),
    StaParamSpec(
        canonical="temperature",
        stopgap="temperature",
        novasta=None,
        dtype=float,
        default=0,
        per_iteration=True,
        note="annealing; 0=disabled",
    ),
    # ── Canonical angle extents (in df; converted to angincr/angiter on STOPGAP write)
    # stopgap=None because they are NOT direct STOPGAP output columns
    # Angle extents: not mandatory for STOPGAP when use_euler_search replaces them.
    StaParamSpec(
        canonical="cone angle",
        stopgap=None,
        novasta="coneAngle",
        dtype=float,
        default=MANDATORY,
        per_iteration=True,
        group="angles",
        mandatory_if=lambda ctx: not ctx.is_avg_row and not (ctx.sta_type == "stopgap" and ctx.use_euler_search),
    ),
    StaParamSpec(
        canonical="cone sampling",
        stopgap=None,
        novasta="coneSampling",
        dtype=float,
        default=MANDATORY,
        per_iteration=True,
        group="angles",
        mandatory_if=lambda ctx: not ctx.is_avg_row and not (ctx.sta_type == "stopgap" and ctx.use_euler_search),
    ),
    StaParamSpec(
        canonical="inplane angle",
        stopgap=None,
        novasta="inplaneAngle",
        dtype=float,
        default=MANDATORY,
        per_iteration=True,
        group="angles",
        mandatory_if=lambda ctx: not ctx.is_avg_row and not (ctx.sta_type == "stopgap" and ctx.use_euler_search),
    ),
    StaParamSpec(
        canonical="inplane sampling",
        stopgap=None,
        novasta="inplaneSampling",
        dtype=float,
        default=MANDATORY,
        per_iteration=True,
        group="angles",
        mandatory_if=lambda ctx: not ctx.is_avg_row and not (ctx.sta_type == "stopgap" and ctx.use_euler_search),
    ),
    # ── STOPGAP full set ──────────────────────────────────────────────────────
    # CONFIRM: search_type allowed values not in manual (e.g. "cone", "euler"?)
    StaParamSpec(
        canonical="search type",
        stopgap="search_type",
        novasta=None,
        dtype=str,
        default="cone",
        group="full",
        note="CONFIRM: allowed values not in STOPGAP manual",
    ),
    # Euler columns: required only for STOPGAP when use_euler_search is True.
    StaParamSpec(canonical="euler axes", stopgap="euler_axes", novasta=None, dtype=str, default="none", group="euler",
                 mandatory_if=lambda ctx: ctx.sta_type == "stopgap" and ctx.use_euler_search),
    StaParamSpec(canonical="euler 1 incr", stopgap="euler_1_incr", novasta=None, dtype=float, default=0, group="euler",
                 mandatory_if=lambda ctx: ctx.sta_type == "stopgap" and ctx.use_euler_search),
    StaParamSpec(canonical="euler 1 iter", stopgap="euler_1_iter", novasta=None, dtype=int, default=0, group="euler",
                 mandatory_if=lambda ctx: ctx.sta_type == "stopgap" and ctx.use_euler_search),
    StaParamSpec(canonical="euler 2 incr", stopgap="euler_2_incr", novasta=None, dtype=float, default=0, group="euler",
                 mandatory_if=lambda ctx: ctx.sta_type == "stopgap" and ctx.use_euler_search),
    StaParamSpec(canonical="euler 2 iter", stopgap="euler_2_iter", novasta=None, dtype=int, default=0, group="euler",
                 mandatory_if=lambda ctx: ctx.sta_type == "stopgap" and ctx.use_euler_search),
    StaParamSpec(canonical="euler 3 incr", stopgap="euler_3_incr", novasta=None, dtype=float, default=0, group="euler",
                 mandatory_if=lambda ctx: ctx.sta_type == "stopgap" and ctx.use_euler_search),
    StaParamSpec(canonical="euler 3 iter", stopgap="euler_3_iter", novasta=None, dtype=int, default=0, group="euler",
                 mandatory_if=lambda ctx: ctx.sta_type == "stopgap" and ctx.use_euler_search),
    StaParamSpec(
        canonical="scoring fcn",
        stopgap="scoring_fcn",
        novasta=None,
        dtype=str,
        default="flcf",
        group="full",
        literals=StaScoringFcn,
    ),
    StaParamSpec(
        canonical="rot mode",
        stopgap="rot_mode",
        novasta=None,
        dtype=str,
        default="linear",
        group="full",
        literals=StaRotMode,
    ),
    StaParamSpec(canonical="fthresh", stopgap="fthresh", novasta=None, dtype=int, default=800, group="full"),
    # ── novaSTA-only core parameters ──────────────────────────────────────────
    StaParamSpec(
        canonical="subtomo size", stopgap=None, novasta="subtomoSize", dtype=int, default=None, group="extraction",
        mandatory_if=lambda ctx: ctx.sta_type == "novasta" and bool(ctx.get("extract subtomos")),
    ),
    StaParamSpec(
        canonical="fsc mask",
        stopgap=None,
        novasta="fscMask",
        dtype=str,
        default="none",
        group="core",
        mandatory_if=lambda ctx: ctx.sta_type == "novasta" and bool(ctx.get("split into even odd")),
    ),
    StaParamSpec(
        canonical="pixel size",
        stopgap=None,
        novasta="pixelSize",
        dtype=float,
        default=None,
        group="core",
        mandatory_if=lambda ctx: ctx.sta_type == "novasta" and bool(ctx.get("split into even odd")),
    ),
    StaParamSpec(canonical="class", stopgap=None, novasta="class", dtype=int, default=1, group="core"),
    # CONFIRM: motlBinFactor described as int with default 1.0 in some docs; implementing as float
    StaParamSpec(
        canonical="motl bin factor", stopgap=None, novasta="motlBinFactor", dtype=float, default=1.0, group="core"
    ),
    StaParamSpec(
        canonical="use Roseman CC", stopgap=None, novasta="useRosemanCC", dtype=bool, default=False, group="core"
    ),
    # ── novaSTA cleaning ──────────────────────────────────────────────────────
    StaParamSpec(
        canonical="renumber particles",
        stopgap=None,
        novasta="renumberParticles",
        dtype=bool,
        default=False,
        group="cleaning",
    ),
    StaParamSpec(
        canonical="clean out of bounds particles",
        stopgap=None,
        novasta="cleanOutOfBoundsParticles",
        dtype=bool,
        default=True,
        group="cleaning",
    ),
    StaParamSpec(
        canonical="clean by distance",
        stopgap=None,
        novasta="cleanByDistance",
        dtype=bool,
        default=False,
        group="cleaning",
    ),
    StaParamSpec(
        canonical="distance threshold",
        stopgap=None,
        novasta="distanceThreshold",
        dtype=int,
        default=None,
        group="cleaning",
        mandatory_if=lambda ctx: bool(ctx.get("clean by distance")),
    ),
    StaParamSpec(
        canonical="clean by mean grey value",
        stopgap=None,
        novasta="cleanByMeanGreyValue",
        dtype=bool,
        default=False,
        group="cleaning",
    ),
    StaParamSpec(
        canonical="unify class number", stopgap=None, novasta="unifyClassNumber", dtype=int, default=0, group="cleaning"
    ),
    # ── novaSTA extraction ────────────────────────────────────────────────────
    StaParamSpec(
        canonical="extract subtomos",
        stopgap=None,
        novasta="extractSubtomos",
        dtype=bool,
        default=False,
        group="extraction",
    ),
    StaParamSpec(
        canonical="tomograms",
        stopgap=None,
        novasta="tomograms",
        dtype=str,
        default="none",
        group="extraction",
        mandatory_if=lambda ctx: bool(ctx.get("extract subtomos")),
    ),
    StaParamSpec(
        canonical="tomo digits",
        stopgap=None,
        novasta="tomoDigits",
        dtype=int,
        default=None,
        group="extraction",
        mandatory_if=lambda ctx: bool(ctx.get("extract subtomos")),
    ),
]

# ── Lookup tables ─────────────────────────────────────────────────────────────

_SCHEMA: dict[str, StaParamSpec] = {s.canonical: s for s in _STA_SCHEMA if s.canonical is not None}
_STOPGAP_COL_TO_CANONICAL: dict[str, str] = {
    s.stopgap: s.canonical for s in _STA_SCHEMA if s.stopgap is not None and s.canonical is not None
}
_CANONICAL_TO_STOPGAP: dict[str, str] = {
    s.canonical: s.stopgap for s in _STA_SCHEMA if s.stopgap is not None and s.canonical is not None
}
_NOVASTA_KEY_TO_CANONICAL: dict[str, str] = {
    s.novasta: s.canonical for s in _STA_SCHEMA if s.novasta is not None and s.canonical is not None
}
_CANONICAL_TO_NOVASTA: dict[str, str] = {
    s.canonical: s.novasta for s in _STA_SCHEMA if s.novasta is not None and s.canonical is not None
}

# STOPGAP angle columns read from file (after Starfile.read strips leading _)
_STOPGAP_ANGLE_READ_COLS: frozenset[str] = frozenset({"angiter", "angincr", "phi_angiter", "phi_angincr"})


# ── Public schema accessor API ────────────────────────────────────────────────


def get_schema(
    sta_type: Literal["stopgap", "novasta"],
    *,
    include_derived: bool = False,
    groups: set[str] | None = None,
) -> list[StaParamSpec]:
    """Return schema entries relevant to *sta_type*.

    Parameters
    ----------
    sta_type : {"stopgap", "novasta"}
        Filter by format: STOPGAP-only entries have ``novasta=None``;
        novaSTA-only entries have ``stopgap=None``.  Shared entries appear in
        both.
    include_derived : bool, default=False
        Include entries whose ``default`` is ``DERIVED`` (e.g. iteration counter
        columns that are computed on write, not stored in the canonical df).
    groups : set of str or None, default=None
        Restrict to entries in these logical groups.  ``None`` returns all groups.

    Returns
    -------
    list of StaParamSpec
    """
    results: list[StaParamSpec] = []
    for spec in _STA_SCHEMA:
        if spec.canonical is None:
            continue
        if not include_derived and spec.default is DERIVED:
            continue
        if groups is not None and spec.group not in groups:
            continue
        if sta_type == "stopgap" and spec.stopgap is None:
            continue
        if sta_type == "novasta" and spec.novasta is None:
            continue
        results.append(spec)
    return results


def get_shared_schema(
    *,
    include_derived: bool = False,
    groups: set[str] | None = None,
) -> list[StaParamSpec]:
    """Return schema entries that appear in **both** STOPGAP and novaSTA.

    Parameters
    ----------
    include_derived : bool, default=False
    groups : set of str or None, default=None

    Returns
    -------
    list of StaParamSpec
    """
    results: list[StaParamSpec] = []
    for spec in _STA_SCHEMA:
        if spec.canonical is None:
            continue
        if not include_derived and spec.default is DERIVED:
            continue
        if groups is not None and spec.group not in groups:
            continue
        if spec.stopgap is not None and spec.novasta is not None:
            results.append(spec)
    return results


def is_mandatory(spec: StaParamSpec, ctx: "StaParamContext") -> bool:
    """Return whether *spec* is required under *ctx*.

    Parameters
    ----------
    spec : StaParamSpec
    ctx : StaParamContext
        Built via :func:`build_ctx`.

    Returns
    -------
    bool
    """
    if spec.default is MANDATORY:
        return spec.mandatory_if is None or spec.mandatory_if(ctx)
    if spec.mandatory_if is not None:
        return spec.mandatory_if(ctx)
    return False


def get_choices(spec: StaParamSpec) -> tuple[str, ...]:
    """Return the allowed literal values for *spec*, or an empty tuple.

    Parameters
    ----------
    spec : StaParamSpec

    Returns
    -------
    tuple of str
    """
    if spec.literals is None:
        return ()
    return get_args(spec.literals)


def get_default(spec: StaParamSpec) -> Any:
    """Return the default value for *spec*, or ``None`` for MANDATORY/DERIVED.

    Parameters
    ----------
    spec : StaParamSpec

    Returns
    -------
    Any
    """
    if spec.default is MANDATORY or spec.default is DERIVED:
        return None
    return spec.default


def build_ctx(
    *,
    sta_type: Literal["stopgap", "novasta"] = "stopgap",
    create_ref: bool = False,
    ref_family: StaRefFamily = "singleref",
    n_iterations: int = 1,
    is_avg_row: bool = False,
    use_euler_search: bool = False,
    row: dict | None = None,
) -> "StaParamContext":
    """Build a :class:`StaParamContext` for use with :func:`is_mandatory`.

    Parameters
    ----------
    sta_type : {"stopgap", "novasta"}, default="stopgap"
    create_ref : bool, default=False
    ref_family : StaRefFamily, default="singleref"
    n_iterations : int, default=1
    is_avg_row : bool, default=False
    use_euler_search : bool, default=False
    row : dict or None, default=None
        Canonical column values for the current row (used by cross-parameter
        ``mandatory_if`` dependencies).  ``None`` is treated as empty.

    Returns
    -------
    StaParamContext
    """
    return StaParamContext(
        create_ref=create_ref,
        ref_family=ref_family,
        n_iterations=n_iterations,
        is_avg_row=is_avg_row,
        sta_type=sta_type,
        use_euler_search=use_euler_search,
        _row=row or {},
    )


# ── Angle conversion helpers (keep unchanged) ────────────────────────────────


def stopgap_to_nova_angles(
    angiter: int,
    angincr: float,
    phi_angiter: int,
    phi_angincr: float,
) -> tuple[float, float, float, float]:
    """Convert STOPGAP angle iteration counts to novaSTA angle extents.

    Parameters
    ----------
    angiter : int
        Cone angle step count (STOPGAP ``_angiter``).
    angincr : float
        Cone angle step size in degrees (STOPGAP ``_angincr``).
    phi_angiter : int
        In-plane angle step count (STOPGAP ``_phi_angiter``).
    phi_angincr : float
        In-plane angle step size in degrees (STOPGAP ``_phi_angincr``).

    Returns
    -------
    cone_angle, cone_sampling, inplane_angle, inplane_sampling : float

    Notes
    -----
    ``cone_angle = 2 * ceil(angiter/2) * angincr`` loses the parity of an odd
    *angiter*.  A STOPGAP→novaSTA→STOPGAP round-trip may therefore turn an odd
    *angiter* into *angiter* + 1 while leaving *cone_angle* unchanged.  Treat the
    novaSTA (angle-extent) convention as canonical when both are available.
    ``cone_angle == 0`` (i.e. ``angiter == 0``) indicates an averaging-only step.
    """
    cone_angle = 2 * math.ceil(angiter / 2) * float(angincr)
    cone_sampling = float(angincr)
    inplane_angle = 2 * int(phi_angiter) * float(phi_angincr)
    inplane_sampling = float(phi_angincr)
    return cone_angle, cone_sampling, inplane_angle, inplane_sampling


def nova_to_stopgap_angles(
    cone_angle: float,
    cone_sampling: float,
    inplane_angle: float,
    inplane_sampling: float,
) -> tuple[int, float, int, float]:
    """Convert novaSTA angle extents to STOPGAP angle iteration counts.

    Parameters
    ----------
    cone_angle : float
    cone_sampling : float
    inplane_angle : float
    inplane_sampling : float

    Returns
    -------
    angiter, angincr, phi_angiter, phi_angincr : int or float
    """
    angincr = float(cone_sampling)
    angiter = int(round(float(cone_angle) / float(cone_sampling))) if float(cone_sampling) else 0
    phi_angincr = float(inplane_sampling)
    phi_angiter = int(round(float(inplane_angle) / (2 * float(inplane_sampling)))) if float(inplane_sampling) else 0
    return angiter, angincr, phi_angiter, phi_angincr


# ── novaSTA log parser ────────────────────────────────────────────────────────


def sta_log_read(log_path: PathOrStr) -> pd.DataFrame:
    """Parse a novaSTA log file into a per-iteration RMSE statistics DataFrame.

    Each iteration block is delimited by a line matching
    ``Starting iteration #N``.  Within each block the RMSE lines are harvested.

    Parameters
    ----------
    log_path : PathOrStr
        Path to the novaSTA log file.

    Returns
    -------
    pandas.DataFrame
        Columns: ``iteration``, ``rmse_x``, ``rmse_y``, ``rmse_z``,
        ``rmse_rotation``, ``rmse_angular_distance``, ``rmse_inplane_rotation``
        (only labels present in the file appear as columns).
        One row per iteration.
    """
    _label_map = {
        "RMSE x shift": "rmse_x",
        "RMSE y shift": "rmse_y",
        "RMSE z shift": "rmse_z",
        "RMSE rotation": "rmse_rotation",
        "RMSE angular distance": "rmse_angular_distance",
        "RMSE in-plane rotation": "rmse_inplane_rotation",
    }
    rows = []
    current_iter = None
    current_rmse = {}

    with open(log_path, "r") as fh:
        for line in fh:
            line = line.rstrip()
            m = re.match(r"\s*Starting iteration\s+#(\d+)", line)
            if m:
                if current_iter is not None and current_rmse:
                    rows.append({"iteration": current_iter, **current_rmse})
                current_iter = int(m.group(1))
                current_rmse = {}
                continue
            for label, col in _label_map.items():
                if line.strip().startswith(label + ":"):
                    try:
                        current_rmse[col] = float(line.split(":", 1)[1].strip())
                    except ValueError:
                        pass
                    break

    if current_iter is not None and current_rmse:
        rows.append({"iteration": current_iter, **current_rmse})

    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["iteration"])


# ── Internal helpers ─────────────────────────────────────────────────────────


def _parse_scalar(v: Any) -> int | float | str:
    """Convert a single string token to int, float, or leave as str."""
    if isinstance(v, (int, float, bool)) and not isinstance(v, bool):
        return v
    s = str(v).strip()
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        return s


def _parse_value_list(raw: Any) -> list:
    """Parse a value into a flat list of scalars.

    Accepts a scalar, a list/tuple, or a whitespace-separated string
    (e.g. ``"30 20 10"`` from a GUI text field).
    """
    if isinstance(raw, str):
        return [_parse_scalar(t) for t in raw.split()]
    if isinstance(raw, (list, tuple, np.ndarray)):
        out = []
        for item in raw:
            out.extend(_parse_value_list(item))
        return out
    return [_parse_scalar(raw)]


def _is_none_val(v: Any) -> bool:
    """Return True for None, NaN, or the string 'none'."""
    if v is None:
        return True
    if isinstance(v, float) and np.isnan(v):
        return True
    if isinstance(v, str) and v.strip().lower() == "none":
        return True
    return False


def _normalize_subtomo_mode(mode_str: str) -> str:
    """Normalise any STOPGAP ``subtomo_mode`` variant to canonical ``{ali|avg}_{family}`` form.

    Handles the old STOPGAP convention (``singleref_ali``, ``multiref_ali``, …) as well as
    the canonical convention (``ali_singleref``, ``avg_multiref``, …).
    """
    m = str(mode_str).strip()
    # Old STOPGAP format: family_mode → canonical mode_family
    _old_to_new = {
        "singleref_ali": "ali_singleref",
        "singleref_avg": "avg_singleref",
        "multiref_ali": "ali_multiref",
        "multiref_avg": "avg_multiref",
        "multiclass_ali": "ali_multiclass",
        "multiclass_avg": "avg_multiclass",
    }
    if m in _old_to_new:
        return _old_to_new[m]
    # Already canonical or unknown: return as-is
    return m


def _fmt_val(v: Any) -> str:
    """Format a scalar for novaSTA output (drop trailing .0 from whole floats)."""
    # bool before float/int — bool is a subclass of int, so must be checked first
    if isinstance(v, bool):
        return "1" if v else "0"
    if isinstance(v, float) and v == math.floor(v):
        return str(int(v))
    return str(v)


def _apply_working_dir(path_str: str, working_dir: PathOrStr | None) -> str:
    """Apply the novaSTA-style ``working_dir`` override to a stored path.

    Rules:

    * ``working_dir is None`` -- the path is returned unchanged (the
      current working directory is assumed by downstream loaders when the
      path is relative).
    * relative ``path_str`` -- ``working_dir`` is prepended.  E.g.
      ``"./ddd"`` + ``working_dir="/scratch/run42"`` →
      ``"/scratch/run42/ddd"``.
    * absolute ``path_str`` -- everything up to the basename is replaced
      with ``working_dir``.  E.g. ``"/gg/cc/motl_base"`` +
      ``working_dir="/scratch/run42"`` → ``"/scratch/run42/motl_base"``.

    Used by :meth:`StaParameters.get_motl_base_name` (the novaSTA default)
    and the novaSTA auxiliary resolvers.  STOPGAP uses its own
    rootdir-based convention -- see
    :meth:`StopgapParams.get_motl_base_name`.
    """
    if working_dir is None:
        return path_str
    # STA params files are written on the cluster (POSIX) so a leading '/'
    # indicates absolute even when the host is Windows -- check both via
    # PurePosixPath AND the native Path so drive-prefixed Windows paths
    # ("C:\\...") are recognised too.
    is_abs = PurePosixPath(path_str).is_absolute() or Path(path_str).is_absolute()
    name = PurePosixPath(path_str).name
    if is_abs:
        return str(Path(working_dir) / name)
    return str(Path(working_dir) / path_str)


def _normalize_rootdir(value: str) -> str:
    """Prepend ``./`` if *value* is a bare folder name (no separator, not absolute).

    Parameters
    ----------
    value : str
        The ``rootdir`` value from the STOPGAP parameter file.

    Returns
    -------
    str
    """
    p = PurePosixPath(value)
    if p.is_absolute() or "/" in value or value.startswith("."):
        return value
    return f"./{value}"


def _generate_temperature_schedule(T: float, n: int) -> list[float]:
    """Generate the per-iteration temperature annealing schedule.

    Parameters
    ----------
    T : float
        Starting temperature.  ``0`` disables annealing (returns all zeros).
    n : int
        Number of alignment iterations.

    Returns
    -------
    list of float
        One value per iteration.  If ``T == 0``, all values are ``0.0``.
        Otherwise, each value is ``max(1.0, T - i)`` for iteration index *i*.
        A :func:`warnings.warn` is emitted when the schedule has not yet
        reached 1 after *n* iterations.
    """
    if T == 0:
        return [0.0] * n
    schedule = []
    for i in range(n):
        val = max(1.0, T - i)
        schedule.append(val)
    final = schedule[-1] if schedule else 0.0
    if final > 1.0:
        remaining = int(final - 1)
        warnings.warn(
            f"Temperature schedule ends at {final:.0f} after {n} iterations; "
            f"{remaining} more iterations would be needed to reach 1.",
            stacklevel=3,
        )
    return schedule


def _to_canonical_key(k: str) -> str:
    """Convert a parameter key in any format to its canonical name.

    Accepts canonical names, ``snake_case``, and novaSTA ``camelCase``.
    Falls back to the original key if nothing matches.
    """
    # Direct canonical match
    if k in _SCHEMA:
        return k
    # snake_case → space-separated
    space = k.replace("_", " ").lower()
    if space in _SCHEMA:
        return space
    # novaSTA camelCase via lookup table
    if k in _NOVASTA_KEY_TO_CANONICAL:
        return _NOVASTA_KEY_TO_CANONICAL[k]
    # Heuristic camelCase conversion
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", k)
    s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", s)
    display = " ".join(w if (w.isupper() and len(w) > 1) else w.lower() for w in s.split())
    if display in _SCHEMA:
        return display
    # Unknown key — return as-is
    return k


# ── Base class ────────────────────────────────────────────────────────────────


class StaParameters:
    """Base class for STA parameter file representations.

    Both :class:`StopgapParams` and :class:`NovaStaParams` store all
    parameters in **canonical column names** (e.g. ``"motl"``, ``"wedge list"``,
    ``"low pass"``).  Format-specific names are applied only on write.

    Attributes
    ----------
    df : pandas.DataFrame
        Canonical columns, one row per *alignment* iteration.
    df_extra : pandas.DataFrame
        Format-specific extra columns not in the schema (same index as ``df``).
    df_stats : pandas.DataFrame or None
        Per-iteration RMSE statistics populated by :meth:`attach_log`.
    fsc : pandas.DataFrame or None
        FSC curve populated by :meth:`attach_fsc`.
    create_ref : bool
        Whether an averaging pre-step should be materialised on write.
    ref_family : StaRefFamily
        Reference strategy: ``"singleref"``, ``"multiref"``, or ``"multiclass"``.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        df_extra: pd.DataFrame | None = None,
        create_ref: bool = False,
        ref_family: StaRefFamily = "singleref",
        use_euler_search: bool = False,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.df_extra = (
            df_extra.reset_index(drop=True)
            if df_extra is not None and not df_extra.empty
            else pd.DataFrame(index=range(len(df)))
        )
        self.df_stats = None
        self.fsc = None
        self.create_ref = bool(create_ref)
        self.ref_family: StaRefFamily = str(ref_family)  # type: ignore[assignment]
        self.use_euler_search: bool = bool(use_euler_search)

    # ── Accessors ──────────────────────────────────────────────────────────

    @property
    def num_iterations(self):
        """Number of alignment iterations stored in ``df``."""
        return len(self.df)

    @property
    def start_iteration(self):
        """First iteration index, or None if ``df`` is empty."""
        if "iteration" not in self.df.columns or self.df.empty:
            return None
        return int(self.df["iteration"].iloc[0])

    @property
    def end_iteration(self):
        """Last iteration index, or None if ``df`` is empty."""
        if "iteration" not in self.df.columns or self.df.empty:
            return None
        return int(self.df["iteration"].iloc[-1])

    @property
    def motl_type(self):
        """Motl type string for this format (implemented by subclasses)."""
        raise NotImplementedError

    def get_motl_base_name(
        self,
        separator: str = "_",
        working_dir: PathOrStr | None = None,
    ) -> str | None:
        """Return the motl base name (column value + *separator*), or None.

        The default implementation matches the novaSTA layout: the motl
        column already carries the (relative or absolute) full path and is
        used verbatim.  ``working_dir``, when supplied, overrides the
        directory portion of that path -- relative paths are joined onto
        ``working_dir``, absolute paths have their directory replaced with
        ``working_dir`` while the basename is preserved.  Subclasses (e.g.
        :class:`StopgapParams`) override this to honour their own
        directory conventions.

        Parameters
        ----------
        separator : str, default='_'
            Appended to the motl path stored in ``df``.  Downstream loaders
            then append ``<iter><extension>`` to build per-iteration paths.
        working_dir : PathOrStr or None, default=None
            Directory override.  When ``None``, the value stored in the
            params file is used as-is (current working directory is assumed
            for relative paths).  When provided, it replaces the directory
            portion of the stored motl path -- see :func:`_apply_working_dir`
            for the exact rules.

        Returns
        -------
        str or None
        """
        if "motl" not in self.df.columns or self.df.empty:
            return None
        val = self.df["motl"].iloc[0]
        if _is_none_val(val):
            return None
        return _apply_working_dir(str(val), working_dir) + separator

    # ── Factory / dispatch ─────────────────────────────────────────────────

    @classmethod
    def load(
        cls,
        path: PathOrStr,
        sta_type: str | None = None,
        **kwargs: Any,
    ) -> "StaParameters":
        """Load a parameter file, dispatching on *sta_type* or file extension.

        Parameters
        ----------
        path : PathOrStr
            Path to the parameter file.
        sta_type : str or None, default=None
            ``"stopgap"`` or ``"novasta"``.  If ``None``, ``.star`` → stopgap,
            otherwise → novasta.
        **kwargs
            Forwarded to the subclass ``from_file``.

        Returns
        -------
        StopgapParams or NovaStaParams
        """
        if sta_type is None:
            sta_type = "stopgap" if str(path).endswith(".star") else "novasta"
        sta_type = sta_type.lower()
        if sta_type == "stopgap":
            return StopgapParams.from_file(path, **kwargs)
        if sta_type in ("novasta", "nova"):
            return NovaStaParams.from_file(path, **kwargs)
        raise ValueError(f"Unknown sta_type {sta_type!r}. Use 'stopgap' or 'novasta'.")

    @classmethod
    def from_dict(
        cls,
        params: dict,
        sta_type: str = "novasta",
    ) -> "StaParameters":
        """Construct an :class:`StaParameters` from a parameter dictionary (GUI path).

        Parameters
        ----------
        params : dict
            Keyed by canonical column names, ``snake_case``, or novaSTA
            ``camelCase`` key names.  Values may be scalars, lists, or
            whitespace-separated strings.

            **Control keys** (extracted and not stored in ``df``):

            * ``start_index`` / ``start index`` (int, default 1)
            * ``create_ref`` / ``create ref`` (bool, default ``False``)
            * ``ref_family`` / ``ref family`` (StaRefFamily, default ``"singleref"``)

        sta_type : str, default="novasta"
            Target subclass: ``"stopgap"`` or ``"novasta"``.

        Returns
        -------
        StopgapParams or NovaStaParams

        Raises
        ------
        ValueError
            If per-iteration sequence lengths disagree.

        Warns
        -----
        UserWarning
            When mandatory parameters are absent (object is still created;
            use :meth:`validate` for a complete problem list).

        """

        # -- Normalise keys → canonical names ---------------------------------
        def _pop_alias(d: dict, *aliases: str, default: Any = None) -> Any:
            for key in aliases:
                if key in d:
                    return d.pop(key)
            return default

        normalised: dict[str, Any] = {_to_canonical_key(k): v for k, v in params.items()}

        # Extract control keys (support both space and underscore variants)
        start_index_raw = _pop_alias(normalised, "start index", "start_index", default=1)
        create_ref_raw = _pop_alias(normalised, "create ref", "create_ref", default=0)
        ref_family_raw = _pop_alias(normalised, "ref family", "ref_family", default="singleref")
        use_euler_raw = _pop_alias(normalised, "use euler search", "use_euler_search", default=0)

        start_index = int(_parse_value_list(start_index_raw)[0])
        create_ref = bool(int(_parse_value_list(create_ref_raw)[0]))
        ref_family_str = str(_parse_value_list(ref_family_raw)[0])
        use_euler_search = bool(int(_parse_value_list(use_euler_raw)[0]))

        # -- Parse all remaining values into lists ----------------------------
        parsed: dict[str, list] = {k: _parse_value_list(v) for k, v in normalised.items()}

        # Infer n_align from the longest sequence
        lengths = {k: len(v) for k, v in parsed.items() if len(v) > 1}
        n_align = max(lengths.values()) if lengths else 1

        bad = {k: le for k, le in lengths.items() if le != n_align}
        if bad:
            raise ValueError(
                f"All per-iteration sequences must share the same length " f"({n_align}).  Mismatched keys: {bad}"
            )

        # Broadcast scalars
        expanded: dict[str, list] = {k: (v * n_align if len(v) == 1 else list(v)) for k, v in parsed.items()}

        # -- Temperature annealing schedule -----------------------------------
        if "temperature" in expanded and n_align > 1:
            tv = expanded["temperature"]
            if len(set(tv)) == 1 and not _is_none_val(tv[0]) and float(tv[0]) != 0:
                expanded["temperature"] = _generate_temperature_schedule(float(tv[0]), n_align)

        # -- Warn on missing mandatory params (do NOT raise) ------------------
        missing_mandatory: list[str] = []
        first_row = {k: v[0] for k, v in expanded.items()} if expanded else {}
        sta_type_lower = (sta_type or "novasta").lower()
        ctx = _WriteCtx(
            create_ref=create_ref,
            ref_family=ref_family_str,
            n_iterations=n_align,
            is_avg_row=False,
            sta_type="stopgap" if sta_type_lower == "stopgap" else "novasta",
            use_euler_search=use_euler_search,
            _row=first_row,
        )
        for spec in _STA_SCHEMA:
            if spec.canonical is None or spec.canonical in expanded:
                continue
            is_required = False
            if spec.default is MANDATORY:
                is_required = spec.mandatory_if is None or spec.mandatory_if(ctx)
            elif spec.mandatory_if is not None:
                is_required = spec.mandatory_if(ctx)
            if is_required:
                missing_mandatory.append(spec.canonical)

        if missing_mandatory:
            warnings.warn(
                f"Missing mandatory parameter(s): {sorted(missing_mandatory)}. "
                f"The object will be created with missing values.",
                stacklevel=2,
            )

        # -- Build DataFrame --------------------------------------------------
        iters = list(range(start_index, start_index + n_align))
        df_data: dict[str, list] = {"iteration": iters}
        df_data.update(expanded)
        df = pd.DataFrame(df_data) if n_align > 0 else pd.DataFrame(columns=["iteration"])

        klass = StopgapParams if sta_type_lower == "stopgap" else NovaStaParams
        return klass(df, pd.DataFrame(), create_ref=create_ref, ref_family=ref_family_str,
                     use_euler_search=use_euler_search)

    # ── Auxiliary data ─────────────────────────────────────────────────────

    def attach_log(self, log_path: PathOrStr) -> pd.DataFrame:
        """Parse a novaSTA log and attach per-iteration RMSE stats.

        Parameters
        ----------
        log_path : PathOrStr
            Path to the novaSTA log file.

        Returns
        -------
        pandas.DataFrame
        """
        self.df_stats = sta_log_read(log_path)
        return self.df_stats

    def attach_fsc(
        self,
        path: PathOrStr,
        pixel_size: float | None = None,
        box_size: int | None = None,
    ) -> pd.DataFrame:
        """Read and attach an FSC curve.

        Parameters
        ----------
        path : PathOrStr
            Path to the FSC curve file.
        pixel_size : float or None, default=None
            Pixel size in Å.
        box_size : int or None, default=None
            Subtomogram box size.  If ``None``, falls back to ``subtomo_size``
            in ``df``.

        Returns
        -------
        pandas.DataFrame
        """
        if box_size is None and "subtomo size" in self.df.columns and not self.df.empty:
            v = self.df["subtomo size"].iloc[0]
            if not _is_none_val(v):
                box_size = int(v)
        self.fsc = ioutils.fsc_read(path, pixel_size=pixel_size, box_size=box_size)
        return self.fsc

    # ── Format conversion ──────────────────────────────────────────────────

    def to_novasta(self) -> "NovaStaParams":
        """Return a :class:`NovaStaParams` backed by the same canonical ``df``.

        Format-specific column conversions (symmetry Schoenflies→integer;
        split into even odd bool→inverted integer) are applied eagerly to
        the copied df so that the result is a self-consistent novaSTA object.
        ``write_out`` further passes values through ``to_format``; for already-
        converted values this is idempotent.
        """
        df = self.df.copy()
        for spec in _STA_SCHEMA:
            if spec.canonical is None or spec.to_format is None:
                continue
            if spec.canonical not in df.columns:
                continue
            df[spec.canonical] = df[spec.canonical].apply(
                lambda v, s=spec: s.to_format(v, "novasta") if not _is_none_val(v) else v
            )
        return NovaStaParams(df, self.df_extra.copy(),
                             create_ref=self.create_ref, ref_family=self.ref_family,
                             use_euler_search=self.use_euler_search)

    def to_stopgap(self) -> "StopgapParams":
        """Return a :class:`StopgapParams` backed by the same canonical ``df``.

        Format-specific column conversions are applied eagerly; ``write_out``
        / ``_build_row`` further pass values through ``to_format``, which is
        idempotent for already-converted values.
        """
        df = self.df.copy()
        for spec in _STA_SCHEMA:
            if spec.canonical is None or spec.to_format is None:
                continue
            if spec.canonical not in df.columns:
                continue
            df[spec.canonical] = df[spec.canonical].apply(
                lambda v, s=spec: s.to_format(v, "stopgap") if not _is_none_val(v) else v
            )
        return StopgapParams(df, self.df_extra.copy(),
                             create_ref=self.create_ref, ref_family=self.ref_family,
                             use_euler_search=self.use_euler_search)

    # ── Validation ─────────────────────────────────────────────────────────

    def validate(self, param_set: Literal["basic", "full"] = "basic") -> list[str]:
        """Return a list of human-readable problem strings (empty = OK).

        Parameters
        ----------
        param_set : {"basic", "full"}, default="basic"
            ``"basic"`` checks only the 34 core STOPGAP parameters;
            ``"full"`` also checks full/euler/spectral groups.

        Returns
        -------
        list of str
            Empty when no problems were found.

        Notes
        -----
        Checks performed (in order):

        1. Missing MANDATORY parameters (considering ``mandatory_if`` conditions).
        2. Literal value violations.
        3. Per-iteration length disagreements.
        4. ``euler_axes`` axis constraint (second must differ from first).
        5. Cone/Euler mutual exclusivity.
        """
        problems: list[str] = []

        if self.df.empty:
            return ["DataFrame is empty — no parameters to validate."]

        sta_type = "stopgap" if isinstance(self, StopgapParams) else "novasta"
        ctx_row = {c: self.df[c].iloc[0] for c in self.df.columns}
        ctx = _WriteCtx(
            create_ref=self.create_ref,
            ref_family=self.ref_family,
            n_iterations=len(self.df),
            is_avg_row=False,
            sta_type=sta_type,
            use_euler_search=getattr(self, "use_euler_search", False),
            _row=ctx_row,
        )

        active_groups = {"core", "filters", "angles"}
        if param_set == "full":
            active_groups.update({"full", "euler", "spectral"})

        for spec in _STA_SCHEMA:
            if spec.canonical is None:
                continue
            if spec.group not in active_groups:
                continue

            val = self.df[spec.canonical].iloc[0] if spec.canonical in self.df.columns else None
            is_none = _is_none_val(val)

            # 1. Mandatory check
            is_required: bool
            if spec.default is MANDATORY:
                is_required = spec.mandatory_if is None or spec.mandatory_if(ctx)
            elif spec.mandatory_if is not None:
                is_required = spec.mandatory_if(ctx)
            else:
                is_required = False

            if is_required and is_none:
                problems.append(f"Missing mandatory parameter: {spec.canonical!r}")

            # 2. Literal check
            if spec.literals is not None and not is_none:
                allowed = get_args(spec.literals)
                if allowed and str(val) not in allowed:
                    problems.append(f"Invalid value for {spec.canonical!r}: {val!r} — " f"allowed values: {allowed}")

            # 3. Per-iteration length check
            if spec.per_iteration and spec.canonical in self.df.columns:
                col_vals = self.df[spec.canonical].tolist()
                non_none_count = sum(1 for v in col_vals if not _is_none_val(v))
                expected = len(self.df)
                if non_none_count not in (0, 1, expected):
                    problems.append(
                        f"Per-iteration mismatch for {spec.canonical!r}: "
                        f"{non_none_count} non-null values, expected 1 or {expected}"
                    )

        # 4. euler_axes axis constraint
        if "euler axes" in self.df.columns:
            axes = self.df["euler axes"].iloc[0]
            if not _is_none_val(axes) and str(axes) not in ("none", ""):
                s = str(axes)
                if len(s) >= 2 and s[0] == s[1]:
                    problems.append(f"euler axes: second axis must differ from first " f"(got {axes!r})")

        # 5. Cone/Euler mutual exclusivity
        cone_cols = {"cone angle", "cone sampling", "inplane angle", "inplane sampling"}
        euler_cols = {"euler axes"}
        has_cone = any(c in self.df.columns and not _is_none_val(self.df[c].iloc[0]) for c in cone_cols)
        has_euler = (
            "euler axes" in self.df.columns
            and not _is_none_val(self.df["euler axes"].iloc[0])
            and str(self.df["euler axes"].iloc[0]) not in ("none", "", "0")
        )
        if has_cone and has_euler:
            problems.append("Cone search and Euler search are mutually exclusive.")

        return problems


# ── StopgapParams ──────────────────────────────────────────────────────────────


class StopgapParams(StaParameters):
    """STOPGAP STAR-file subtomogram-averaging parameter representation.

    All parameters are stored in ``df`` using **canonical column names** (e.g.
    ``"motl"``, ``"wedge list"``, ``"low pass"``).  STOPGAP-specific column
    names (e.g. ``motl_name``, ``wedgelist_name``, ``lp_rad``) are applied only
    when writing.  Raw STOPGAP angle-iteration columns (``angincr``, ``angiter``,
    ``phi_angincr``, ``phi_angiter``) are converted to the canonical angle-extent
    form (``"cone angle"``, ``"cone sampling"``, ``"inplane angle"``,
    ``"inplane sampling"``) on :meth:`from_file` and converted back on
    :meth:`write_out`.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        df_extra: pd.DataFrame | None = None,
        create_ref: bool = False,
        ref_family: StaRefFamily = "singleref",
        use_euler_search: bool = False,
    ) -> None:
        super().__init__(df, df_extra, create_ref=create_ref, ref_family=ref_family,
                         use_euler_search=use_euler_search)

    @property
    def motl_type(self) -> str:
        return "stopgap"

    def get_motl_base_name(
        self,
        separator: str = "_",
        working_dir: PathOrStr | None = None,
    ) -> str | None:
        """STOPGAP motl base name -- ``rootdir/lists/<motl><separator>``.

        STOPGAP runs follow a fixed directory layout under a single
        ``rootdir`` (stored as the ``rootdir`` column of every row): motls
        live in ``lists/``, masks in ``masks/``, and references in
        ``refs/``.  When ``working_dir`` is supplied it overrides the
        ``rootdir`` column; the ``lists/`` subdirectory + the ``motl name``
        column + ``separator`` are still appended verbatim.  Downstream
        loaders append ``<iter>.star`` to form the per-iteration path.

        Parameters
        ----------
        separator : str, default='_'
            Appended to the motl name.
        working_dir : PathOrStr or None, default=None
            Directory override.  When ``None``, the ``rootdir`` column is
            used; when both are absent, only the motl name + separator is
            returned (no leading directory).

        Returns
        -------
        str or None
        """
        col = "motl"
        if col not in self.df.columns or self.df.empty:
            return None
        val = self.df[col].iloc[0]
        if _is_none_val(val):
            return None
        root = self._effective_rootdir(working_dir)
        if root is None:
            return str(val) + separator
        return str(Path(root) / "lists" / str(val)) + separator

    def _effective_rootdir(self, working_dir: PathOrStr | None) -> str | None:
        """Return the directory that anchors STOPGAP's subdir layout.

        ``working_dir`` wins when supplied; otherwise the ``rootdir``
        column from the first row is used.  Returns ``None`` if neither
        is available -- callers then fall back to bare column values.
        """
        if working_dir is not None:
            return str(working_dir)
        if "rootdir" in self.df.columns and not self.df.empty:
            v = self.df["rootdir"].iloc[0]
            if not _is_none_val(v):
                return str(v)
        return None

    def _resolve_in_subdir(
        self,
        col: str,
        subdir: str,
        working_dir: PathOrStr | None,
    ) -> str | None:
        """Join ``<rootdir>/<subdir>/<self.df[col]>`` if both pieces are present."""
        if col not in self.df.columns or self.df.empty:
            return None
        val = self.df[col].iloc[0]
        if _is_none_val(val):
            return None
        root = self._effective_rootdir(working_dir)
        if root is None:
            return str(val)
        return str(Path(root) / subdir / str(val))

    def resolve_wedge_list(self, working_dir: PathOrStr | None = None) -> str | None:
        """STOPGAP wedge list path -- ``rootdir/lists/<wedge list>``.

        Parameters
        ----------
        working_dir : PathOrStr or None, default=None
            Overrides the ``rootdir`` column when supplied.

        Returns
        -------
        str or None
        """
        return self._resolve_in_subdir("wedge list", "lists", working_dir)

    def resolve_mask(self, working_dir: PathOrStr | None = None) -> str | None:
        """STOPGAP particle mask path -- ``rootdir/masks/<mask>``.

        Parameters
        ----------
        working_dir : PathOrStr or None, default=None
            Overrides the ``rootdir`` column when supplied.

        Returns
        -------
        str or None
        """
        return self._resolve_in_subdir("mask", "masks", working_dir)

    def resolve_ccmask(self, working_dir: PathOrStr | None = None) -> str | None:
        """STOPGAP CC mask path -- ``rootdir/masks/<cc mask>``.

        Parameters
        ----------
        working_dir : PathOrStr or None, default=None
            Overrides the ``rootdir`` column when supplied.

        Returns
        -------
        str or None
        """
        return self._resolve_in_subdir("cc mask", "masks", working_dir)

    def resolve_ref_base(
        self,
        working_dir: PathOrStr | None = None,
        separator: str = "_",
    ) -> str | None:
        """STOPGAP reference base name -- ``rootdir/refs/<ref><separator>``.

        Downstream code appends ``<iter>.em`` to form the per-iteration
        reference path, mirroring the motl-base-name convention.

        Parameters
        ----------
        working_dir : PathOrStr or None, default=None
            Overrides the ``rootdir`` column when supplied.
        separator : str, default='_'
            Appended to the reference name.

        Returns
        -------
        str or None
        """
        col = "ref"
        if col not in self.df.columns or self.df.empty:
            return None
        val = self.df[col].iloc[0]
        if _is_none_val(val):
            return None
        root = self._effective_rootdir(working_dir)
        if root is None:
            return str(val) + separator
        return str(Path(root) / "refs" / str(val)) + separator

    def get_half_map_paths(
        self,
        iteration: int,
        working_dir: PathOrStr | None = None,
    ) -> tuple[str, str] | None:
        """Return the (half_A, half_B) half-map paths for a given iteration.

        STOPGAP names half-maps as ``<refbase>_A_<iteration>.<ext>`` and
        ``<refbase>_B_<iteration>.<ext>``.  The extension (``.em`` or ``.mrc``)
        is resolved by checking what exists on disk; ``.em`` is the default
        when neither (or both) exist.

        Parameters
        ----------
        iteration : int
            Iteration number.
        working_dir : PathOrStr or None, default=None
            Directory override (see :meth:`resolve_ref_base`).

        Returns
        -------
        tuple[str, str] or None
            ``(path_A, path_B)``, or ``None`` if ``ref`` is not set.

        Warns
        -----
        UserWarning
            When both the ``.em`` and ``.mrc`` forms exist for the same half.
        """
        ref_base = self.resolve_ref_base(working_dir=working_dir, separator="")
        if ref_base is None:
            return None

        def _pick_ext(letter: str) -> str:
            em = f"{ref_base}_{letter}_{iteration}.em"
            mrc = f"{ref_base}_{letter}_{iteration}.mrc"
            em_exists = Path(em).exists()
            mrc_exists = Path(mrc).exists()
            if em_exists and mrc_exists:
                warnings.warn(
                    f"Both {em!r} and {mrc!r} exist; using .em.",
                    UserWarning,
                    stacklevel=3,
                )
                return em
            if mrc_exists:
                return mrc
            return em

        return _pick_ext("A"), _pick_ext("B")

    @classmethod
    def from_file(
        cls,
        path: PathOrStr,
        load_completed_only: bool = False,
    ) -> "StopgapParams":
        """Load a STOPGAP subtomogram parameter STAR file.

        Parameters
        ----------
        path : PathOrStr
            Path to the STOPGAP ``.star`` parameter file.
        load_completed_only : bool, default=False
            If ``True``, keep only rows where ``completed_ali == 1``.

        Returns
        -------
        StopgapParams

        Raises
        ------
        ValueError
            If the file specifier is wrong, or if columns still carry a
            leading ``_`` after :func:`Starfile.read` strips one
            (indicating a double-underscore in the file from a broken writer).
        """
        frame, spec, _ = Starfile.read(path, data_id=0)
        expected = "data_stopgap_subtomo_parameters"
        if spec != expected:
            raise ValueError(
                f"Not a valid STOPGAP subtomo parameter file: " f"specifier is {spec!r}, expected {expected!r}."
            )

        # Starfile.read strips ONE leading '_'; any column still starting
        # with '_' means the file had '__' (double) which indicates a broken writer.
        bad_cols = [c for c in frame.columns if c.startswith("_")]
        if bad_cols:
            raise ValueError(
                f"STOPGAP file has double-underscore columns {bad_cols!r}. "
                f"The file was written by a broken writer that prefixed "
                f"already-prefixed column names."
            )

        if load_completed_only and "completed_ali" in frame.columns:
            frame = frame[frame["completed_ali"].astype(str) == "1"].reset_index(drop=True)

        # Normalise subtomo_mode and extract meta-flags
        if "subtomo_mode" in frame.columns:
            frame["subtomo_mode"] = frame["subtomo_mode"].apply(_normalize_subtomo_mode)
            ali_mask = frame["subtomo_mode"].str.startswith("ali_")
            ref_family_str = "singleref"
            for mode in frame.loc[ali_mask, "subtomo_mode"]:
                parts = str(mode).split("_", 1)
                if len(parts) == 2 and parts[1] in get_args(StaRefFamily):
                    ref_family_str = parts[1]
                    break
            create_ref = (~ali_mask).any()
        else:
            ali_mask = pd.Series([True] * len(frame))
            ref_family_str = "singleref"
            create_ref = False

        ali_frame = frame[ali_mask].reset_index(drop=True)

        # Convert STOPGAP angle columns to canonical angle extents
        _sg_angle_cols = _STOPGAP_ANGLE_READ_COLS
        if _sg_angle_cols <= set(ali_frame.columns):

            def _conv_angles(row: pd.Series) -> pd.Series:
                ai = row.get("angiter", None)
                ac = row.get("angincr", None)
                pai = row.get("phi_angiter", None)
                pac = row.get("phi_angincr", None)
                if any(_is_none_val(x) for x in (ai, ac, pai, pac)):
                    return pd.Series(
                        [None, None, None, None],
                        index=["cone angle", "cone sampling", "inplane angle", "inplane sampling"],
                    )
                try:
                    ca, cs, ia, is_ = stopgap_to_nova_angles(float(ai), float(ac), float(pai), float(pac))
                except (ValueError, TypeError):
                    ca = cs = ia = is_ = None
                return pd.Series(
                    [ca, cs, ia, is_],
                    index=["cone angle", "cone sampling", "inplane angle", "inplane sampling"],
                )

            angle_df = ali_frame.apply(_conv_angles, axis=1)
            ali_frame = ali_frame.drop(columns=list(_sg_angle_cols))
            ali_frame = pd.concat(
                [ali_frame.reset_index(drop=True), angle_df.reset_index(drop=True)],
                axis=1,
            )

        # Map remaining STOPGAP column names to canonical names
        rename_map = {c: _STOPGAP_COL_TO_CANONICAL[c] for c in ali_frame.columns if c in _STOPGAP_COL_TO_CANONICAL}
        ali_frame = ali_frame.rename(columns=rename_map)

        # Apply from_format converters and coerce dtype=bool columns
        for spec in _STA_SCHEMA:
            if spec.canonical is None or spec.canonical not in ali_frame.columns:
                continue
            if spec.from_format is not None:
                ali_frame[spec.canonical] = ali_frame[spec.canonical].apply(
                    lambda v, s=spec: s.from_format(v, "stopgap") if not _is_none_val(v) else v
                )
            elif spec.dtype is bool:
                ali_frame[spec.canonical] = ali_frame[spec.canonical].apply(
                    lambda v: bool(int(v)) if not _is_none_val(v) else v
                )

        # Normalise rootdir
        if "rootdir" in ali_frame.columns:
            ali_frame["rootdir"] = ali_frame["rootdir"].apply(
                lambda v: _normalize_rootdir(str(v)) if not _is_none_val(v) else v
            )

        return cls(ali_frame, pd.DataFrame(), create_ref=create_ref, ref_family=ref_family_str)

    def write_out(
        self,
        path: PathOrStr,
        create_ref: bool | None = None,
        ref_family: StaRefFamily | None = None,
        total_iterations: int | None = None,
        param_set: Literal["basic", "full"] = "basic",
        strict: bool = False,
    ) -> None:
        """Write a STOPGAP subtomogram parameter STAR file.

        Parameters
        ----------
        path : PathOrStr
            Destination path for the ``.star`` file.
        create_ref : bool or None, default=None
            If ``None``, uses the value stored on the object.
        ref_family : StaRefFamily or None, default=None
            ``"singleref"``, ``"multiref"``, or ``"multiclass"``.
            If ``None``, uses the value stored on the object.
        total_iterations : int or None, default=None
            If larger than the current alignment-iteration count, pad with
            extra rows (params copied from the last row, progress flags 0).
        param_set : {"basic", "full"}, default="basic"
            ``"basic"`` writes the 34 standard STOPGAP columns.
            ``"full"`` additionally writes full/euler/spectral groups.
            Auto-promotes to ``"full"`` when any full-group columns are set
            in ``df``.
        strict : bool, default=False
            If ``True``, raise :class:`ValueError` on validation problems
            instead of emitting a warning.
        """
        cr = self.create_ref if create_ref is None else bool(create_ref)
        family = self.ref_family if ref_family is None else str(ref_family)

        # Validate
        problems = self.validate(param_set)
        if problems:
            msg = "STOPGAP params validation problems:\n" + "\n".join(f"  - {p}" for p in problems)
            if strict:
                raise ValueError(msg)
            warnings.warn(msg, stacklevel=2)

        ali_df = self.df.reset_index(drop=True)

        # Optional padding
        if total_iterations is not None and total_iterations > len(ali_df):
            n_pad = total_iterations - len(ali_df)
            last_it = (
                int(ali_df["iteration"].iloc[-1])
                if not ali_df.empty and "iteration" in ali_df.columns
                else (self.start_iteration or 1)
            )
            pad_ali = ali_df.iloc[[-1] * n_pad].copy().reset_index(drop=True)
            pad_ali["iteration"] = [last_it + i + 1 for i in range(n_pad)]
            ali_df = pd.concat([ali_df, pad_ali], ignore_index=True)

        # Determine groups to include
        basic_groups = {"core", "filters", "angles"}
        full_groups = basic_groups | {"full", "euler", "spectral"}

        if param_set == "full":
            active_groups = full_groups
        else:
            active_groups = basic_groups
            # Auto-promote when user has populated full-group columns
            has_full = any(
                spec.group in {"full", "euler", "spectral"}
                and spec.canonical is not None
                and spec.canonical in self.df.columns
                and not self.df[spec.canonical].isna().all()
                for spec in _STA_SCHEMA
            )
            if has_full:
                active_groups = full_groups

        # STOPGAP column list for this param_set (in schema order, stopgap cols only)
        out_cols: list[str] = [
            spec.stopgap for spec in _STA_SCHEMA if spec.stopgap is not None and spec.group in active_groups
        ]

        # ── Row builder ──────────────────────────────────────────────────────
        _angle_canonical = {"cone angle", "cone sampling", "inplane angle", "inplane sampling"}
        _angle_sg = {"angincr", "angiter", "phi_angincr", "phi_angiter"}

        def _build_row(df_row: pd.Series, is_avg: bool, iteration: int | None = None) -> dict:
            ctx = _WriteCtx(
                create_ref=cr,
                ref_family=family,
                n_iterations=len(ali_df),
                is_avg_row=is_avg,
                sta_type="stopgap",
                use_euler_search=self.use_euler_search,
                _row={c: df_row.get(c) for c in df_row.index},
            )
            row: dict = {}

            for spec in _STA_SCHEMA:
                if spec.stopgap is None or spec.group not in active_groups:
                    continue
                col = spec.stopgap

                # ── DERIVED columns ──────────────────────────────────────────
                if spec.default is DERIVED:
                    if col == "completed_ali":
                        row[col] = 0
                    elif col == "completed_p_avg":
                        row[col] = 0
                    elif col == "completed_f_avg":
                        row[col] = 0
                    elif col == "iteration":
                        row[col] = iteration if iteration is not None else int(df_row.get("iteration", 1))
                    elif col == "subtomo_mode":
                        mode = "avg" if is_avg else "ali"
                        row[col] = f"{mode}_{family}"
                    elif col in _angle_sg:
                        pass  # handled below
                    else:
                        row[col] = "none"
                    continue

                # ── Angle iteration columns (canonical=None, derived on write) ─
                if col in _angle_sg:
                    # Handled in the angle block below
                    continue

                # ── Normal columns ───────────────────────────────────────────
                if spec.canonical is not None and spec.canonical in df_row.index:
                    val = df_row.get(spec.canonical)
                else:
                    val = None

                if _is_none_val(val):
                    default = spec.default
                    if default is MANDATORY or default is DERIVED:
                        row[col] = "none"
                    elif spec.to_format is not None:
                        row[col] = spec.to_format(default, "stopgap")
                    elif isinstance(default, bool):
                        row[col] = 1 if default else 0
                    else:
                        row[col] = default
                else:
                    # Apply to_format converter when present (symmetry, split into even odd, etc.)
                    if spec.to_format is not None:
                        val = spec.to_format(val, "stopgap")
                    # STOPGAP is a STAR file (text); convert remaining booleans to integers
                    elif isinstance(val, bool):
                        val = 1 if val else 0
                    row[col] = val

                # Suppress values that are not applicable in this write context
                # e.g. ccmask_name must be "none" in avg rows
                if spec.mandatory_if is not None and not spec.mandatory_if(ctx):
                    row[col] = "none"

            # ── Angle columns ────────────────────────────────────────────────
            if is_avg:
                for ag in _angle_sg:
                    if ag in out_cols:
                        row[ag] = "none"
            else:
                ca = df_row.get("cone angle")
                cs = df_row.get("cone sampling")
                ia = df_row.get("inplane angle")
                isp = df_row.get("inplane sampling")
                if not any(_is_none_val(x) for x in (ca, cs, ia, isp)):
                    ai, ac, pai, pac = nova_to_stopgap_angles(float(ca), float(cs), float(ia), float(isp))
                    if "angiter" in out_cols:
                        row["angiter"] = ai
                    if "angincr" in out_cols:
                        row["angincr"] = ac
                    if "phi_angiter" in out_cols:
                        row["phi_angiter"] = pai
                    if "phi_angincr" in out_cols:
                        row["phi_angincr"] = pac
                else:
                    for ag in _angle_sg:
                        if ag in out_cols:
                            row[ag] = "none"

            return row

        # ── Assemble rows ────────────────────────────────────────────────────
        rows: list[dict] = []

        if cr:
            # Leading avg row
            if not ali_df.empty:
                first_row = ali_df.iloc[0]
                first_it = int(first_row.get("iteration", 1))
                rows.append(_build_row(first_row, is_avg=True, iteration=first_it))
            else:
                empty_row = {c: "none" for c in out_cols}
                empty_row.update(
                    completed_ali=0,
                    completed_p_avg=0,
                    completed_f_avg=0,
                    iteration=self.start_iteration or 1,
                    subtomo_mode=f"avg_{family}",
                )
                rows.append(empty_row)

        for _, df_row in ali_df.iterrows():
            rows.append(_build_row(df_row, is_avg=False))

        out_df = pd.DataFrame(rows, columns=out_cols)
        out_df = out_df.where(out_df.notna(), other="none")
        out_df = out_df.replace({None: "none"})

        Starfile.write([out_df], path, specifiers=["data_stopgap_subtomo_parameters"])


# ── NovaStaParams ──────────────────────────────────────────────────────────────


class NovaStaParams(StaParameters):
    """novaSTA flat key-value parameter file representation.

    All parameters are stored in ``df`` using **canonical column names**
    (e.g. ``"cone angle"``, ``"low pass"``, ``"wedge list"``).  Run-level
    keys (``iter``, ``startIndex``, ``createRef``) become object attributes,
    not columns.  Unknown camelCase keys (not in the schema) are stored
    under their original camelCase name in ``df``.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        df_extra: pd.DataFrame | None = None,
        create_ref: bool = False,
        ref_family: StaRefFamily = "singleref",
        use_euler_search: bool = False,
    ) -> None:
        super().__init__(df, df_extra, create_ref=create_ref, ref_family=ref_family,
                         use_euler_search=use_euler_search)
        self._orig_key_order: list[str] | None = None  # original camelCase key order

    @property
    def motl_type(self) -> str:
        return "emmotl"

    def _resolve_path_col(
        self,
        col: str,
        working_dir: PathOrStr | None,
    ) -> str | None:
        """Apply :func:`_apply_working_dir` to ``self.df[col].iloc[0]``."""
        if col not in self.df.columns or self.df.empty:
            return None
        val = self.df[col].iloc[0]
        if _is_none_val(val):
            return None
        return _apply_working_dir(str(val), working_dir)

    def resolve_wedge_list(self, working_dir: PathOrStr | None = None) -> str | None:
        """novaSTA wedge list path -- ``self.df['wedge list']`` with optional override.

        Parameters
        ----------
        working_dir : PathOrStr or None, default=None
            Directory override.  Applied via :func:`_apply_working_dir`
            (relative paths get prepended; absolute paths have their
            directory replaced).

        Returns
        -------
        str or None
        """
        return self._resolve_path_col("wedge list", working_dir)

    def resolve_mask(self, working_dir: PathOrStr | None = None) -> str | None:
        """novaSTA particle mask path -- ``self.df['mask']`` with optional override.

        Parameters
        ----------
        working_dir : PathOrStr or None, default=None
            Directory override (see :func:`_apply_working_dir`).

        Returns
        -------
        str or None
        """
        return self._resolve_path_col("mask", working_dir)

    def resolve_ccmask(self, working_dir: PathOrStr | None = None) -> str | None:
        """novaSTA CC mask path -- ``self.df['cc mask']`` with optional override.

        Parameters
        ----------
        working_dir : PathOrStr or None, default=None
            Directory override (see :func:`_apply_working_dir`).

        Returns
        -------
        str or None
        """
        return self._resolve_path_col("cc mask", working_dir)

    def resolve_ref_base(
        self,
        working_dir: PathOrStr | None = None,
        separator: str = "_",
    ) -> str | None:
        """novaSTA reference base -- ``self.df['ref'] + separator`` with optional override.

        novaSTA stores the reference column as a base name (downstream
        appends ``<iter>.em``); this helper mirrors
        :meth:`get_motl_base_name` for symmetry.

        Parameters
        ----------
        working_dir : PathOrStr or None, default=None
            Directory override (see :func:`_apply_working_dir`).
        separator : str, default='_'
            Appended after the resolved path.

        Returns
        -------
        str or None
        """
        if "ref" not in self.df.columns or self.df.empty:
            return None
        val = self.df["ref"].iloc[0]
        if _is_none_val(val):
            return None
        return _apply_working_dir(str(val), working_dir) + separator

    def get_fsc_filename(
        self,
        iteration: int,
        working_dir: PathOrStr | None = None,
    ) -> str | None:
        """Return the novaSTA FSC curve path for a given iteration.

        novaSTA writes its FSC curve as ``<refname>_<iteration>_fsc.txt``
        (one correlation value per line).  The path is constructed from the
        same ``ref`` column and the same optional ``working_dir`` override
        used by :meth:`resolve_ref_base`.

        Parameters
        ----------
        iteration : int
            Iteration number, embedded verbatim (not zero-padded).
        working_dir : PathOrStr or None, default=None
            Directory override applied to the stored ``ref`` value
            (see :func:`_apply_working_dir`).

        Returns
        -------
        str or None
            Full FSC filename, or ``None`` if ``ref`` is not set.
        """
        ref_base = self.resolve_ref_base(working_dir=working_dir, separator="")
        if ref_base is None:
            return None
        return f"{ref_base}_{iteration}_fsc.txt"

    def get_half_map_paths(
        self,
        iteration: int,
        working_dir: PathOrStr | None = None,
    ) -> tuple[str, str] | None:
        """Return the (even, odd) half-map paths for a given iteration.

        novaSTA names half-maps as ``<refname>_even_<iteration>.em`` and
        ``<refname>_odd_<iteration>.em``.

        Parameters
        ----------
        iteration : int
            Iteration number.
        working_dir : PathOrStr or None, default=None
            Directory override (see :meth:`resolve_ref_base`).

        Returns
        -------
        tuple[str, str] or None
            ``(even_path, odd_path)``, or ``None`` if ``ref`` is not set.
        """
        ref_base = self.resolve_ref_base(working_dir=working_dir, separator="")
        if ref_base is None:
            return None
        return (
            f"{ref_base}_even_{iteration}.em",
            f"{ref_base}_odd_{iteration}.em",
        )

    @classmethod
    def from_file(cls, path: PathOrStr) -> "NovaStaParams":
        """Load a novaSTA flat key-value parameter file.

        Parameters
        ----------
        path : PathOrStr
            Path to the novaSTA flat parameter file (typically ``.txt`` /
            ``.params``).

        Returns
        -------
        NovaStaParams

        Raises
        ------
        ValueError
            If any parameter has a value count other than 1 or ``iter``.
        """
        raw: dict[str, list] = {}
        key_order: list[str] = []
        with open(path, "r") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                key = parts[0]
                vals = [_parse_scalar(p) for p in parts[1:]]
                if key not in raw:
                    key_order.append(key)
                raw[key] = vals  # last occurrence wins for duplicates

        n_align = int(raw.get("iter", [1])[0])
        start_index = int(raw.get("startIndex", [1])[0])
        create_ref = bool(int(raw.get("createRef", [0])[0]))

        # Validate 1-or-N rule
        for k, v in raw.items():
            if k in _NOVA_RUN_LEVEL_KEYS:
                continue
            if n_align > 0 and len(v) not in (1, n_align):
                raise ValueError(
                    f"Parameter {k!r} has {len(v)} value(s) but iter={n_align}. "
                    f"Each parameter must supply 1 or {n_align} values."
                )

        def broadcast(vals: list) -> list:
            return vals * max(n_align, 1) if len(vals) == 1 else vals

        iters = list(range(start_index, start_index + max(n_align, 0)))

        # Build df: map camelCase → canonical where possible; keep originals otherwise
        df_data: dict[str, list] = {"iteration": iters}
        for k in key_order:
            if k in _NOVA_RUN_LEVEL_KEYS:
                continue
            col_name = _NOVASTA_KEY_TO_CANONICAL.get(k, k)
            df_data[col_name] = broadcast(raw[k])[:n_align] if n_align > 0 else []

        df = pd.DataFrame(df_data) if n_align > 0 else pd.DataFrame(columns=["iteration"])

        # Handle temperature annealing schedule (scalar → per-iteration)
        if "temperature" in df.columns and n_align > 1:
            tv = df["temperature"].tolist()
            if len(set(str(v) for v in tv)) == 1 and not _is_none_val(tv[0]) and float(tv[0]) != 0:
                df["temperature"] = _generate_temperature_schedule(float(tv[0]), n_align)

        # Apply from_format converters and coerce dtype=bool columns
        for spec in _STA_SCHEMA:
            if spec.canonical is None or spec.canonical not in df.columns:
                continue
            if spec.from_format is not None:
                df[spec.canonical] = df[spec.canonical].apply(
                    lambda v, s=spec: s.from_format(v, "novasta") if not _is_none_val(v) else v
                )
            elif spec.dtype is bool:
                df[spec.canonical] = df[spec.canonical].apply(
                    lambda v: bool(int(v)) if not _is_none_val(v) else v
                )

        # Normalise rootdir (mapped from novaSTA 'folder')
        if "rootdir" in df.columns:
            df["rootdir"] = df["rootdir"].apply(
                lambda v: _normalize_rootdir(str(v)) if not _is_none_val(v) else v
            )

        obj = cls(df, pd.DataFrame(), create_ref=create_ref, ref_family="singleref")
        obj._orig_key_order = key_order
        return obj

    def write_out(
        self,
        path: PathOrStr,
        create_ref: bool | None = None,
        param_set: Literal["basic", "full"] = "basic",
        strict: bool = False,
    ) -> None:
        """Write a novaSTA flat key-value parameter file.

        Parameters
        ----------
        path : PathOrStr
            Destination path for the flat key-value file.
        create_ref : bool or None, default=None
            If ``None``, uses the flag stored on the object.
        param_set : {"basic", "full"}, default="basic"
            Controls which columns are emitted.
        strict : bool, default=False
            If ``True``, raise on validation problems; otherwise warn.
        """
        cr = self.create_ref if create_ref is None else bool(create_ref)
        n_align = len(self.df)

        # Build canonical-name → original novaSTA key (from stored key order)
        canonical_to_key: dict[str, str] = {}
        if self._orig_key_order:
            for k in self._orig_key_order:
                if k in _NOVA_RUN_LEVEL_KEYS:
                    continue
                canon = _NOVASTA_KEY_TO_CANONICAL.get(k, k)
                canonical_to_key[canon] = k

        def get_nova_key(col_name: str) -> str:
            """Return novaSTA camelCase key for a canonical or unknown column."""
            if col_name in canonical_to_key:
                return canonical_to_key[col_name]
            if col_name in _CANONICAL_TO_NOVASTA:
                return _CANONICAL_TO_NOVASTA[col_name]
            # Unknown column (not in schema): write key as-is (already camelCase from file)
            return col_name

        # Pre-build to_format lookup for columns that need format-specific encoding
        _to_fmt: dict[str, Any] = {
            s.canonical: s.to_format for s in _STA_SCHEMA if s.canonical is not None and s.to_format is not None
        }

        def col_vals(col: str) -> list | None:
            if col not in self.df.columns:
                return None
            vals = list(self.df[col])
            if all(_is_none_val(v) for v in vals):
                return None
            # Apply to_format when present (e.g. symmetry Schoenflies→int, split into even odd inversion)
            if col in _to_fmt:
                fn = _to_fmt[col]
                vals = [fn(v, "novasta") if not _is_none_val(v) else v for v in vals]
            return vals

        def is_constant(vals: list) -> bool:
            return len({_fmt_val(v) for v in vals}) == 1

        def write_param(lines: list[str], key: str, vals: list) -> None:
            if is_constant(vals):
                lines.append(f"{key} {_fmt_val(vals[0])}")
            else:
                lines.append(f"{key} {' '.join(_fmt_val(v) for v in vals)}")

        def ensure_length(vals: list, n: int) -> list:
            return vals * n if len(vals) == 1 else vals

        lines: list[str] = [f"createRef {1 if cr else 0}", f"iter {n_align}"]
        if not self.df.empty and "iteration" in self.df.columns:
            lines.append(f"startIndex {int(self.df['iteration'].iloc[0])}")

        # Angle coupling: if any angle param varies per iteration, expand all
        _angle_cols = ["cone angle", "cone sampling", "inplane angle", "inplane sampling"]
        angle_vals = {f: col_vals(f) for f in _angle_cols}
        angle_per_iter = n_align > 1 and any(v is not None and not is_constant(v) for v in angle_vals.values())
        if angle_per_iter:
            for f in _angle_cols:
                angle_vals[f] = ensure_length(angle_vals[f], n_align) if angle_vals[f] else [None] * n_align

        # Filter coupling: if any filter param varies, expand all
        _filter_cols = ["low pass", "high pass"]
        filter_vals = {f: col_vals(f) for f in _filter_cols}
        filter_per_iter = n_align > 1 and any(v is not None and not is_constant(v) for v in filter_vals.values())
        if filter_per_iter:
            for f in _filter_cols:
                filter_vals[f] = ensure_length(filter_vals[f], n_align) if filter_vals[f] else [None] * n_align

        _special_cols = set(_angle_cols + _filter_cols)

        # Write order: original key order first, then any new canonical columns
        write_order: list[str] = []
        seen: set[str] = set()
        if self._orig_key_order:
            for k in self._orig_key_order:
                if k in _NOVA_RUN_LEVEL_KEYS:
                    continue
                canon = _NOVASTA_KEY_TO_CANONICAL.get(k, k)
                if canon not in seen:
                    seen.add(canon)
                    write_order.append(canon)
        for c in self.df.columns:
            if c != "iteration" and c not in seen:
                seen.add(c)
                write_order.append(c)

        # Non-angle/filter columns first
        for col_name in write_order:
            if col_name in _special_cols:
                continue
            vals = col_vals(col_name)
            if vals is not None:
                write_param(lines, get_nova_key(col_name), vals)

        # Angle group
        for col_name in _angle_cols:
            vals = angle_vals[col_name]
            if vals is not None:
                write_param(lines, get_nova_key(col_name), vals)

        # Filter group
        for col_name in _filter_cols:
            vals = filter_vals[col_name]
            if vals is not None:
                write_param(lines, get_nova_key(col_name), vals)

        with open(path, "w") as fh:
            fh.write("\n".join(lines) + "\n")


# ── File-driven progress evaluation wrappers ──────────────────────────────────


def _resolve_sta_params(
    input_params: "PathOrStr | dict | StaParameters",
    sta_type: str | None = None,
    **kwargs: Any,
) -> "StaParameters":
    """Resolve a path, dict, or StaParameters into an StaParameters object."""
    if isinstance(input_params, StaParameters):
        return input_params
    if isinstance(input_params, dict):
        return StaParameters.from_dict(input_params, sta_type=sta_type or "novasta")
    return StaParameters.load(str(input_params), sta_type=sta_type, **kwargs)


def evaluate_alignment_from_params(
    input_params: "PathOrStr | dict | StaParameters",
    sta_type: str | None = None,
    motl_separator: str = "_",
    working_dir: PathOrStr | None = None,
    **kwargs: Any,
) -> list:
    """Run :func:`evaluate_alignment` driven by a parameter file, dict, or object.

    Pulls ``motl_base_name``, ``start_iteration``, ``end_iteration``, and
    ``motl_type`` directly off the resolved :class:`StaParameters` instance
    and forwards everything else through to :func:`evaluate_alignment`.

    Parameters
    ----------
    input_params : PathOrStr, dict, or StaParameters
        Path to a parameter file, a canonical parameter dict, or an already-
        loaded StaParameters object.
    sta_type : str or None, default=None
        ``"stopgap"`` or ``"novasta"``.  Auto-detected from extension if ``None``.
    motl_separator : str, default='_'
        Appended to the stored motl path to form the base name
        (e.g. ``"./allmotl_lt"`` → ``"./allmotl_lt_"``).
    working_dir : PathOrStr or None, default=None
        Optional directory override.  For STOPGAP it replaces the
        ``rootdir`` column (the ``lists/`` subdirectory is still appended,
        so the motl prefix becomes ``working_dir/lists/<motl name>_``).
        For novaSTA it overrides the directory portion of the stored motl
        path -- relative paths get prepended, absolute paths have their
        directory replaced.  Pass ``None`` to use the path embedded in the
        parameter file as-is.
    **kwargs
        Forwarded to :func:`evaluate_alignment`.  ``motl_type`` defaults to
        the format-native type (``"stopgap"`` or ``"emmotl"``) if not supplied.

    Returns
    -------
    list of pandas.DataFrame
    """
    params = _resolve_sta_params(input_params, sta_type=sta_type)
    base = params.get_motl_base_name(motl_separator, working_dir=working_dir)
    if base is None:
        raise ValueError("No motl path found in the parameter file.")
    kwargs.setdefault("motl_type", params.motl_type)
    return evaluate_alignment(base, params.start_iteration, params.end_iteration, **kwargs)


def compute_alignment_statistics_from_params(
    input_params: "PathOrStr | dict | StaParameters",
    sta_type: str | None = None,
    motl_separator: str = "_",
    working_dir: PathOrStr | None = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """Run :func:`compute_alignment_statistics` driven by a parameter file, dict, or object.

    Parameters
    ----------
    input_params : PathOrStr, dict, or StaParameters
        Path to a parameter file, a canonical parameter dict, or an already-
        loaded StaParameters object.
    sta_type : str or None, default=None
        ``"stopgap"`` or ``"novasta"``.  Auto-detected from extension if ``None``.
    motl_separator : str, default='_'
        Appended to the stored motl path to form the base name.
    working_dir : PathOrStr or None, default=None
        Optional directory override.  See
        :func:`evaluate_alignment_from_params` for the per-format rules.
    **kwargs
        Forwarded to :func:`compute_alignment_statistics`.

    Returns
    -------
    pandas.DataFrame
    """
    params = _resolve_sta_params(input_params, sta_type=sta_type)
    base = params.get_motl_base_name(motl_separator, working_dir=working_dir)
    if base is None:
        raise ValueError("No motl path found in the parameter file.")
    kwargs.setdefault("motl_type", params.motl_type)
    return compute_alignment_statistics(base, params.start_iteration, params.end_iteration, **kwargs)


# ── Block schedule ─────────────────────────────────────────────────────────────


@_dataclass
class Block:
    """One block in a STOPGAP / novaSTA run schedule.

    Parameters
    ----------
    n_iterations : int
        Number of iterations this block contributes.
    job : {"avg", "ali"}
        Whether this block is an averaging or alignment step.
    motl_name : str
        Motl name pattern.  Supports ``{base}``, ``{run}``, and ``{iter}``
        placeholders, which are expanded by :func:`expand_motl_name`.
    search_mode : {"hc", "shc"} or None, default=None
        Search mode.  ``None`` is valid for averaging blocks.
    temperature : float, default=0.0
        Simulated annealing temperature.  0 = disabled.
    overrides : dict, default={}
        Per-block overrides for any ``base_params`` field.
    """

    n_iterations: int
    job: Literal["avg", "ali"]
    motl_name: str
    search_mode: Literal["hc", "shc"] | None = None
    temperature: float = 0.0
    overrides: dict = _field(default_factory=dict)


@_dataclass
class StaRun:
    """Serialisable descriptor for one STOPGAP / novaSTA STA run.

    Parameters
    ----------
    input_motl_id : str
        Pool ID of the input motl.
    run_mode : {"singleref", "multiref", "multiclass"}
        Run mode (property of the run, not the block).
    output_base : Path
        Parent directory inside which run folder(s) are created.
    folder_name : str
        Base name for the run folder.  When ``n_runs > 1`` the per-run
        suffix ``_mr{i}`` is appended automatically.
    subtomo_path : Path
        Path to the subtomogram directory; symlinked into each run folder.
    base_params : dict
        Run-wide parameters (masks, wedgelist, binning, filters, angular
        search, …).
    schedule : list[Block]
        Per-block parameters.  At least one :class:`Block` required.
    n_runs : int, default=1
        Number of parallel runs (mr1..mrN).  Single run when 1.
    references : list[Path], default=[]
        Existing reference files (when applicable; copied to ``ref/``).
    """

    input_motl_id: str
    run_mode: Literal["singleref", "multiref", "multiclass"]
    output_base: Path
    folder_name: str
    subtomo_path: Path
    base_params: dict
    schedule: list[Block]
    n_runs: int = 1
    references: list[Path] = _field(default_factory=list)


# ── Schedule helper functions ──────────────────────────────────────────────────


def compute_startidx_sequence(
    schedule: list[Block],
    starting_iter: int = 1,
) -> list[int]:
    """Return the ``startidx`` for every block in *schedule*.

    The first block starts at *starting_iter*; each subsequent block
    starts immediately after the previous one finishes.

    Parameters
    ----------
    schedule : list[Block]
        Ordered list of blocks.
    starting_iter : int, default=1
        Absolute iteration index of the very first iteration.

    Returns
    -------
    list[int]
        One entry per block in the same order as *schedule*.

    Examples
    --------
    >>> blks = [Block(1, "avg", "{base}"), Block(10, "ali", "{base}"), Block(20, "ali", "{base}")]
    >>> compute_startidx_sequence(blks, starting_iter=1)
    [1, 2, 12]
    """
    result: list[int] = []
    cur = starting_iter
    for blk in schedule:
        result.append(cur)
        cur += blk.n_iterations
    return result


def compose_subtomo_mode(
    job: Literal["avg", "ali"],
    run_mode: Literal["singleref", "multiref", "multiclass"],
) -> str:
    """Compose a STOPGAP ``subtomo_mode`` string.

    Parameters
    ----------
    job : {"avg", "ali"}
    run_mode : {"singleref", "multiref", "multiclass"}

    Returns
    -------
    str
        One of the six ``StaSubtomoMode`` literals.

    Examples
    --------
    >>> compose_subtomo_mode("ali", "multiref")
    'ali_multiref'
    """
    return f"{job}_{run_mode}"


def expand_motl_name(
    pattern: str,
    *,
    base: str,
    run: int,
    iter_: int,
) -> str:
    """Expand a motl name *pattern* with ``{base}``, ``{run}``, ``{iter}`` placeholders.

    Parameters
    ----------
    pattern : str
        Template string, e.g. ``"{base}_ref_mr{run}"`` or ``"{base}"``.
    base : str
        Value to substitute for ``{base}``.
    run : int
        Value to substitute for ``{run}``.
    iter_ : int
        Value to substitute for ``{iter}``.

    Returns
    -------
    str

    Examples
    --------
    >>> expand_motl_name("{base}_ref_mr{run}", base="motl", run=2, iter_=1)
    'motl_ref_mr2'
    """
    return pattern.format(base=base, run=run, iter=iter_)


def denovo_template_blocks() -> list[Block]:
    """Return the 3-block de-novo reference creation template.

    Consists of 1 averaging block (``{base}_ref_mr{run}``) followed by
    10 SHC-annealing alignment iterations and 20 SHC zero-temperature
    alignment iterations (30 total).

    Returns
    -------
    list[Block]
    """
    return [
        Block(n_iterations=1, job="avg", motl_name="{base}_ref_mr{run}"),
        Block(n_iterations=10, job="ali", motl_name="{base}", search_mode="shc", temperature=10.0),
        Block(n_iterations=20, job="ali", motl_name="{base}", search_mode="shc", temperature=0.0),
    ]


def existing_refs_template_blocks() -> list[Block]:
    """Return the 2-block classification-with-existing-references template.

    Consists of 1 HC alignment iteration followed by 29 SHC iterations
    (30 total), temperature 0 throughout.

    Returns
    -------
    list[Block]
    """
    return [
        Block(n_iterations=1, job="ali", motl_name="{base}", search_mode="hc", temperature=0.0),
        Block(n_iterations=29, job="ali", motl_name="{base}", search_mode="shc", temperature=0.0),
    ]


def continue_run_prefill(last_row: dict) -> dict:
    """Extract the starting-iteration and base-params for a continue run.

    Parameters
    ----------
    last_row : dict
        Canonical column values from the last row of an existing parameter
        file (i.e. ``params.df.iloc[-1].to_dict()``).

    Returns
    -------
    dict
        ``{"starting_iter": int, "base_params": dict}``

    Notes
    -----
    The ``temperature`` field is forced to ``0`` in the returned
    ``base_params`` because simulated annealing is only meaningful at the
    start of a de-novo run — see spec §A5.
    """
    iteration = int(last_row.get("iteration", 1))
    n_iters = 1  # default; not stored per-row in STA format
    starting_iter = iteration + n_iters

    # Strip derived / meta columns; keep parameter columns
    _skip = {"iteration", "completed ali", "completed p avg", "completed f avg",
              "subtomo mode", "startidx"}
    base_params = {k: v for k, v in last_row.items() if k not in _skip and v is not None}
    base_params["temperature"] = 0.0
    return {"starting_iter": starting_iter, "base_params": base_params}


# ── Reference renaming helper (Part C.4) ────────────────────────────────────────


def validate_ref_mapping(
    mapping: list[dict],
) -> list[str]:
    """Validate a reference-renaming mapping table before applying it.

    Each entry in *mapping* is ``{"src_run": int, "src_class": int,
    "src_iter": int, "dst_class": int, "src_ref_dir": str}``.

    Checks performed:

    * Target classes are contiguous starting from 1.
    * For every source entry, the main map plus ``A`` and ``B`` half maps exist.

    Parameters
    ----------
    mapping : list[dict]
        Rows from the reference-renaming table in the GUI.

    Returns
    -------
    list[str]
        Validation errors; empty list means the mapping is valid.
    """
    errors: list[str] = []
    dst_classes = sorted({int(row["dst_class"]) for row in mapping})
    expected = list(range(1, len(dst_classes) + 1))
    if dst_classes != expected:
        errors.append(
            f"Target classes must be contiguous from 1; got {dst_classes} (expected {expected})."
        )

    for row in mapping:
        ref_dir = Path(row["src_ref_dir"])
        r, c, it = int(row["src_run"]), int(row["src_class"]), int(row["src_iter"])
        stem = f"ref_{it}_{c}"
        for suffix in ("", "_A", "_B"):
            candidate = ref_dir / f"ref{suffix}_{it}_{c}.em"
            alt = ref_dir / f"{stem}{suffix}.em"
            if not candidate.is_file() and not alt.is_file():
                errors.append(
                    f"Run {r} class {c} iter {it}: half-map '{suffix or 'main'}' not found "
                    f"(tried {candidate} and {alt})."
                )
    return errors


# ── Run-folder creation (Part C) ─────────────────────────────────────────────


_RUN_FOLDER_SUBDIRS: tuple[str, ...] = (
    "ref", "comm", "fsc", "raw", "meta", "lists", "masks", "temp", "blank"
)
_SUBTOMO_SETTINGS_CONTENT = "vol_ext=.em\n"


def _run_rootdir(sta_run: StaRun, run_idx: int | None) -> Path:
    """Return the rootdir Path for *run_idx* (1-based), or for a single run."""
    base = Path(sta_run.output_base) / sta_run.folder_name
    if sta_run.n_runs > 1 and run_idx is not None:
        return base.parent / f"{base.name}_mr{run_idx}"
    return base


def preflight_run_folder(
    sta_run: StaRun,
    motl_paths: list[Path],
    starting_iter: int = 1,
) -> list[str]:
    """Validate all inputs for a run-folder creation without touching the filesystem.

    All problems are collected before returning (not stop-at-first).

    Parameters
    ----------
    sta_run : StaRun
        Run descriptor.
    motl_paths : list[Path]
        Paths returned by :func:`create_multiref_run` or
        :func:`create_denovo_multiref_run`.
    starting_iter : int, default=1

    Returns
    -------
    list[str]
        Human-readable problem descriptions; empty list means OK to proceed.
    """
    errors: list[str] = []
    bp = sta_run.base_params

    for key in ("mask_name", "ccmask_name", "wedgelist_name"):
        val = bp.get(key)
        if val:
            if not Path(val).is_file():
                errors.append(f"{key} not found or not a file: {val}")

    for mp in motl_paths:
        if not Path(mp).is_file():
            errors.append(f"Motl file not found: {mp}")

    for ref in sta_run.references:
        if not Path(ref).is_file():
            errors.append(f"Reference not found: {ref}")

    if not Path(sta_run.subtomo_path).is_dir():
        errors.append(f"subtomo_path is not a directory: {sta_run.subtomo_path}")

    run_indices = list(range(1, sta_run.n_runs + 1)) if sta_run.n_runs > 1 else [None]
    for ri in run_indices:
        rd = _run_rootdir(sta_run, ri)
        if rd.exists():
            errors.append(f"Target run folder already exists: {rd}")

    return errors


def create_run_folder(
    sta_run: StaRun,
    motl_paths: list[Path],
    starting_iter: int = 1,
    overwrite: bool = False,
) -> dict:
    """Create STOPGAP run folders according to the C1 layout spec.

    Validates first (via :func:`preflight_run_folder`), then creates
    everything atomically (all or nothing per run).

    Parameters
    ----------
    sta_run : StaRun
        Run descriptor.
    motl_paths : list[Path]
        Paths of motl files to copy into ``lists/``.
    starting_iter : int, default=1
        First iteration index for the ``startidx`` sequence.
    overwrite : bool, default=False
        When ``True``, remove existing run folders before creating.
        Never merges — partial overwrites are not permitted.

    Returns
    -------
    dict
        Manifest with keys ``"dirs_created"``, ``"files_copied"``,
        ``"symlinks_created"`` — each a list of str paths.

    Raises
    ------
    FileExistsError
        When a target folder already exists and *overwrite* is False.
    ValueError
        When *overwrite* is True but the folder cannot be removed.
    """
    import shutil
    import os

    manifest: dict[str, list[str]] = {
        "dirs_created": [],
        "files_copied": [],
        "symlinks_created": [],
    }
    run_indices = list(range(1, sta_run.n_runs + 1)) if sta_run.n_runs > 1 else [None]

    for run_idx in run_indices:
        rd = _run_rootdir(sta_run, run_idx)

        if rd.exists():
            if overwrite:
                shutil.rmtree(rd)
            else:
                raise FileExistsError(
                    f"Run folder {rd} already exists. "
                    "Pass overwrite=True or choose a different folder_name."
                )

        for subdir in _RUN_FOLDER_SUBDIRS:
            d = rd / subdir
            d.mkdir(parents=True, exist_ok=True)
            manifest["dirs_created"].append(str(d))

        subtomo_link = rd / "subtomograms"
        os.symlink(Path(sta_run.subtomo_path).resolve(), subtomo_link)
        manifest["symlinks_created"].append(str(subtomo_link))

        settings_file = rd / "subtomo_settings.txt"
        settings_file.write_text(_SUBTOMO_SETTINGS_CONTENT)
        manifest["files_copied"].append(str(settings_file))

        bp = sta_run.base_params
        for key in ("mask_name", "ccmask_name"):
            val = bp.get(key)
            if val:
                src = Path(val)
                dst = rd / "masks" / src.name
                shutil.copy2(src, dst)
                manifest["files_copied"].append(str(dst))

        wl = bp.get("wedgelist_name")
        if wl:
            src = Path(wl)
            dst = rd / "lists" / src.name
            shutil.copy2(src, dst)
            manifest["files_copied"].append(str(dst))

        for mp in motl_paths:
            src = Path(mp)
            dst = rd / "lists" / src.name
            shutil.copy2(src, dst)
            manifest["files_copied"].append(str(dst))

        for ref in sta_run.references:
            src = Path(ref)
            dst = rd / "ref" / src.name
            shutil.copy2(src, dst)
            manifest["files_copied"].append(str(dst))

        _write_subtomo_param(sta_run, run_idx, starting_iter, rd)
        manifest["files_copied"].append(str(rd / "subtomo_param.star"))

    return manifest


def _write_subtomo_param(
    sta_run: StaRun,
    run_idx: int | None,
    starting_iter: int,
    run_dir: Path,
) -> None:
    """Write ``subtomo_param.star`` into *run_dir* from *sta_run*'s schedule."""
    startidx_seq = compute_startidx_sequence(sta_run.schedule, starting_iter)
    actual_run = run_idx if run_idx is not None else 1
    rootdir_str = str(run_dir).rstrip("/").rstrip("\\") + "/"

    base_motl = sta_run.base_params.get("motl", "allmotl")

    rows: list[dict] = []
    for block_idx, blk in enumerate(sta_run.schedule):
        startidx = startidx_seq[block_idx]
        merged = {**sta_run.base_params, **blk.overrides}
        for it_offset in range(blk.n_iterations):
            iter_num = startidx + it_offset
            row: dict = {k: v for k, v in merged.items()}
            row["iteration"] = iter_num
            row["subtomo mode"] = compose_subtomo_mode(blk.job, sta_run.run_mode)
            row["motl"] = expand_motl_name(
                blk.motl_name, base=base_motl, run=actual_run, iter_=iter_num
            )
            row["rootdir"] = rootdir_str
            if blk.search_mode is not None:
                row["search mode"] = blk.search_mode
            row["temperature"] = blk.temperature
            rows.append(row)

    df = pd.DataFrame(rows)
    params = StopgapParams(df, create_ref=False, ref_family=sta_run.run_mode)
    params.write_out(str(run_dir / "subtomo_param.star"))


# ── Co-assignment factor and consensus functions ───────────────────────────────
# Spec: MULTI_CLASSIFICATION_CONSENSUS_1.md
# M = (1/R) B B^T  where  B = [P_1 | ... | P_R]  is the (N, sum K_r) indicator.
# The factored form avoids ever materialising the N × N matrix.

_ABSENT: int = -1
_MATRIX_GUARD: int = 5000


@_dataclass(frozen=True)
class CoassignmentFactor:
    """Factored representation of a co-assignment matrix over several runs.

    Attributes
    ----------
    labels : numpy.ndarray
        ``(N, R)`` int32 array of *global* class indices; class blocks of
        different runs never overlap. ``-1`` marks a particle absent from that
        run.
    particle_ids : numpy.ndarray
        ``(N,)`` sorted particle identifiers; row ``i`` of everything refers to
        ``particle_ids[i]``.
    run_labels : list of str
        Human-readable name per run, in column order of ``labels``.
    n_classes : numpy.ndarray
        ``(R,)`` number of classes observed in each run.
    """

    labels: np.ndarray
    particle_ids: np.ndarray
    run_labels: list[str]
    n_classes: np.ndarray

    @property
    def n_particles(self) -> int:
        """Number of distinct particles across all runs."""
        return self.labels.shape[0]

    @property
    def n_runs(self) -> int:
        """Number of classification runs."""
        return self.labels.shape[1]

    @property
    def full_participation(self) -> bool:
        """True when every particle is classified in every run."""
        return bool((self.labels != _ABSENT).all())

    def indicator(self) -> csr_matrix:
        """Return the sparse indicator ``B`` with ``M = (1/R) B B^T``.

        Returns
        -------
        scipy.sparse.csr_matrix
            Shape ``(n_particles, sum(n_classes))``, one non-zero per particle
            per run in which that particle appears.
        """
        rows, cols = np.nonzero(self.labels != _ABSENT)
        vals = self.labels[rows, cols]
        return csr_matrix(
            (np.ones(rows.size, dtype=np.float32), (rows, vals)),
            shape=(self.n_particles, int(self.n_classes.sum())),
        )

    def presence(self) -> csr_matrix:
        """Return the sparse presence indicator ``C``; co-occurrence is ``C C^T``."""
        rows, cols = np.nonzero(self.labels != _ABSENT)
        return csr_matrix(
            (np.ones(rows.size, dtype=np.float32), (rows, cols)),
            shape=(self.n_particles, self.n_runs),
        )

    def matrix(self, max_particles: int = 5000, dtype: str = "float32") -> np.ndarray:
        """Materialise the dense ``(N, N)`` co-assignment matrix.

        Only for visualisation of modest particle counts -- everything else in
        this class avoids it. Memory is ``N**2 * itemsize`` bytes.

        Parameters
        ----------
        max_particles : int, default=5000
            Refuse above this, rather than attempting a multi-gigabyte
            allocation. Raise it deliberately if you mean it.
        dtype : str, default='float32'
            Accumulator dtype.

        Returns
        -------
        numpy.ndarray
            ``(N, N)`` matrix; ``M[i, j]`` is the fraction of runs containing
            both particles in which they shared a class. Diagonal is 1.

        Raises
        ------
        UserInputError
            If ``n_particles`` exceeds ``max_particles``.
        """
        n = self.n_particles
        if n > max_particles:
            gib = n * n * np.dtype(dtype).itemsize / 2**30
            raise UserInputError(
                f"Refusing to build a {n} x {n} co-assignment matrix "
                f"({gib:.1f} GiB). Use pca(), consistency_groups() or "
                f"agreement_histogram(), or raise max_particles deliberately."
            )
        b = self.indicator()
        same = np.asarray((b @ b.T).todense(), dtype=dtype)
        if self.full_participation:
            m = same / self.n_runs
        else:
            c = self.presence()
            cooc = np.asarray((c @ c.T).todense(), dtype=dtype)
            with np.errstate(divide="ignore", invalid="ignore"):
                m = np.where(cooc > 0, same / cooc, 0.0).astype(dtype)
        np.fill_diagonal(m, 1.0)
        return m

    def pca(self, n_components: int = 10) -> tuple[np.ndarray, np.ndarray]:
        """Principal coordinates of the co-assignment matrix.

        Exact, not approximate: the eigenvectors of ``M = (1/R) B B^T`` are the
        left singular vectors of ``B``, obtained here from the small
        ``(P, P)`` Gram matrix ``B^T B`` where ``P = sum(n_classes)`` is a few
        dozen. Cost is ``O(N P^2)``; no ``N x N`` array is formed.

        Parameters
        ----------
        n_components : int, default=10
            Number of components; silently capped at ``rank(M)``.

        Returns
        -------
        scores : numpy.ndarray
            ``(N, n_components)`` particle coordinates, components ordered by
            decreasing eigenvalue.
        eigenvalues : numpy.ndarray
            ``(n_components,)`` eigenvalues of ``M``.
        """
        b = self.indicator()
        gram = np.asarray((b.T @ b).todense(), dtype=np.float64)
        w, v = np.linalg.eigh(gram)
        order = np.argsort(w)[::-1]
        w, v = w[order], v[:, order]
        keep = min(n_components, int((w > 1e-12).sum()))
        scores = np.asarray(b @ v[:, :keep], dtype=np.float64)
        return scores, w[:keep] / self.n_runs

    def consistency_groups(self) -> pd.DataFrame:
        """Group particles by their full label tuple across runs.

        Returns
        -------
        pandas.DataFrame
            One row per particle: ``particle_id``, ``group`` (tuple id), and
            ``group_size``.
        """
        _, group, counts = np.unique(
            self.labels, axis=0, return_inverse=True, return_counts=True
        )
        return pd.DataFrame(
            {
                "particle_id": self.particle_ids,
                "group": group.astype(np.int32),
                "group_size": counts[group].astype(np.int32),
            }
        )

    def agreement_histogram(
        self,
        method: str = "auto",
        n_samples: int = 1_000_000,
        max_exact_tuples: int = 5_000,
        seed: int | None = 0,
        max_runs: int | None = None,
    ) -> pd.DataFrame:
        """Distribution of pairwise agreement counts.

        Exact by tuple-based O(T² R) enumeration when the number of distinct
        label tuples T is at most ``max_exact_tuples``; otherwise estimated
        from ``n_samples`` uniformly drawn particle pairs.

        Parameters
        ----------
        method : {'auto', 'exact', 'sampled'}, default 'auto'
            ``'auto'`` picks exact when ``T <= max_exact_tuples``.
        n_samples : int, default 1_000_000
            Pairs drawn when method is ``'sampled'`` or auto chooses sampled.
        max_exact_tuples : int, default 5_000
            T threshold for ``'auto'``.
        seed : int or None, default 0
            RNG seed for the sampled path.
        max_runs : int or None, default None
            Legacy parameter.  When given, raises ``UserInputError`` if
            ``n_runs > max_runs`` (preserves pre-existing test contracts) and
            requires ``full_participation``.

        Returns
        -------
        pandas.DataFrame
            Columns: ``n_runs_agreeing``, ``n_pairs``, ``fraction_of_pairs``,
            ``fraction_at_least_k``, ``exact``.

        Raises
        ------
        UserInputError
            If ``max_runs`` is given and ``n_runs > max_runs``, or if
            participation is incomplete.
        """
        r = self.n_runs

        if max_runs is not None:
            if r > max_runs:
                raise UserInputError(
                    f"agreement_histogram supports up to {max_runs} runs, got {r}."
                )
            if not self.full_participation:
                raise UserInputError(
                    "agreement_histogram requires every particle in every run; "
                    "restrict to the common subset first."
                )
        elif not self.full_participation:
            raise UserInputError(
                "agreement_histogram requires every particle in every run; "
                "restrict to the common subset first."
            )

        tuples, _, counts = np.unique(
            self.labels, axis=0, return_inverse=True, return_counts=True
        )
        T = len(tuples)

        use_exact = method == "exact" or (method == "auto" and T <= max_exact_tuples)

        if use_exact:
            return self._agreement_exact(tuples, counts, r)
        return self._agreement_sampled(n_samples, seed, r)

    def _agreement_exact(self, tuples: np.ndarray, counts: np.ndarray, r: int) -> pd.DataFrame:
        """O(T² R) exact histogram via tuple comparison, chunked to bound memory."""
        T = len(tuples)
        n_pairs_total = self.n_particles * (self.n_particles - 1) // 2
        exact = np.zeros(r + 1, dtype=np.int64)

        # Within-tuple pairs agree in all r runs
        exact[r] += int((counts * (counts - 1) // 2).sum())

        # Across-tuple pairs: process in blocks so peak memory stays bounded
        CHUNK = 256
        for a0 in range(0, T, CHUNK):
            a1 = min(a0 + CHUNK, T)
            a_block = tuples[a0:a1]       # (ca, R)
            a_cnt = counts[a0:a1]         # (ca,)

            # Upper triangle within this block
            ca = a1 - a0
            if ca > 1:
                m = (a_block[:, None, :] == a_block[None, :, :]).sum(axis=2)  # (ca, ca)
                ia, ib = np.triu_indices(ca, k=1)
                w = (a_cnt[ia] * a_cnt[ib]).astype(np.float64)
                bc = np.bincount(m[ia, ib], weights=w, minlength=r + 1)
                exact += bc.astype(np.int64)

            # Cross with all later blocks
            for b0 in range(a1, T, CHUNK):
                b1 = min(b0 + CHUNK, T)
                b_block = tuples[b0:b1]   # (cb, R)
                b_cnt = counts[b0:b1]

                m = (a_block[:, None, :] == b_block[None, :, :]).sum(axis=2)  # (ca, cb)
                w = (a_cnt[:, None] * b_cnt[None, :]).astype(np.float64).ravel()
                bc = np.bincount(m.ravel(), weights=w, minlength=r + 1)
                exact += bc.astype(np.int64)

        return self._histogram_frame(exact, n_pairs_total, r, is_exact=True)

    def _agreement_sampled(self, n_samples: int, seed: int | None, r: int) -> pd.DataFrame:
        """O(n_samples R) sampled histogram."""
        rng = np.random.default_rng(seed)
        n = self.n_particles
        idx_i = rng.integers(0, n, n_samples)
        idx_j = rng.integers(0, n, n_samples)
        keep = idx_i != idx_j
        idx_i, idx_j = idx_i[keep], idx_j[keep]
        li, lj = self.labels[idx_i], self.labels[idx_j]
        agree = ((li == lj) & (li != _ABSENT) & (lj != _ABSENT)).sum(axis=1)
        sample_counts = np.bincount(agree, minlength=r + 1)
        n_pairs_true = n * (n - 1) // 2
        n_sample_pairs = int(keep.sum())
        scale = n_pairs_true / n_sample_pairs if n_sample_pairs > 0 else 1.0
        est = np.round(sample_counts * scale).astype(np.int64)
        return self._histogram_frame(est, n_pairs_true, r, is_exact=False)

    @staticmethod
    def _histogram_frame(counts: np.ndarray, n_pairs: int, r: int, *, is_exact: bool) -> pd.DataFrame:
        frac = counts / n_pairs if n_pairs else np.zeros(r + 1)
        cum = np.array([float(frac[k:].sum()) for k in range(r + 1)])
        return pd.DataFrame(
            {
                "n_runs_agreeing": np.arange(r + 1),
                "n_pairs": counts,
                "fraction_of_pairs": frac,
                "fraction_at_least_k": cum,
                "exact": is_exact,
            }
        )

    def run_agreement(self) -> dict:
        """Pairwise adjusted Rand index between every pair of runs.

        Returns
        -------
        dict with keys:

        - ``"summary"`` : DataFrame[run, n_classes, mean_ari, min_ari, max_ari]
        - ``"matrix"``  : R×R DataFrame of pairwise ARI indexed by run label
        """
        from sklearn.metrics import adjusted_rand_score

        R = self.n_runs
        labels = self.run_labels or [str(i) for i in range(R)]
        ari = np.zeros((R, R), dtype=np.float64)
        for a in range(R):
            for b in range(a + 1, R):
                v = adjusted_rand_score(self.labels[:, a], self.labels[:, b])
                ari[a, b] = ari[b, a] = v
        np.fill_diagonal(ari, 1.0)

        off_diag = ari.copy()
        np.fill_diagonal(off_diag, np.nan)

        rows = []
        for r in range(R):
            others = off_diag[r][~np.isnan(off_diag[r])]
            rows.append({
                "run": labels[r],
                "n_classes": int(self.n_classes[r]),
                "mean_ari": float(others.mean()) if len(others) else 0.0,
                "min_ari": float(others.min()) if len(others) else 0.0,
                "max_ari": float(others.max()) if len(others) else 0.0,
            })

        return {
            "summary": pd.DataFrame(rows),
            "matrix": pd.DataFrame(ari, index=labels, columns=labels),
        }


def build_coassignment_factor(
    dataframes: list[pd.DataFrame],
    particle_column: str = "subtomo_id",
    class_column: str = "class",
    run_labels: list[str] | None = None,
) -> CoassignmentFactor:
    """Build the factored co-assignment representation from several runs.

    Particles are matched across runs by ``particle_column``, never by row
    order.  Class labels need not be consistent between runs; co-assignment is
    permutation invariant.

    Parameters
    ----------
    dataframes : list of pandas.DataFrame
        One per classification run; each must contain both columns.
    particle_column : str, default='subtomo_id'
        Column holding particle identifiers, unique within each run.
    class_column : str, default='class'
        Column holding the within-run class assignment.
    run_labels : list of str, optional
        Names for the runs, in order.  Defaults to ``run_1 ... run_R``.

    Returns
    -------
    CoassignmentFactor

    Raises
    ------
    UserInputError
        If fewer than two runs are given, a required column is missing, a run
        is empty, or a run repeats a particle identifier.
    """
    if len(dataframes) < 2:
        raise UserInputError("At least two classification runs are required.")
    if run_labels is None:
        run_labels = [f"run_{i + 1}" for i in range(len(dataframes))]
    if len(run_labels) != len(dataframes):
        raise UserInputError("run_labels must have one entry per dataframe.")

    for label, df in zip(run_labels, dataframes):
        missing = {particle_column, class_column} - set(df.columns)
        if missing:
            raise UserInputError(f"Run {label!r} is missing column(s): {sorted(missing)}.")
        if df.empty:
            raise UserInputError(f"Run {label!r} is empty.")
        if df[particle_column].duplicated().any():
            raise UserInputError(
                f"Run {label!r} repeats particle identifiers in {particle_column!r}; "
                "renumber the motl first."
            )

    particle_ids = np.unique(
        np.concatenate([df[particle_column].to_numpy() for df in dataframes])
    )
    n = particle_ids.size

    labels = np.full((n, len(dataframes)), _ABSENT, dtype=np.int32)
    n_classes = np.zeros(len(dataframes), dtype=np.int32)
    offset = 0
    for r, df in enumerate(dataframes):
        idx = np.searchsorted(particle_ids, df[particle_column].to_numpy())
        _, class_idx = np.unique(df[class_column].to_numpy(), return_inverse=True)
        k = int(class_idx.max()) + 1
        labels[idx, r] = class_idx + offset
        n_classes[r] = k
        offset += k

    return CoassignmentFactor(
        labels=labels,
        particle_ids=particle_ids,
        run_labels=list(run_labels),
        n_classes=n_classes,
    )


@_dataclass(frozen=True)
class ConsensusResult:
    """Output of :func:`consensus_groups`.

    Attributes
    ----------
    labels : numpy.ndarray
        ``(N,)`` int32 consensus class per particle; ``junk_class`` where
        unassigned.
    particle_ids : numpy.ndarray
        ``(N,)`` particle identifiers matching the input factor.
    group_sizes : pandas.Series
        Mapping class → particle count (including junk class if non-empty).
    min_agreement : float
        The threshold actually used, after snapping to the nearest achievable
        ``k/R``.
    linkage : str
        Linkage rule used.
    method : str
        Which computation ladder row ran.
    n_assigned : int
        Particles assigned to a non-junk class.
    n_junk : int
        Particles collapsed to the junk class.
    reliable : bool
        Whether the ensemble supports a stable grouping.
    verdict : str
        One human-readable sentence summarising the result.
    junk_class : int
        The class value used for unassigned / small-group particles.
    """

    labels: np.ndarray
    particle_ids: np.ndarray
    group_sizes: pd.Series
    min_agreement: float
    linkage: str
    method: str
    n_assigned: int
    n_junk: int
    reliable: bool
    verdict: str
    junk_class: int


def _snap_agreement(min_agreement: float, n_runs: int) -> tuple[float, int]:
    """Round min_agreement up to the nearest achievable k/R; return (t, k)."""
    k = int(np.ceil(min_agreement * n_runs))
    k = max(0, min(k, n_runs))
    return k / n_runs, k


def _spectral_apply_threshold(
    all_labels: np.ndarray, raw_labels: np.ndarray, k: int, r: int
) -> None:
    """In-place: set ``raw_labels[i] = -1`` when particle *i* agrees with its
    cluster's modal label in fewer than *k* of *r* runs."""
    for cid in np.unique(raw_labels):
        if cid < 0:
            continue
        mask = raw_labels == cid
        ml = all_labels[mask]
        consensus = np.full(r, _ABSENT, dtype=np.int32)
        for ri in range(r):
            col = ml[:, ri]
            valid = col[col != _ABSENT]
            if len(valid):
                min_v = int(valid.min())
                bc = np.bincount((valid - min_v).astype(np.intp))
                consensus[ri] = min_v + int(bc.argmax())
        agree = (
            (ml == consensus[None, :])
            & (ml != _ABSENT)
            & (consensus[None, :] != _ABSENT)
        ).sum(axis=1)
        member_idx = np.where(mask)[0]
        raw_labels[member_idx[agree < k]] = -1


def _spectral_n_clusters(eigenvalues: np.ndarray) -> int:
    """Estimate number of clusters from the eigenvalue elbow."""
    if eigenvalues.size <= 1 or eigenvalues[0] <= 0:
        return 1
    gaps = eigenvalues[:-1] - eigenvalues[1:]
    return int(np.argmax(gaps)) + 1


def _histogram_shape(fracs: np.ndarray) -> str:
    """Classify an agreement histogram as 'flat', 'decaying', or 'structured'."""
    f_top = float(fracs[-1])
    f_max = float(fracs.max())
    f_min = float(fracs.min())
    if f_max - f_min < 0.05:
        return "flat"
    if f_top < 0.01:
        return "decaying"
    return "structured"


def _verdict_from_shape(shape: str, n_runs: int) -> tuple[bool, str]:
    if shape == "flat":
        return (
            False,
            f"Co-assignment fractions are approximately uniform across all {n_runs} agreement "
            "levels; the classifications appear mutually uninformative and do not support a consensus.",
        )
    if shape == "decaying":
        return (
            True,
            f"No particle pair is co-assigned in all {n_runs} runs; "
            "a partial-agreement threshold is required and the consensus will be approximate.",
        )
    return (
        True,
        f"Agreement is concentrated at k={n_runs} (always together) and k=0 (never); "
        "strict co-assignment grouping should work well.",
    )


def _compute_reliability(
    factor: CoassignmentFactor,
) -> tuple[bool, pd.DataFrame | None, np.ndarray, str]:
    """Return (reliable, histogram, eigenvalues, verdict)."""
    R = factor.n_runs
    histogram = None
    reliable = True
    verdict = ""

    if R <= 12 and factor.full_participation:
        try:
            histogram = factor.agreement_histogram()
            fracs = histogram["fraction_of_pairs"].to_numpy()
            shape = _histogram_shape(fracs)
            reliable, verdict = _verdict_from_shape(shape, R)
        except UserInputError:
            pass

    n_comp = min(20, int(factor.n_classes.sum()))
    _, eigenvalues = factor.pca(n_components=n_comp)

    if histogram is None:
        gap = float(eigenvalues[0] / (eigenvalues.sum() + 1e-12)) if eigenvalues.size > 0 else 0.0
        if gap > 0.5:
            reliable = True
            verdict = (
                f"Spectral gap {gap:.2f} suggests stable structure "
                f"(agreement histogram skipped: R={R} > 12)."
            )
        elif gap > 0.2:
            reliable = True
            verdict = (
                f"Moderate spectral gap {gap:.2f}; structure present but "
                f"some runs may be inconsistent (R={R} > 12, histogram skipped)."
            )
        else:
            reliable = False
            verdict = (
                f"Low spectral gap {gap:.2f}; classifications appear mutually "
                f"uninformative (R={R} > 12, histogram skipped)."
            )

    return reliable, histogram, eigenvalues, verdict


def consensus_groups(
    factor: CoassignmentFactor,
    min_agreement: float = 1.0,
    linkage: str = "complete",
    min_group_size: int = 10,
    junk_class: int = 0,
) -> ConsensusResult:
    """Group particles that co-occur in at least ``min_agreement`` of the runs.

    Parameters
    ----------
    factor : CoassignmentFactor
        Built by :func:`build_coassignment_factor`.
    min_agreement : float, default=1.0
        Fraction of runs in which a pair must share a class to be grouped
        together.  Snapped up to the nearest achievable ``k/R`` value.
    linkage : {'complete', 'average', 'single'}, default='complete'
        Agglomerative linkage rule.  Ignored at ``min_agreement=1.0``.
        ``'single'`` is warned against: a single weak link can chain two
        genuine classes.
    min_group_size : int, default=10
        Groups smaller than this collapse to ``junk_class``.
    junk_class : int, default=0
        Class label written for unassigned / small-group particles.

    Returns
    -------
    ConsensusResult
    """
    if linkage not in ("complete", "average", "single"):
        raise UserInputError(
            f"linkage must be 'complete', 'average', or 'single'; got {linkage!r}."
        )
    if linkage == "single":
        warnings.warn(
            "single linkage can chain distant classes via a single weak link; "
            "consider 'complete' or 'average'.",
            UserWarning,
            stacklevel=2,
        )

    R = factor.n_runs
    N = factor.n_particles
    t_actual, k = _snap_agreement(min_agreement, R)

    reliable = True
    if k == R:
        cg = factor.consistency_groups()
        raw_labels = cg["group"].to_numpy().astype(np.int32)
        method = "exact-tuple"

    elif N <= _MATRIX_GUARD:
        import scipy.cluster.hierarchy as sch
        from scipy.spatial.distance import squareform

        m = factor.matrix(max_particles=_MATRIX_GUARD)
        dist_condensed = squareform(1.0 - m, checks=False)
        z = sch.linkage(dist_condensed, method=linkage)
        raw_labels = (sch.fcluster(z, t=1.0 - t_actual, criterion="distance") - 1).astype(
            np.int32
        )
        method = f"scipy-{linkage}"
        n_comp_r = min(20, int(factor.n_classes.sum()))
        _, evals_r = factor.pca(n_components=n_comp_r)
        gap = float(evals_r[0] / (evals_r.sum() + 1e-12)) if evals_r.size > 0 else 0.0
        reliable = gap > 0.2

    else:
        from sklearn.cluster import KMeans

        n_comp = min(50, int(factor.n_classes.sum()))
        scores, evals = factor.pca(n_components=n_comp)
        n_clusters = max(2, _spectral_n_clusters(evals))
        km = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto")
        raw_labels = km.fit_predict(scores).astype(np.int32)
        method = f"spectral-kmeans(seed=0,k={n_clusters})"
        gap = float(evals[0] / (evals.sum() + 1e-12)) if evals.size > 0 else 0.0
        reliable = gap > 0.2
        if k < R:
            _spectral_apply_threshold(factor.labels, raw_labels, k, R)

    unique_raw, raw_counts = np.unique(raw_labels, return_counts=True)
    small = set(unique_raw[raw_counts < min_group_size].tolist())
    collapsed = np.where(np.isin(raw_labels, list(small) if small else []), -1, raw_labels)

    valid_mask = collapsed != -1
    final_labels = np.full(N, junk_class, dtype=np.int32)
    if valid_mask.any():
        valid_unique, valid_counts = np.unique(collapsed[valid_mask], return_counts=True)
        order = np.argsort(valid_counts)[::-1]
        sorted_raw = valid_unique[order]
        raw_to_final = {int(r): junk_class + 1 + i for i, r in enumerate(sorted_raw)}
        for i, raw in enumerate(collapsed):
            if raw != -1:
                final_labels[i] = raw_to_final[int(raw)]

    all_unique, all_counts = np.unique(final_labels, return_counts=True)
    group_sizes = pd.Series(all_counts, index=all_unique, name="n_particles")
    n_junk = int((final_labels == junk_class).sum())
    n_assigned = N - n_junk
    n_groups = int((all_unique != junk_class).sum())

    verdict = (
        f"{n_assigned}/{N} particles assigned to {n_groups} group(s) "
        f"at t={t_actual:.3f} ({method}); {n_junk} junk. "
        + ("Stable ensemble." if reliable else "Ensemble reliability: uncertain.")
    )

    return ConsensusResult(
        labels=final_labels,
        particle_ids=factor.particle_ids,
        group_sizes=group_sizes,
        min_agreement=t_actual,
        linkage=linkage,
        method=method,
        n_assigned=n_assigned,
        n_junk=n_junk,
        reliable=reliable,
        verdict=verdict,
        junk_class=junk_class,
    )


def reliability_summary(factor: CoassignmentFactor) -> dict:
    """Agreement histogram plus a verdict on whether the ensemble supports a
    stable grouping.

    Returns
    -------
    dict with keys:
        ``histogram``   — ``pd.DataFrame | None``
        ``eigenvalues`` — ``np.ndarray``
        ``reliable``    — ``bool``
        ``verdict``     — ``str``
        ``n_runs``      — ``int``
        ``n_particles`` — ``int``
    """
    reliable, histogram, eigenvalues, verdict = _compute_reliability(factor)
    if histogram is None:
        try:
            histogram = factor.agreement_histogram()
        except Exception:
            pass
    return {
        "histogram": histogram,
        "eigenvalues": eigenvalues,
        "reliable": reliable,
        "verdict": verdict,
        "n_runs": factor.n_runs,
        "n_particles": factor.n_particles,
    }


def consensus_motl(
    result: ConsensusResult,
    source_motl,
    class_column: str = "class",
    keep_junk: bool = True,
):
    """Geometry from ``source_motl``, consensus classes written to ``class_column``.

    Parameters
    ----------
    result : ConsensusResult
        Output of :func:`consensus_groups`.
    source_motl : Motl
        Provides the particle geometry; matched by ``subtomo_id``.
    class_column : str, default='class'
        Column in the motl DataFrame to overwrite with consensus labels.
    keep_junk : bool, default=True
        If False, drop particles absent from the factor or in the junk class.

    Returns
    -------
    Motl
        A copy of ``source_motl`` with ``class_column`` updated.
    """
    motl = copy.copy(source_motl)
    motl.df = source_motl.df.copy()

    all_pids = motl.df["subtomo_id"].to_numpy()
    idx = np.searchsorted(result.particle_ids, all_pids)
    idx_clipped = np.clip(idx, 0, len(result.particle_ids) - 1)
    in_factor = result.particle_ids[idx_clipped] == all_pids
    labels = np.where(in_factor, result.labels[idx_clipped], result.junk_class).astype(np.int32)
    motl.df[class_column] = labels

    if not keep_junk:
        motl.df = motl.df[motl.df[class_column] != result.junk_class].reset_index(drop=True)

    return motl
