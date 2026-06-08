import math
import re
from pathlib import Path, PurePosixPath
from typing import Any

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
                load_kwargs=load_kwargs
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

    for i in np.arange(0, end_it-start_it): ## FIXME this fixes 'index out of range' when start_it=!0, but does not account for the correct plot labels in such case (when called by evaluate_alignment)
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


def create_multiref_run(
    input_motl: MotlSource,
    number_of_classes: int,
    output_motl_base: PathOrStr,
    input_motl_type: MotlType = "emmotl",
    iteration_number: int = 1,
    number_of_runs: int = 1,
    output_motl_type: str = "stopgap",
) -> None:
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
    None

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
    motl.df.fillna(0.0)

    for i in range(1, number_of_runs + 1):
        # create motl with randomly assigned classes
        motl.assign_random_classes(number_of_classes)

        output_path = output_motl_base + "_mr" + str(i) + "_" + str(iteration_number)
        write_out_motl(motl, output_path, output_motl_type=output_motl_type)


def create_denovo_multiref_run(
    input_motl: MotlSource,
    number_of_classes: int,
    output_motl_base: PathOrStr,
    input_motl_type: MotlType = "emmotl",
    class_occupancy: int | None = None,
    iteration_number: int = 1,
    number_of_runs: int = 1,
    output_motl_type: str = "stopgap",
) -> None:
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
    None

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
    motl.df.fillna(0.0)

    n_particles = motl.df.shape[0]

    # create motl for reference creation
    if class_occupancy is None:
        class_occupancy = np.ceil(n_particles / 10)

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

    # create motl with randomly assigned classes
    motl.assign_random_classes(number_of_classes)
    write_out_motl(
        motl, output_file_base=output_motl_base + "_" + str(iteration_number), output_motl_type=output_motl_type
    )

    #instead of calling conversion functions kind of the same is happening

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
        merged.columns = [f'{col[0]}_{col[1]}' if col[1] else col[0]
                          for col in merged.columns]
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


# ── Internal column/key mapping tables ────────────────────────────────────────

# Canonical (novaSTA-convention) column names for df
# Run-level novaSTA keys — not broadcast per iteration, not stored in df
_NOVA_RUN_LEVEL_KEYS = {"iter", "startIndex", "createRef"}

# Raw STOPGAP angle columns — converted to unified names on load, back on write
_STOPGAP_ANGLE_COLS = {"_angiter", "_angincr", "_phi_angiter", "_phi_angincr"}

# One cross-format map: STOPGAP display name → novaSTA display name.
# Columns whose display names already match (e.g. 'iteration', 'symmetry',
# 'binning', 'cone angle', 'cone sampling', ...) are NOT listed here.
_SG_TO_NOVA_NAME = {
    "motl name":      "motl",
    "ref name":       "ref",
    "mask name":      "mask",
    "ccmask name":    "cc mask",
    "wedgelist name": "wedge list",
    "lp rad":         "low pass",
    "hp rad":         "high pass",
    "score thresh":   "threshold",
}
_NOVA_TO_SG_NAME = {v: k for k, v in _SG_TO_NOVA_NAME.items()}


# ── Angle conversion helpers ──────────────────────────────────────────────────

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


# ── Internal helpers ──────────────────────────────────────────────────────────

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


def _normalize_stopgap_mode(mode_str: str) -> str:
    """Normalise STOPGAP _subtomo_mode aliases to canonical form."""
    m = str(mode_str).strip()
    return {"ali_multiref": "multiref_ali", "avg_multiref": "multiref_avg"}.get(m, m)


def _stopgap_col_to_name(col: str) -> str:
    """``_motl_name`` → ``'motl name'``; strips leading underscore, replaces the rest with spaces."""
    return col.lstrip("_").replace("_", " ")


def _nova_key_to_name(key: str) -> str:
    """Convert a novaSTA camelCase key to a space-separated display name.

    ``coneAngle`` → ``'cone angle'``,  ``useGPU`` → ``'use GPU'``.
    Consecutive capitals (acronyms) are kept together.
    """
    s = re.sub(r'([a-z0-9])([A-Z])', r'\1 \2', key)
    s = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1 \2', s)
    return ' '.join(w if (w.isupper() and len(w) > 1) else w.lower() for w in s.split())


def _display_to_nova_key(name: str) -> str:
    """Inverse of ``_nova_key_to_name``: ``'cone angle'`` → ``'coneAngle'``, ``'use GPU'`` → ``'useGPU'``."""
    words = name.split()
    result = words[0].lower()
    for w in words[1:]:
        result += w if w.isupper() else w.capitalize()
    return result


def _fmt_val(v: Any) -> str:
    """Format a scalar for novaSTA output (drop trailing .0 from whole floats)."""
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


# ── Base class ────────────────────────────────────────────────────────────────

class StaParameters:
    """Base class for STA parameter file representations.

    Holds two parallel one-row-per-alignment-iteration DataFrames.

    Attributes
    ----------
    df : pandas.DataFrame
        Canonical columns, one row per *alignment* iteration.
    df_extra : pandas.DataFrame
        Format-specific extra columns (same index as ``df``).
    df_stats : pandas.DataFrame or None
        Per-iteration RMSE statistics populated by :meth:`attach_log`.
    fsc : pandas.DataFrame or None
        FSC curve populated by :meth:`attach_fsc`.
    create_ref : bool
        Whether an averaging pre-step should be materialised on write.
    multiref : bool
        Whether multi-reference mode is active.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        df_extra: pd.DataFrame | None = None,
        create_ref: bool = False,
        multiref: bool = False,
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
        self.multiref = bool(multiref)

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
        """Construct an StaParameters from a parameter dictionary (GUI path).

        Parameters
        ----------
        params : dict
            Keyed by canonical column names or novaSTA file key names.  Values
            may be scalars, lists, or whitespace-separated strings.

            **Mandatory keys:** ``cone_angle``, ``cone_sampling``,
            ``inplane_angle``, ``inplane_sampling``, ``high_pass``,
            ``start_index``.

            **Control keys (removed before building df):** ``start_index``
            (int, default 1), ``create_ref`` (bool, default False),
            ``multiref`` (bool, default False).

        sta_type : str, default="novasta"
            Target subclass (``"stopgap"`` or ``"novasta"``).

        Returns
        -------
        StopgapParams or NovaStaParams

        Raises
        ------
        ValueError
            If mandatory keys are missing, or if sequence lengths disagree.
        """
        # Mandatory keys in novaSTA display-name form
        mandatory = {"cone angle", "cone sampling", "inplane angle",
                     "inplane sampling", "high pass", "start index"}

        # Normalise: snake_case or camelCase → novaSTA display name
        normalised = {}
        for k, v in params.items():
            display = _nova_key_to_name(k).replace("_", " ")
            normalised[display] = v

        missing = mandatory - set(normalised.keys())
        if missing:
            raise ValueError(f"Missing mandatory parameter(s): {sorted(missing)}")

        start_index = int(_parse_value_list(normalised.pop("start index", 1))[0])
        create_ref  = bool(int(_parse_value_list(normalised.pop("create ref", 0))[0]))
        multiref    = bool(int(_parse_value_list(normalised.pop("multiref", 0))[0]))

        # Parse all values into lists
        parsed = {k: _parse_value_list(v) for k, v in normalised.items()}

        # Infer n_align from longest sequence
        lengths = {k: len(v) for k, v in parsed.items() if len(v) > 1}
        n_align = max(lengths.values()) if lengths else (
            1 if all(len(v) == 1 for v in parsed.values()) else 0
        )
        bad = {k: l for k, l in lengths.items() if l != n_align}
        if bad:
            raise ValueError(
                f"All per-iteration sequences must share the same length "
                f"({n_align}). Mismatched keys: {bad}"
            )

        # Broadcast scalars to n_align rows
        expanded = {
            k: (v * n_align if len(v) == 1 else list(v))
            for k, v in parsed.items()
        }

        # Build df with novaSTA display names
        iters = list(range(start_index, start_index + n_align))
        df_data = {"iteration": iters, **{k: expanded[k] for k in expanded}}
        df = pd.DataFrame(df_data) if n_align > 0 else pd.DataFrame(columns=["iteration"])

        # For STOPGAP, rename shared columns to STOPGAP display names
        sta_type_lower = (sta_type or "novasta").lower()
        if sta_type_lower == "stopgap":
            df = df.rename(columns=_NOVA_TO_SG_NAME)

        klass = StopgapParams if sta_type_lower == "stopgap" else NovaStaParams
        return klass(df, pd.DataFrame(), create_ref=create_ref, multiref=multiref)

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
        """Return a NovaStaParams with columns renamed to novaSTA display names."""
        df = self.df.rename(columns=_SG_TO_NOVA_NAME)
        return NovaStaParams(df, pd.DataFrame(), create_ref=self.create_ref, multiref=self.multiref)

    def to_stopgap(self) -> "StopgapParams":
        """Return a StopgapParams with columns renamed to STOPGAP display names."""
        df = self.df.rename(columns=_NOVA_TO_SG_NAME)
        return StopgapParams(df, pd.DataFrame(), create_ref=self.create_ref, multiref=self.multiref)


# ── StopgapParams ──────────────────────────────────────────────────────────────

class StopgapParams(StaParameters):
    """STOPGAP STAR-file subtomogram-averaging parameter representation.

    All STOPGAP columns are stored in ``df`` with display names derived by
    stripping the leading underscore and replacing remaining underscores with
    spaces (e.g. ``_lp_sigma`` → ``'lp sigma'``).  Raw angle-iteration columns
    are converted to unified angle-extent names matching novaSTA convention
    (``'cone angle'``, ``'cone sampling'``, ``'inplane angle'``,
    ``'inplane sampling'``).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        df_extra: pd.DataFrame | None = None,
        create_ref: bool = False,
        multiref: bool = False,
    ) -> None:
        super().__init__(df, df_extra, create_ref=create_ref, multiref=multiref)
        self._orig_columns = None  # set by from_file for round-trip column order

    @property
    def motl_type(self) -> str:
        return "stopgap"

    def get_motl_base_name(
        self,
        separator: str = "_",
        working_dir: PathOrStr | None = None,
    ) -> str | None:
        """STOPGAP motl base name -- ``rootdir/lists/<motl name><separator>``.

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
        col = "motl name"
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
        """STOPGAP wedge list path -- ``rootdir/lists/<wedgelist name>``.

        Parameters
        ----------
        working_dir : PathOrStr or None, default=None
            Overrides the ``rootdir`` column when supplied.

        Returns
        -------
        str or None
        """
        return self._resolve_in_subdir("wedgelist name", "lists", working_dir)

    def resolve_mask(self, working_dir: PathOrStr | None = None) -> str | None:
        """STOPGAP particle mask path -- ``rootdir/masks/<mask name>``.

        Parameters
        ----------
        working_dir : PathOrStr or None, default=None
            Overrides the ``rootdir`` column when supplied.

        Returns
        -------
        str or None
        """
        return self._resolve_in_subdir("mask name", "masks", working_dir)

    def resolve_ccmask(self, working_dir: PathOrStr | None = None) -> str | None:
        """STOPGAP CC mask path -- ``rootdir/masks/<ccmask name>``.

        Parameters
        ----------
        working_dir : PathOrStr or None, default=None
            Overrides the ``rootdir`` column when supplied.

        Returns
        -------
        str or None
        """
        return self._resolve_in_subdir("ccmask name", "masks", working_dir)

    def resolve_ref_base(
        self,
        working_dir: PathOrStr | None = None,
        separator: str = "_",
    ) -> str | None:
        """STOPGAP reference base name -- ``rootdir/refs/<ref name><separator>``.

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
        col = "ref name"
        if col not in self.df.columns or self.df.empty:
            return None
        val = self.df[col].iloc[0]
        if _is_none_val(val):
            return None
        root = self._effective_rootdir(working_dir)
        if root is None:
            return str(val) + separator
        return str(Path(root) / "refs" / str(val)) + separator

    # ── Default STOPGAP column order (used when _orig_columns is not set) ──
    _DEFAULT_STOPGAP_COLS = [
        "_completed_ali", "_completed_p_avg", "_completed_f_avg",
        "_iteration", "_subtomo_mode",
        "_rootdir", "_motl_name", "_wedgelist_name", "_binning",
        "_ref_name", "_subtomo_name", "_mask_name", "_ccmask_name",
        "_search_mode", "_angincr", "_angiter", "_phi_angincr", "_phi_angiter",
        "_cone_search_type", "_apply_laplacian",
        "_lp_rad", "_lp_sigma", "_hp_rad", "_hp_sigma",
        "_calc_exp", "_calc_ctf", "_cos_weight", "_score_weight",
        "_symmetry", "_score_thresh", "_subset", "_avg_mode",
        "_ignore_halfsets", "_temperature",
    ]

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
            If True, keep only rows where ``_completed_ali == 1``.

        Returns
        -------
        StopgapParams
        """
        frame, spec, _ = Starfile.read(path, data_id=0)
        expected = "data_stopgap_subtomo_parameters"
        if spec != expected:
            raise ValueError(
                f"Not a valid STOPGAP subtomo parameter file: "
                f"specifier is {spec!r}, expected {expected!r}."
            )
        orig_columns = list(frame.columns)

        if load_completed_only and "_completed_ali" in frame.columns:
            frame = frame[frame["_completed_ali"] == 1].reset_index(drop=True)

        if "_subtomo_mode" in frame.columns:
            frame["_subtomo_mode"] = frame["_subtomo_mode"].apply(_normalize_stopgap_mode)
            ali_mask = frame["_subtomo_mode"].str.endswith("_ali")
            multiref = frame["_subtomo_mode"].str.startswith("multi").any()
            create_ref = (~ali_mask).any()
        else:
            ali_mask = pd.Series([True] * len(frame))
            multiref = False
            create_ref = False

        ali_frame = frame[ali_mask].reset_index(drop=True)

        # Convert angle iteration counts → unified angle-extent columns
        if _STOPGAP_ANGLE_COLS <= set(ali_frame.columns):
            def _conv_angles(row):
                ai, ac = row["_angiter"], row["_angincr"]
                pai, pac = row["_phi_angiter"], row["_phi_angincr"]
                if any(_is_none_val(x) for x in (ai, ac, pai, pac)):
                    return pd.Series([None, None, None, None],
                                     index=["cone angle", "cone sampling",
                                            "inplane angle", "inplane sampling"])
                try:
                    ca, cs, ia, is_ = stopgap_to_nova_angles(
                        float(ai), float(ac), float(pai), float(pac)
                    )
                except (ValueError, TypeError):
                    ca = cs = ia = is_ = None
                return pd.Series([ca, cs, ia, is_],
                                 index=["cone angle", "cone sampling",
                                        "inplane angle", "inplane sampling"])

            angle_df = ali_frame.apply(_conv_angles, axis=1)
            ali_frame = ali_frame.drop(columns=list(_STOPGAP_ANGLE_COLS))
            ali_frame = pd.concat([ali_frame.reset_index(drop=True),
                                   angle_df.reset_index(drop=True)], axis=1)

        # Rename all remaining columns: strip leading _, replace _ with space
        ali_frame = ali_frame.rename(columns={c: _stopgap_col_to_name(c)
                                              for c in ali_frame.columns})

        obj = cls(ali_frame, pd.DataFrame(), create_ref=create_ref, multiref=multiref)
        obj._orig_columns = orig_columns
        return obj

    def write_out(
        self,
        path: PathOrStr,
        create_ref: bool | None = None,
        multiref: bool | None = None,
        total_iterations: int | None = None,
    ) -> None:
        """Write a STOPGAP subtomogram parameter STAR file.

        Parameters
        ----------
        path : PathOrStr
            Destination path for the ``.star`` file.
        create_ref : bool or None, default=None
            If ``None``, uses the value stored on the object.
        multiref : bool or None, default=None
            If ``None``, uses the value stored on the object.
        total_iterations : int or None, default=None
            If larger than the current alignment-iteration count, pad with
            extra rows (params copied from the last row, flags 0).
        """
        cr = self.create_ref if create_ref is None else bool(create_ref)
        mr = self.multiref  if multiref  is None else bool(multiref)
        family = "multiref" if mr else "singleref"

        out_cols = self._orig_columns if self._orig_columns else self._DEFAULT_STOPGAP_COLS

        ali_df = self.df.reset_index(drop=True)

        # Optional padding
        if total_iterations is not None and total_iterations > len(ali_df):
            n_pad = total_iterations - len(ali_df)
            last_it = int(ali_df["iteration"].iloc[-1]) if not ali_df.empty else (self.start_iteration or 1)
            pad_ali = ali_df.iloc[[-1] * n_pad].copy().reset_index(drop=True)
            pad_ali["iteration"] = [last_it + i + 1 for i in range(n_pad)]
            ali_df = pd.concat([ali_df, pad_ali], ignore_index=True)

        rows = []

        # Leading avg row
        if cr:
            if not ali_df.empty:
                first_it = int(ali_df["iteration"].iloc[0])
                rows.append(self._build_row(ali_df.iloc[0], family, is_avg=True,
                                            out_cols=out_cols, iteration=first_it))
            else:
                row = {c: "none" for c in out_cols}
                row.update(_completed_ali=0, _completed_p_avg=0, _completed_f_avg=0,
                           _iteration=self.start_iteration or 1,
                           _subtomo_mode=f"{family}_avg")
                rows.append(row)

        for _, df_row in ali_df.iterrows():
            rows.append(self._build_row(df_row, family, is_avg=False, out_cols=out_cols))

        out_df = pd.DataFrame(rows, columns=out_cols)
        out_df = out_df.where(out_df.notna(), other="none")
        out_df = out_df.replace({None: "none"})

        Starfile.write([out_df], path, specifiers=["data_stopgap_subtomo_parameters"])

    def _build_row(
        self,
        df_row: pd.Series,
        family: str,
        is_avg: bool,
        out_cols: list[str],
        iteration: int | None = None,
    ) -> dict:
        """Assemble one STOPGAP output row dict from a display-name df row."""
        _angle_display = {"cone angle", "cone sampling", "inplane angle", "inplane sampling"}
        _skip_display  = {"iteration", "subtomo mode"} | _angle_display

        row = {}
        row["_completed_ali"]   = 0
        row["_completed_p_avg"] = 0
        row["_completed_f_avg"] = 0
        row["_iteration"]       = iteration if iteration is not None else df_row.get("iteration")
        row["_subtomo_mode"]    = f"{family}_{'avg' if is_avg else 'ali'}"

        # Angle columns
        if is_avg:
            row["_angincr"] = row["_angiter"] = row["_phi_angincr"] = row["_phi_angiter"] = None
        else:
            ca  = df_row.get("cone angle")
            cs  = df_row.get("cone sampling")
            ia  = df_row.get("inplane angle")
            isp = df_row.get("inplane sampling")
            if not any(_is_none_val(x) for x in (ca, cs, ia, isp)):
                ai, ac, pai, pac = nova_to_stopgap_angles(
                    float(ca), float(cs), float(ia), float(isp)
                )
                row.update(_angiter=ai, _angincr=ac, _phi_angiter=pai, _phi_angincr=pac)
            else:
                row["_angincr"] = row["_angiter"] = row["_phi_angincr"] = row["_phi_angiter"] = None

        # All other display-name columns → _underscore_name; only write if in out_cols
        for display_col, v in df_row.items():
            if display_col in _skip_display:
                continue
            sg_col = "_" + display_col.replace(" ", "_")
            if sg_col in out_cols and sg_col not in row:
                row[sg_col] = None if _is_none_val(v) else v

        # Fill any remaining output columns
        for col in out_cols:
            row.setdefault(col, None)

        return row


# ── NovaStaParams ──────────────────────────────────────────────────────────────

class NovaStaParams(StaParameters):
    """novaSTA flat key-value parameter file representation.

    All parameters are stored in ``df`` with display names derived by converting
    camelCase keys to space-separated words (e.g. ``coneAngle`` → ``'cone angle'``,
    ``useGPU`` → ``'use GPU'``).  Run-level keys (``iter``, ``startIndex``,
    ``createRef``) become object attributes, not columns.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        df_extra: pd.DataFrame | None = None,
        create_ref: bool = False,
        multiref: bool = False,
    ) -> None:
        super().__init__(df, df_extra, create_ref=create_ref, multiref=multiref)
        self._orig_key_order = None  # preserved for write-order hints

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
        raw = {}
        key_order = []
        with open(path, "r") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                key  = parts[0]
                vals = [_parse_scalar(p) for p in parts[1:]]
                if key not in raw:
                    key_order.append(key)
                raw[key] = vals  # last occurrence wins for duplicates

        n_align     = int(raw.get("iter", [1])[0])
        start_index = int(raw.get("startIndex", [1])[0])
        create_ref  = bool(int(raw.get("createRef", [0])[0]))

        # Validate 1-or-N rule
        for k, v in raw.items():
            if k in _NOVA_RUN_LEVEL_KEYS:
                continue
            if n_align > 0 and len(v) not in (1, n_align):
                raise ValueError(
                    f"Parameter {k!r} has {len(v)} value(s) but iter={n_align}. "
                    f"Each parameter must supply 1 or {n_align} values."
                )

        def broadcast(vals):
            return vals * max(n_align, 1) if len(vals) == 1 else vals

        iters = list(range(start_index, start_index + max(n_align, 0)))

        # Build df: all non-run-level keys as display-name columns
        df_data = {"iteration": iters}
        for k in key_order:
            if k in _NOVA_RUN_LEVEL_KEYS:
                continue
            df_data[_nova_key_to_name(k)] = broadcast(raw[k])[:n_align] if n_align > 0 else []

        df = pd.DataFrame(df_data) if n_align > 0 else pd.DataFrame(columns=["iteration"])

        obj = cls(df, pd.DataFrame(), create_ref=create_ref, multiref=False)
        obj._orig_key_order = key_order
        return obj

    def write_out(
        self,
        path: PathOrStr,
        create_ref: bool | None = None,
    ) -> None:
        """Write a novaSTA flat key-value parameter file.

        Parameters
        ----------
        path : PathOrStr
            Destination path for the flat key-value file.
        create_ref : bool or None, default=None
            If ``None``, uses the flag stored on the object.
        """
        cr = self.create_ref if create_ref is None else bool(create_ref)
        n_align = len(self.df)

        # Build display-name → original novaSTA key from stored key order
        display_to_key = {}
        if self._orig_key_order:
            for k in self._orig_key_order:
                if k not in _NOVA_RUN_LEVEL_KEYS:
                    display_to_key[_nova_key_to_name(k)] = k

        def get_nova_key(display_name):
            return display_to_key.get(display_name, _display_to_nova_key(display_name))

        def col_vals(col):
            if col not in self.df.columns:
                return None
            vals = list(self.df[col])
            return None if all(_is_none_val(v) for v in vals) else vals

        def is_constant(vals):
            return len({_fmt_val(v) for v in vals}) == 1

        def write_param(lines, key, vals):
            if is_constant(vals):
                lines.append(f"{key} {_fmt_val(vals[0])}")
            else:
                lines.append(f"{key} {' '.join(_fmt_val(v) for v in vals)}")

        def ensure_length(vals, n):
            return vals * n if len(vals) == 1 else vals

        lines = [f"createRef {1 if cr else 0}", f"iter {n_align}"]
        if not self.df.empty:
            lines.append(f"startIndex {int(self.df['iteration'].iloc[0])}")

        # Angle coupling
        _angle_display = ["cone angle", "cone sampling", "inplane angle", "inplane sampling"]
        angle_vals = {f: col_vals(f) for f in _angle_display}
        angle_per_iter = n_align > 1 and any(
            v is not None and not is_constant(v) for v in angle_vals.values()
        )
        if angle_per_iter:
            for f in _angle_display:
                angle_vals[f] = ensure_length(angle_vals[f], n_align) if angle_vals[f] else [None] * n_align

        # Filter coupling
        _filter_display = ["low pass", "high pass"]
        filter_vals = {f: col_vals(f) for f in _filter_display}
        filter_per_iter = n_align > 1 and any(
            v is not None and not is_constant(v) for v in filter_vals.values()
        )
        if filter_per_iter:
            for f in _filter_display:
                filter_vals[f] = ensure_length(filter_vals[f], n_align) if filter_vals[f] else [None] * n_align

        _angle_filter_set = set(_angle_display + _filter_display)

        # Determine write order: original key order first, then any new df columns
        write_order = []
        seen = set()
        if self._orig_key_order:
            for k in self._orig_key_order:
                if k in _NOVA_RUN_LEVEL_KEYS:
                    continue
                d = _nova_key_to_name(k)
                if d not in seen:
                    seen.add(d)
                    write_order.append(d)
        for c in self.df.columns:
            if c != "iteration" and c not in seen:
                seen.add(c)
                write_order.append(c)

        # Non-angle/filter columns
        for display_col in write_order:
            if display_col in _angle_filter_set:
                continue
            vals = col_vals(display_col)
            if vals is not None:
                write_param(lines, get_nova_key(display_col), vals)

        # Angle group
        for display_col in _angle_display:
            vals = angle_vals[display_col]
            if vals is not None:
                write_param(lines, get_nova_key(display_col), vals)

        # Filter group
        for display_col in _filter_display:
            vals = filter_vals[display_col]
            if vals is not None:
                write_param(lines, get_nova_key(display_col), vals)

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