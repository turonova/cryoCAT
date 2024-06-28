import numpy as np
import pandas as pd
from cryocat import cryomotl
from cryocat import geom
from cryocat import mathutils
from cryocat import visplot
from cryocat import ioutils


def get_stable_particles(motl_base_name, start_it, end_it, motl_type="emmotl"):
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
    motl_type : str (stopgap|emmotl|relion), default="stopgap"
        Type of the input motl. Defaults to "stopgap".

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

    motl_ext = get_motl_extension(motl_type)

    dfs = []
    for i in np.arange(start_it, end_it + 1):
        m = cryomotl.Motl.load(motl_base_name + str(i) + motl_ext, motl_type=motl_type)
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
    motl_base_names,
    start_it,
    end_it,
    motl_type="stopgap",
    write_out_stats=False,
    plot_values=True,
    filter_rows=None,
    filter_columns="subtomo_id",
    labels=None,
    graph_title="Alignment stability",
    graph_output_file=None,
):
    """Evaluate alignment stability for specified motls and iterations.

    Parameters
    ----------
    motl_base_names  : str or list
        List of MOTL base names or a single motl base name to perform the evaluation on. Base name means without the
        iteration number and extension. For example for name motl_shift_3.em the base name is motl_shift\_.
    start_it  : int
        Starting iteration number.
    end_it  : int
        Ending iteration number.
    motl_type  : str (stopgap|emmotl|relion), default="stopgap"
        Type of the input motl. Defaults to "stopgap".
    write_out_stats  : bool, default=False
        Whether to write out stats. If True, the stats will be written to the motl_base_name + _as_motlID.csv where the
        motlID is given by its position in the motl_base_names list. For example, for motl_shift_3.em the final will
        be motl_shift_as_1.em if the motl_shift\_ is the first motl in the motl_base_names. Defaults to False.
    plot_values  : bool, default=True
        Whether to plot values. Defaults to True.
    filter_rows  : array-like or list of array-like, optional
        Rows to filter. Only rows that are within the filter_rows will be kept. Defaults to None which means no filtering.
    filter_columns  : str or list, default="subtomo_id"
        Column names based on which the filtering is perfomed. If fitler_rows is None, no filtering will be done and
        this parameter will not be used. Defaults to "subtomo_id".
    labels  : str or list, optional
        Labels for the plot. Should have the same length as the motl_base_names. In case of None, the labels will
        be automatically set as motl_base_names (in case those names contain paths, the paths will be removed).
        Used only if plot_values is True. Defaults to None.
    graph_title  : str, default="Alignment stability"
        Title of the graph. Used only if plot_values is True. Defaults to "Alignment stability".
    graph_output_file  : str, optional
        Output file for the graph. Used only if plot_values is True. If None no file will be written out. Defaults to None.

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
    ...     filter_rows=filter_rows, filter_column="geom3",
    ...     motl_type="stopgap", plot_values=True, write_out_stats=True
    ... )

    >>> # Multiple motls, motls motl1_1.star to motl1_17.star and motl3_1.star to motl3_17.star will be loaded for
    >>> # evaluation. Statistics will be written into /path/to/the/motl1_as_1.csv and /path/to/the/motl3_as_2.csv files.
    >>> # Filtering will be done based on column geom3 for motl1 and based on subtomo_id for motl3.
    >>> # Only particles with values in filter_rows will be evaluated.
    >>> motl_base_names = ["/path/to/the/motl1_", "/path/to/the/motl3_"]
    >>> filter_rows = [values_to_keep_for_motl1, values_to_keep_for_motl3]
    >>> filter_column = ["geom3", "subtomo_id"]
    >>> stats_df = evaluate_alignment(
    ...     motl_base_names, 1, 17,
    ...     filter_rows=filter_rows, filter_column=filter_column,
    ...     motl_type="stopgap", plot_values=True, write_out_stats=True
    ... )

    >>> # Multiple motls, motls motl1_1.star to motl1_17.star and motl3_1.star to motl3_17.star will be loaded for
    >>> # evaluation. Statistics will be written into /path/to/the/motl1_as_1.csv and /path/to/the/motl3_as_2.csv files.
    >>> # Filtering will be done based on column geom3 for motl1 and no filtering will be done for motl3.
    >>> # Only particles with values in filter_rows will be evaluated.
    >>> motl_base_names = ["/path/to/the/motl1_", "/path/to/the/motl3_"]
    >>> filter_rows = [values_to_keep_for_motl1, None]
    >>> filter_column = ["geom3", None]
    >>> stats_df = evaluate_alignment(
    ...     motl_base_names, 1, 17,
    ...     filter_rows=filter_rows, filter_column=filter_column,
    ...     motl_type="stopgap", plot_values=True, write_out_stats=True
    ... )
    """

    if not isinstance(motl_base_names, list):
        motl_base_names = [motl_base_names]
    # ensure correct input formats in case there is only one filter_rows and filter_column specified
    if not isinstance(filter_columns, list):
        filter_columns = [filter_columns]
        if len(motl_base_names) > 1 and len(filter_columns) == 1:
            filter_columns = filter_columns * len(motl_base_names)

    if filter_rows is None:
        filter_rows = [None] * len(motl_base_names)
    # filter_rows = np.full((1, len(motl_base_names)), None)
    elif not isinstance(filter_rows, list):
        filter_rows = [filter_rows]
        if len(filter_rows) != len(motl_base_names) and len(filter_rows) == 1:
            filter_rows = filter_rows * len(motl_base_names)

    stats_dfs = []
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
                filter_column=filter_columns[i],
                output_file=stats_file_name,
            )
        )

    if plot_values:
        if labels is None:
            labels = [ioutils.get_filename_from_path(m)[0:-1] for m in motl_base_names]
        visplot.plot_alignment_stability(
            stats_dfs, labels=labels, graph_title=graph_title, output_file=graph_output_file
        )

    return stats_dfs


def get_motl_extension(motl_type):
    """Return the file extension for a given motl type.

    Parameters
    ----------
    motl_type : str (emmotl|relion|stopgap)
        The type of motl file.

    Returns
    -------
    str
        The file extension corresponding to the motl type.

    Raises
    ------
    ValueError
        If the motl type is not supported.
    """

    if motl_type == "stopgap" or motl_type == "relion":
        motl_ext = ".star"
    elif motl_type == "emmotl":
        motl_ext = ".em"
    else:
        raise ValueError(f"The motl type {motl_type} is not cyrrently supported.")

    return motl_ext


def compute_alignment_statistics(
    motl_base_name,
    start_it,
    end_it,
    motl_type="stopgap",
    filter_rows=None,
    filter_column="subtomo_id",
    output_file=None,
):
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
    motl_type : str (stopgap|emmotl|relion), default="stopgap"
        Type of the input motl. Defaults to "stopgap".
    filter_rows : array-like, optional
        Rows to filter. Only rows that are within the filter_rows will be kept. Defaults to None which means no filtering.
    filter_columns : str, default="subtomo_id"
        Column names based on which the filtering is perfomed. If fitler_rows is None, no filtering will be done and
        this parameter will not be used. Defaults to "subtomo_id".
    output_file : str, optional
        Output file for the statistics. If None no file will be written out. Defaults to None.

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
    ...     motl_type="stopgap", output_file="/path/to/the/motl_alignment_stats.csv"
    ... )

    >>> # Motls motl_1.star to motl_17.star will be loaded for evaluation, no file will be written out.
    >>> # Filtering will be done based on column geom3 and only particles with values in filter_rows will be evaluated.
    >>> stats_df = compute_alignment_statistics(
    ...     "/path/to/the/motl_", 1, 17,
    ...     filter_rows=values_to_keep_for_motl, filter_column="geom3",
    ...     motl_type="stopgap"
    ... )
    """

    motl_ext = get_motl_extension(motl_type)

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
    for i in np.arange(start_it, end_it + 1):
        m = cryomotl.Motl.load(motl_base_name + str(i) + motl_ext, motl_type=motl_type)
        if filter_rows is not None:
            m.df = m.df[m.df[filter_column].isin(filter_rows)]
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

    if output_file is not None:
        stats_df.to_csv(output_file, index=False)

    return stats_df


def write_out_motl(input_motl, output_file_base, output_motl_type):
    """Writes out a given motl file to a specified output format.

    Parameters
    ----------
    input_motl : motl
        Input motl file to be written out.
    output_file_base : str
        Base name for the output file.
    output_motl_type : str (emfile|relion|stopgap)
        Type of the output motl file.

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
    elif output_motl_type == "emfile":
        final_motl = cryomotl.EmMotl(input_motl.df)
        final_motl.write_out(output_file_base + ".em")
    else:
        raise ValueError(f"The output motl type {output_motl_type} is not supported.")


def create_multiref_run(
    input_motl,
    number_of_classes,
    output_motl_base,
    input_motl_type="emmotl",
    iteration_number=1,
    number_of_runs=1,
    output_motl_type="stopgap",
):
    """Creates motls for multiple runs of a multi-reference alignment. In essence, it will randomly assign specified number
    of classes to each motl that will be created. New motls will be written out into files
    output_motl_base_mr#runID_iterationNumber either in stopgap, emmotl or relion format.

    Parameters
    ----------
    input_motl : str, pandas dataframe or Motl
        Input motl (specified either as a path, dataframe or Motl object).
    number_of_classes : int
        Number of classes to assign randomly.
    output_motl_base : str
        Base path for the output motl files. The final name will be created as output_motl_base_mr#runID_iterationNumber
        where runID is from 1 to number_of_runs and iterationNumber is iteration_number. The extension will be determined
        based on the output_motl_type.
    input_motl_type : str (emmotl|stopgap|relion), default="emmotl"
        Type of the input motl file. Defaults to "emmotl".
    iteration_number : int, default=1
        Iteration number to be used in the output name creation. Defaults to 1.
    number_of_runs : int, default=1
        Number of motls to create. Defaults to 1.
    output_motl_type : str (stopgap|emmotl|relion), default="stopgap"
        Type of the output motl file. Defaults to "stopgap".

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

        output_file = output_motl_base + "_mr" + str(i) + "_" + str(iteration_number)
        write_out_motl(motl, output_file, output_motl_type=output_motl_type)


def create_denovo_multiref_run(
    input_motl,
    number_of_classes,
    output_motl_base,
    input_motl_type="emmotl",
    class_occupancy=None,
    iteration_number=1,
    number_of_runs=1,
    output_motl_type="stopgap",
):
    """Creates number_of_runs motls for reference averaging and one motl for alignment. The motls for reference averaging
    are created by random selection of N particles for each class from the input_motl, where N equals to class_occupancy.
    The particles within the classes of each motl can overlap, i.e. each class will have a unique set of particles, but
    some particles can be assigned in mutliple classes. The alignment motl is just input motl where the class was
    randomly assign to be from 1 to number_of_classes. The idea behind this is to run multi-reference alignment
    where different runs will have different starting references while due to simmulated annealing only one motl
    for alignment is needed afterwards.

    Parameters
    ----------
    input_motl : str, pandas dataframe or Motl
        Input motl (specified either as a path, dataframe or Motl object).
    number_of_classes : int
        Number of classes to create references for and to assign randomly to the alignment motl.
    output_motl_base : str
        Base path for the output motl files. The final name will be created as output_motl_base_ref_mr#runID_iterationNumber
        where runID is from 1 to number_of_runs and iterationNumber is iteration_number. The alignment motl will be named
        output_motl_base_iterationNumber. In both cases, the extension will be determined based on the output_motl_type.
    input_motl_type : str (emmotl|stopgap|relion), default="emmotl"
        Type of the input motl file. Defaults to "emmotl".
    class_occupancy : int, optional
        Number of particles per class for the reference averaging motls. If None, the number is determined as 1/10
        of total number of particles in the input motl. Defaults to None.
    iteration_number : int, default=1
        Iteration number to be used in the output name creation. Defaults to 1.
    number_of_runs : int, default=1
        Number of motls to create. Defaults to 1.
    output_motl_type : str (stopgap|emmotl|relion), default="stopgap"
        Type of the output motl file. Defaults to "stopgap".

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
        output_file = output_motl_base + "_ref_mr" + str(i) + "_" + str(iteration_number)
        write_out_motl(new_motl, output_file_base=output_file, output_motl_type=output_motl_type)

    # create motl with randomly assigned classes
    motl.assign_random_classes(number_of_classes)
    write_out_motl(
        motl, output_file_base=output_motl_base + "_" + str(iteration_number), output_motl_type=output_motl_type
    )


def evaluate_multirun_stability(input_motls, input_motl_type="stopgap"):
    """Evaluate how many particles ended up within the same class among all the classification runs. It is meant to be
    used for multiruns with existing references (i.e. not de novo ones) where all runs uses the same references in the
    same order.

    Parameters
    ----------
    input_motls: list
        List of input motl files. At least two are required.
    motl_type : str (stopgap|emmotl|relion), default="stopgap"
        Type of the input motl. Defaults to "stopgap".

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


def get_subtomos_class_stability(motl_base_name, start_it, end_it, motl_type="stopgap"):
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
    motl_type : str (stopgap|emmotl|relion), default="stopgap"
        Type of the input motl. Defaults to "stopgap".

    Returns
    -------
    different_sids : dict
        A dictionary containing the number of different subtomogram IDs for each class over iterations.

    Notes
    -----
    Loading of many motls can take some time. If you also want to compute occupancy of classes it is recommended to
    use :meth:`cryocat.sta.evaluate_classification` which gives both occupancy and stability and reads in all the motls
    only once.
    """

    motl_ext = get_motl_extension(motl_type)

    dfs = []
    for i in np.arange(start_it, end_it + 1):
        m = cryomotl.Motl.load(motl_base_name + str(i) + motl_ext, motl_type=motl_type)
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
    motl_base_name,
    start_it,
    end_it,
    motl_type="stopgap",
    output_file_stats=None,
    plot_results=False,
    output_file_graphs=None,
):
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
    motl_type : str (stopgap|emmotl|relion), default="stopgap"
        Type of the input motl. Defaults to "stopgap".
    output_file_stats : str, optional
        Name of the file into which the results will be written out. If None, no results will be written out. Defaults
        to None.
    plot_results: bool, default=False
        Whether to plot the results. Defaults to False.
    output_file_graphs: str, optional
        Name of the file into which the plotted graphs will be written out. If None, the graphs will not be written out.
        If plot_results is False, this parameter is unused. Defaults to None.

    Returns
    -------
    occupancy : dict
        A dictionary containing the occupancy of each class over the iterations.
    changing_subtomos : dict
        A dictionary containing the number of different subtomogram IDs for each class over iterations.
    """

    motl_ext = get_motl_extension(motl_type)

    dfs = []
    for i in np.arange(start_it, end_it + 1):
        m = cryomotl.Motl.load(motl_base_name + str(i) + motl_ext, motl_type=motl_type)
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
            occupancy, changing_subtomos, graph_title="Classification progress", output_file=output_file_graphs
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
        merged.to_csv(output_file_stats, index=False)

    return occupancy, changing_subtomos


def get_class_occupancy(motl_base_name, start_it, end_it, motl_type="stopgap"):
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
    motl_type : str (stopgap|emmotl|relion), default="stopgap"
        Type of the input motl. Defaults to "stopgap".

    Returns
    -------
    occupancy : dict
        A dictionary containing the occupancy of each class over the iterations.

    Notes
    -----
    Loading of many motls can take some time. If you also want to compute stability of classes it is recommended to
    use :meth:`cryocat.sta.evaluate_classification` which gives both occupancy and stability and reads in all the motls
    only once.
    """

    motl_ext = get_motl_extension(motl_type)

    dfs = []
    for i in np.arange(start_it, end_it + 1):
        m = cryomotl.Motl.load(motl_base_name + str(i) + motl_ext, motl_type=motl_type)
        dfs.append(m.df)

    # Create a dictionary to store the occupancy of each class per dataframe
    occupancy = {}
    for i, df in enumerate(dfs):
        for c in df["class"].unique():
            if c not in occupancy:
                occupancy[c] = [0] * len(dfs)
            occupancy[c][i] = len(df[df["class"] == c])

    return occupancy
