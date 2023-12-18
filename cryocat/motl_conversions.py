import numpy as np
import pandas as pd
from cryocat.cryomotl import Motl
from cryocat import cryomotl
from scipy.spatial.transform import Rotation as srot
import re
import warnings


def create_multiref_run(
    motl,
    number_of_classes,
    iteration_number=1,
    number_of_runs=1,
    output_motl_base=None,
):
    n_particles = motl.df.shape[0]
    motl.df.fillna(0.0)

    for i in range(1, number_of_runs + 1):
        # create motl with randomly assigned classes
        r_classes = np.random.randint(1, number_of_classes + 1, size=n_particles)
        motl.df["class"] = r_classes
        if output_motl_base is not None:
            cryomotl.emmotl2stopgap(
                motl,
                output_starfile=output_motl_base + "_mr" + str(i) + "_" + str(iteration_number) + ".star",
                update_coord=False,
            )


def create_denovo_multiref_run(
    motl,
    number_of_classes,
    class_occupancy=None,
    iteration_number=1,
    number_of_runs=1,
    output_motl_base=None,
):
    n_particles = motl.df.shape[0]
    motl.df.fillna(0.0)

    # create motl for reference creation
    if class_occupancy is None:
        class_occupancy = np.ceil(n_particles / 10)

    for i in range(1, number_of_runs + 1):
        ref_df = pd.DataFrame()
        for c in range(1, number_of_classes + 1):
            r_indices = np.random.choice(range(n_particles), class_occupancy, replace=False)
            class_df = motl.df.iloc[r_indices, :].copy()
            class_df["class"] = c
            ref_df = pd.concat([ref_df, class_df], ignore_index=True)

        cryomotl.emmotl2stopgap(
            Motl(ref_df),
            output_starfile=output_motl_base + "_ref_mr" + str(i) + "_" + str(iteration_number) + ".star",
            update_coord=False,
            reset_index=True,
        )

    # create motl with randomly assigned classes
    r_classes = np.random.randint(1, number_of_classes + 1, size=n_particles)
    motl.df["class"] = r_classes
    if output_motl_base is not None:
        cryomotl.emmotl2stopgap(
            motl,
            output_starfile=output_motl_base + "_" + str(iteration_number) + ".star",
            update_coord=False,
        )
