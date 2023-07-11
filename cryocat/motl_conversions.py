import numpy as np
import pandas as pd
import starfile
from cryocat.cryomotl import Motl
from scipy.spatial.transform import Rotation as srot
import re
import warnings


def emmotl2stopgap(
    emmotl_file, output_starfile=None, update_coord=False, reset_index=False
):
    if isinstance(emmotl_file, str):
        motl = Motl(motl_path=emmotl_file)
    else:
        motl = emmotl_file

    if update_coord:
        motl.update_coordinates()

    motl_df = motl.df

    stopgap_df = pd.DataFrame(data=np.zeros((motl_df.shape[0], 16)))
    stopgap_df.columns = [
        "motl_idx",
        "tomo_num",
        "object",
        "subtomo_num",
        "halfset",
        "orig_x",
        "orig_y",
        "orig_z",
        "score",
        "x_shift",
        "y_shift",
        "z_shift",
        "phi",
        "psi",
        "the",
        "class",
    ]

    pairs = {
        "subtomo_id": "subtomo_num",
        "tomo_id": "tomo_num",
        "object_id": "object",
        "x": "orig_x",
        "y": "orig_y",
        "z": "orig_z",
        "score": "score",
        "shift_x": "x_shift",
        "shift_y": "y_shift",
        "shift_z": "z_shift",
        "phi": "phi",
        "psi": "psi",
        "theta": "the",
        "class": "class",
    }

    for em_key, star_key in pairs.items():
        stopgap_df[star_key] = motl_df[em_key]

    stopgap_df["halfset"] = "A"

    if reset_index:
        stopgap_df["motl_idx"] = range(1, stopgap_df.shape[0] + 1)
    else:
        stopgap_df["motl_idx"] = stopgap_df["subtomo_num"]

    if output_starfile is not None:
        starfile.write(stopgap_df, output_starfile, overwrite=True)

    return stopgap_df


def stopgap2emmotl(stopgap_star_file, output_emmotl=None, update_coord=False):
    if isinstance(stopgap_star_file, str):
        stopgap_df = pd.DataFrame(starfile.read(stopgap_star_file))
    else:
        stopgap_df = stopgap_star_file

    motl_df = Motl.create_empty_motl()

    pairs = {
        "subtomo_id": "subtomo_num",
        "tomo_id": "tomo_num",
        "object_id": "object",
        "x": "orig_x",
        "y": "orig_y",
        "z": "orig_z",
        "score": "score",
        "shift_x": "x_shift",
        "shift_y": "y_shift",
        "shift_z": "z_shift",
        "phi": "phi",
        "psi": "psi",
        "theta": "the",
        "class": "class",
    }

    for em_key, star_key in pairs.items():
        motl_df[em_key] = stopgap_df[star_key]

    motl_df["geom4"] = [
        0.0 if hs.lower() == "a" else 1.0 for hs in stopgap_df["halfset"]
    ]

    new_motl = Motl(motl_df)

    if update_coord:
        new_motl.update_coordinates()

    if output_emmotl is not None:
        new_motl.write_to_emfile(output_emmotl)

    return new_motl


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
            emmotl2stopgap(
                motl,
                output_starfile=output_motl_base
                + "_mr"
                + str(i)
                + "_"
                + str(iteration_number)
                + ".star",
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
            r_indices = np.random.choice(
                range(n_particles), class_occupancy, replace=False
            )
            class_df = motl.df.iloc[r_indices, :].copy()
            class_df["class"] = c
            ref_df = pd.concat([ref_df, class_df], ignore_index=True)

        emmotl2stopgap(
            Motl(ref_df),
            output_starfile=output_motl_base
            + "_ref_mr"
            + str(i)
            + "_"
            + str(iteration_number)
            + ".star",
            update_coord=False,
            reset_index=True,
        )

    # create motl with randomly assigned classes
    r_classes = np.random.randint(1, number_of_classes + 1, size=n_particles)
    motl.df["class"] = r_classes
    if output_motl_base is not None:
        emmotl2stopgap(
            motl,
            output_starfile=output_motl_base + "_" + str(iteration_number) + ".star",
            update_coord=False,
        )


def dynamo2emmotl(dynamo_table, output_emmotl=None):
    if isinstance(dynamo_table, str):
        dynt = pd.read_table(dynamo_table, sep=" ", header=None)
    else:
        dynt = dynamo_table

    new_motl = Motl.create_empty_motl()

    new_motl["score"] = dynt.loc[:, 9]

    new_motl["subtomo_id"] = dynt.loc[:, 0]
    new_motl["tomo_id"] = dynt.loc[:, 19]
    new_motl["object_id"] = dynt.loc[:, 20]

    new_motl["x"] = dynt.loc[:, 23]
    new_motl["y"] = dynt.loc[:, 24]
    new_motl["z"] = dynt.loc[:, 25]

    new_motl["shift_x"] = dynt.loc[:, 3]
    new_motl["shift_y"] = dynt.loc[:, 4]
    new_motl["shift_z"] = dynt.loc[:, 5]

    new_motl["phi"] = -dynt.loc[:, 8]
    new_motl["psi"] = -dynt.loc[:, 6]
    new_motl["theta"] = -dynt.loc[:, 7]

    new_motl["class"] = dynt.loc[:, 21]

    new_motl = Motl(new_motl)

    new_motl.update_coordinates()

    if output_emmotl is not None:
        new_motl.write_to_emfile(output_emmotl)

    return new_motl


def relion2emmotl(
    relion_star_file,
    output_emmotl=None,
    relion_version=3.1,
    pixel_size=None,
    binning=None,
    update_coord=False,
):
    if isinstance(relion_star_file, str):
        relion_df = pd.DataFrame(starfile.read(relion_star_file))
    else:
        relion_df = relion_star_file

    # get number of particles
    n_particles = relion_df.shape[0]

    new_motl = Motl.create_empty_motl()

    # version based changes for shifts
    for coord in ("x", "y", "z"):
        relion_column = "rlnOrigin" + coord.upper()
        motl_column = "shift_" + coord

        if (
            relion_column in relion_df.columns
        ):  # Relion 3.0 and lower has shifts in binned pixels/voxels
            new_motl[motl_column] = relion_df[relion_column]
        else:  # Relion 3.1 and higher has shifts in Angstroms
            relion_column = relion_column + "Angst"
            if relion_column in relion_df.columns:
                new_motl[motl_column] = relion_df[relion_column]

                if pixel_size is None:
                    warnings.warn(
                        "The pixel size was not set!!! Default value of 1.0 Angstrom will be used to rescale the shifts!"
                    )
                elif pixel_size != 1.0:
                    new_motl[motl_column] = new_motl[motl_column] / pixel_size

    # version based changes for coordinates
    for coord in ("x", "y", "z"):
        relion_column = "rlnCoordinate" + coord.upper()

        if relion_column in relion_df.columns:
            new_motl[coord] = relion_df[relion_column]

            if relion_version >= 4.0:
                if binning is None:
                    warnings.warn(
                        "No binning specified for Relion4 coordinates - unbinned coordinates will be used!"
                    )
                elif binning != 1.0:
                    new_motl[coord] = new_motl[coord] / binning

    # parsing out tomogram number
    if relion_version >= 4.0:
        tomo_id_name = "rlnTomoName"
        subtomo_id_name = "rlnTomoParticleName"
    else:
        tomo_id_name = "rlnMicrographName"
        subtomo_id_name = "rlnImageName"

    if tomo_id_name in relion_df.columns:
        micrograph_names = relion_df[tomo_id_name].tolist()
        tomo_names = [i.split("/")[-1] for i in micrograph_names]
        tomo_idx = []

        for j in tomo_names:
            tomo_idx.append(float(re.search(r"\d+", j).group()))

        new_motl["tomo_id"] = tomo_idx

    elif subtomo_id_name in relion_df.columns:  # in case there is no migrograph name...
        micrograph_names = relion_df[subtomo_id_name].tolist()
        tomo_names = [i.split("/")[-2] for i in micrograph_names]
        tomo_idx = []

        for j in tomo_names:
            tomo_idx.append(float(re.findall(r"\d+", j)[0]))

        new_motl["tomo_id"] = tomo_idx

    # parsing out subtomo number
    if subtomo_id_name in relion_df.columns:
        image_names = relion_df[subtomo_id_name].tolist()
        subtomo_names = [i.split("/")[-1] for i in image_names]
        subtomo_idx = []

        for j in subtomo_names:
            if relion_version >= 4.0:
                subtomo_idx.append(float(j))
            else:
                subtomo_idx.append(float(re.findall(r"\d+", j)[-3]))

        # Check if the subtomo_idx are unique and if not store them at geom4 and renumber particles
        if len(np.unique(subtomo_idx)) != len(subtomo_idx):
            new_motl["geom4"] = subtomo_idx
            new_motl["subtomo_id"] = np.arange(1, n_particles + 1, 1)
        else:
            new_motl["subtomo_id"] = subtomo_idx

    # getting class number
    if "rlnClassNumber" in relion_df.columns:
        new_motl["class"] = relion_df["rlnClassNumber"]

    # getting the max value contribution per distribution - not really same as CCC but has similar indications
    if "rlnMaxValueProbDistribution" in relion_df.columns:
        new_motl["score"] = relion_df["rlnMaxValueProbDistribution"]

    # getting the angles
    angles = np.zeros((n_particles, 3))
    if "rlnAngleRot" in relion_df.columns:
        angles[:, 0] = relion_df["rlnAngleRot"].to_numpy()

    if "rlnAngleTilt" in relion_df.columns:
        angles[:, 1] = relion_df["rlnAngleTilt"].to_numpy()

    if "rlnAnglePsi" in relion_df.columns:
        angles[:, 2] = relion_df["rlnAnglePsi"].to_numpy()

    # convert from ZYZ to zxz
    rot_ZYZ = srot.from_euler("ZYZ", angles, degrees=True)
    rot_zxz = rot_ZYZ.as_euler("zxz", degrees=True)

    # save so the rotation describes reference rotation
    new_motl["phi"] = -rot_zxz[:, 2]
    new_motl["psi"] = -rot_zxz[:, 0]
    new_motl["theta"] = -rot_zxz[:, 1]

    # conversions of shifts - emmotl stores shifts for the reference, relion for the subtomo
    new_motl["shift_x"] = -new_motl["shift_x"]
    new_motl["shift_y"] = -new_motl["shift_y"]
    new_motl["shift_z"] = -new_motl["shift_z"]

    new_motl = Motl(new_motl)

    if update_coord:
        new_motl.update_coordinates()
        warnings.warn(
            "The shifts were added to the extraction coordinates - subtomograms have to be reextracted!"
        )

    if output_emmotl is not None:
        new_motl.write_to_emfile(output_emmotl)

    return new_motl

def emmotl2relion(
    emmotl_file,
    output_starfile=None,
    relion_version=4.0,
    pixel_size=None,
    binning=1.0,
    tomo_base_prefix = "TS_",
    tomo_digits = 3
):

    if isinstance(emmotl_file, str):
        motl = Motl(motl_path=emmotl_file)
    else:
        motl = emmotl_file

    relion_df = pd.DataFrame(data=np.zeros((motl.df.shape[0], 9)))
    relion_df.columns = [
        "rlnCoordinateX",
        "rlnCoordinateY",
        "rlnCoordinateZ",
        "rlnAngleRot",
        "rlnAngleTilt",
        "rlnAnglePsi",
        "rlnTomoName",
        "rlnTomoParticleName",
        "rlnObjectName"
    ]


    for coord in ("X", "Y", "Z"):
        relion_df["rlnCoordinate" + coord] = motl.df[coord.lower()] + motl.df["shift_" + coord.lower()]
    
    rotations = srot.from_euler("ZXZ",  motl.get_angles(), degrees=True)
    angles = rotations.as_euler("ZYZ", degrees=True)
    
    # save so the rotation describes reference rotation
    relion_df["rlnAngleRot"] = -angles[:, 0]
    relion_df["rlnAngleTilt"] = angles[:, 1]
    relion_df["rlnAnglePsi"] = -angles[:, 2]

    relion_df["rlnObjectName"] = motl.df["object_id"]

    relion_df["rlnTomoName"] = tomo_base_prefix + motl.df["tomo_id"].astype(int).astype(str).str.zfill(tomo_digits)
    relion_df["rlnTomoParticleName"] = relion_df["rlnTomoName"].astype(str) + "/" + motl.df["subtomo_id"].astype(int).astype(str)

    if binning != 1.0 and relion_version >= 4.0 :
        for coord in ("X", "Y", "Z"):
            relion_df["rlnCoordinate" + coord] = relion_df["rlnCoordinate" + coord] * binning
    
    if binning != 1.0 and relion_version < 4.0 :
        for coord in ("X", "Y", "Z"):
            relion_df["rlnCoordinate" + coord] = relion_df["rlnCoordinate" + coord] / binning

    if output_starfile is not None:
        starfile.write(relion_df, output_starfile, overwrite=True)

    return relion_df