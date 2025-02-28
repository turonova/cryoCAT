import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from cryocat import cryomap
from cryocat import geom
from cryocat import ioutils
from cryocat import cryomotl
from cryocat import tmana
from cryocat import mathutils
from cryocat import wedgeutils
from scipy.spatial.transform import Rotation as srot
import re
from pathlib import Path
from cryocat import cryomask
import os
from skimage import measure
from skimage import morphology
import matplotlib.gridspec as gridspec
from scipy.interpolate import RegularGridInterpolator
from skimage.transform import rotate as skimage_rotate

import warnings

warnings.filterwarnings("ignore")


def rotate_image(image, alpha, fill_mode="constant", fill_value=0.0):
    return skimage_rotate(image, alpha, resize=False, mode=fill_mode, cval=fill_value)


def _ctf(defocus, pshift, famp, cs, evk, f):
    """Original 'docstring':
    %% sg_ctf
    % Calculate a CTF curve.
    %
    % WW 07-2018
    """

    defocus = defocus * 1.0e4
    cs = cs * 1.0e7
    pshift = pshift * np.pi / 180

    h = 6.62606957e-34
    c = 299792458
    erest = 511000
    v = evk * 1000
    e = 1.602e-19

    lam = (c * h) / np.sqrt(((2 * erest * v) + (v**2)) * (e**2)) * (10**10)
    w = (1 - (famp**2)) ** 0.5

    v = (np.pi * lam * (f**2)) * (defocus - 0.5 * (lam**2) * (f**2) * cs)
    v += pshift
    ctf = (w * np.sin(v)) + (famp * np.cos(v))

    return ctf


def generate_ctf(wl, slice_idx, slice_weight, binning):
    """Generate ctf filter."""

    tmpl_size = slice_weight.shape[0]
    full_size = int(max(wl["tomo_x"].values[0], wl["tomo_y"].values[0], wl["tomo_z"].values[0]))
    pixelsize = wl["pixelsize"].values[0] * binning

    freqs_full = mathutils.compute_frequency_array((full_size,), pixelsize)
    freqs_crop = mathutils.compute_frequency_array((tmpl_size,), pixelsize)[tmpl_size // 2 :]

    f_idx = np.zeros(full_size, dtype="bool")
    f_idx[:tmpl_size] = 1
    f_idx = np.roll(f_idx, -tmpl_size // 2)
    f_idx = np.nonzero(f_idx)[0]

    defocus = np.array(wl["defocus"])
    pshift = wl.get("pshift", np.zeros_like(defocus))

    # microscope parameters
    famp = wl["amp_contrast"].values[0]
    cs = wl["cs"].values[0]
    evk = wl["voltage"].values[0]

    full_ctf = np.abs(_ctf(defocus[:, None], pshift[:, None], famp, cs, evk, freqs_full))
    ft_ctf = np.fft.fft(full_ctf, axis=1)
    ft_ctf = ft_ctf[:, f_idx] * tmpl_size / full_size
    crop_ctf = np.real(np.fft.ifft(ft_ctf, axis=1))
    crop_ctf = crop_ctf[:, tmpl_size // 2 :]

    ctf_filt = np.zeros_like(slice_weight)
    x = np.fft.ifftshift(mathutils.compute_frequency_array(slice_weight.shape, pixelsize))
    for ictf, sidx in zip(crop_ctf, slice_idx):
        ip = RegularGridInterpolator(
            (freqs_crop,),  # Grid points
            ictf,  # Values on the grid
            method="linear",  # Interpolation method ('linear', 'nearest')
            bounds_error=False,  # Do not raise error for out-of-bound points
            fill_value=0,  # Fill with 0 for out-of-bound points
        )
        ctf_filt[sidx] += ip(x[sidx])

    # ? np.nan_to_num(ctf_filt)
    ctf_filt *= slice_weight

    return ctf_filt


def generate_exposure(wedgelist, slice_idx, slice_weight, binning):
    """Generate exposure filter."""

    expo = wedgelist["exposure"].values
    a, b, c = (0.245, -1.665, 2.81)
    pixelsize = wedgelist["pixelsize"].values[0] * binning

    freq_array = np.fft.ifftshift(mathutils.compute_frequency_array(slice_weight.shape, pixelsize))

    exp_filt = np.zeros_like(slice_weight)
    for expi, idx in zip(expo, slice_idx):
        freqs = freq_array[idx]
        exp_filt[idx] += np.exp(-expi / (2 * ((a * freqs**b) + c)))

    exp_filt *= slice_weight

    return exp_filt


def generate_wedgemask_slices_template(wedgelist, template_filter):
    """Generate wedgemask slices for template"""

    template_size = np.array(template_filter.shape)

    # original codes uses matlab axes 1 and 3 which correspond to x and z
    # according to the original gsg_mrcread; so with mrcfile this is 2 and 0
    mx = np.max(template_size[[2, 0]])
    img = np.zeros((mx, mx))
    img[:, mx // 2] = 1.0

    bpf_idx = template_filter > 0

    # tmpl filter stuff
    active_slices_idx = []
    wedge_slices_weights = np.zeros_like(template_filter)
    weight = np.zeros_like(template_filter)

    for alpha in wedgelist["tilt_angle"]:

        r_img = rotate_image(img, alpha)

        # template filter
        crop_r_img = r_img > np.exp(-2)

        slice_vol = np.fft.ifftshift(np.transpose(np.tile(crop_r_img, (mx, 1, 1)), (2, 0, 1)))

        slice_idx = slice_vol & bpf_idx
        weight += slice_idx
        active_slices_idx.append(np.nonzero(slice_idx))

    # invert values for filter
    w_idx = np.nonzero(weight)
    wedge_slices_weights[w_idx] = 1.0 / weight[w_idx]

    wedge_slices = np.zeros_like(weight)
    wedge_slices[w_idx] = 1.0

    return active_slices_idx, wedge_slices_weights, wedge_slices


def generate_wedgemask_slices_tile(wedgelist, tile_filter):
    """Generate wedgemask slices for tiles/subtomograms"""

    tile_size = np.array(tile_filter.shape)

    # original codes uses matlab axes 1 and 3 which correspond to x and z
    # according to the original gsg_mrcread; so with mrcfile this is 2 and 0
    mx = np.max(tile_size[[2, 0]])
    img = np.zeros((mx, mx))
    img[:, mx // 2] = 1.0

    # tile filter stuff
    tile_bin_slice = np.zeros(tile_size[[2, 0]], dtype="float32")

    for alpha in wedgelist["tilt_angle"]:

        r_img = rotate_image(img, alpha)
        # tile filter
        r_img = np.fft.fftshift(np.fft.fft2(r_img))
        new_img = np.real(np.fft.ifft2(np.fft.ifftshift(r_img)))
        new_img /= np.max(new_img)
        tile_bin_slice += new_img > np.exp(-2)

    # generate tile binary wedge filter
    tile_bin_slice = (tile_bin_slice > 0).astype("float32")
    wedge_slices = np.fft.ifftshift(np.transpose(np.tile(tile_bin_slice, (tile_size[1], 1, 1)), (2, 0, 1)))

    return wedge_slices


def generate_wedge_masks(
    template_size,
    tile_size,
    wedgelist,
    tomo_number,
    binning=1,
    low_pass_filter=None,
    high_pass_filter=None,
    ctf_weighting=False,
    exposure_weighting=False,
    output_template=None,
    output_tile=None,
):

    filter_template = np.ones(cryomask.get_correct_format(template_size))
    filter_tile = np.ones(cryomask.get_correct_format(tile_size))

    # get relevant subset of the wedgelist
    wedgelist = wedgeutils.load_wedge_list_sg(wedgelist)
    wedgelist = wedgelist.loc[wedgelist["tomo_num"] == tomo_number]

    if low_pass_filter:
        filter_template = cryomap.lowpass(filter_template, fourier_pixels=low_pass_filter)
        filter_tile = cryomap.lowpass(filter_tile, fourier_pixels=low_pass_filter)

    if high_pass_filter:
        filter_template = cryomap.highpass(filter_template, fourier_pixels=low_pass_filter)
        filter_tile = cryomap.highpass(filter_tile, fourier_pixels=low_pass_filter)

    active_slices_idx, wedge_slices_weights, wedge_slices_template = generate_wedgemask_slices_template(
        wedgelist, filter_template
    )
    wedge_slices_template_tile = generate_wedgemask_slices_tile(wedgelist, filter_tile)

    # init filters
    filter_template = wedge_slices_template * filter_template
    filter_tile = wedge_slices_template_tile * filter_tile

    # update template filter
    if exposure_weighting:
        filter_template *= generate_exposure(wedgelist, active_slices_idx, wedge_slices_weights, binning)

    if ctf_weighting:
        filter_template *= generate_ctf(wedgelist, active_slices_idx, wedge_slices_weights, binning)

    if output_template:
        cryomap.write(filter_template, output_template, transpose=False, data_type=np.single)

    if output_tile:
        cryomap.write(filter_tile, output_tile, transpose=False, data_type=np.single)

    return filter_template.transpose(2, 1, 0), filter_tile.transpose(2, 1, 0)


def create_structure_path(folder_path, structure_name):
    structure_folder = folder_path + structure_name + "/"
    return structure_folder


def create_em_path(folder_path, structure_name, em_filename):
    structure_folder_path = create_structure_path(folder_path, structure_name)
    em_path = structure_folder_path + em_filename + ".em"
    return em_path


def create_subtomo_name(structure_name, motl_name, tomo_id, boxsize):
    subtomo_name = "subtomo_" + structure_name + "_m" + motl_name + "_t" + tomo_id + "_s" + str(boxsize) + ".em"
    return subtomo_name


def create_tomo_name(
    folder_path,
    tomo,
):
    tomo_name = folder_path + tomo + ".mrc"
    return tomo_name


def create_wedge_names(wedge_path, tomo_number, boxsize, binning, filter=None):
    if filter is None:
        filter = boxsize // 2

    file_ending = str(boxsize) + "_t" + str(tomo_number) + "_b" + str(binning) + "_f" + str(filter) + ".em"
    tomo_wedge = wedge_path + "tile_filt_" + file_ending
    tmpl_wedge = wedge_path + "tmpl_filt_" + file_ending

    return tomo_wedge, tmpl_wedge


def create_output_base_name(tmpl_index):
    output_base = "id_" + str(tmpl_index)
    return output_base


def create_output_folder_name(tmpl_index):
    return create_output_base_name(tmpl_index) + "_results"


def create_output_folder_path(folder_path, structure_name, folder_spec):
    if isinstance(folder_spec, int):
        output_path = create_structure_path(folder_path, structure_name) + create_output_folder_name(folder_spec) + "/"
    else:
        output_path = create_structure_path(folder_path, structure_name) + folder_spec + "/"

    return output_path


def get_indices(template_list, conditions, sort_by=None):
    temp_df = pd.read_csv(template_list, index_col=0)

    for key, value in conditions.items():
        temp_df = temp_df.loc[temp_df[key] == value, :]
        # display(temp_df)

    if sort_by is not None:
        temp_df = temp_df.sort_values(by=sort_by, ascending=True)

    return temp_df.index


def get_sharp_mask_stats(input_mask):
    mask_bb = cryomask.get_mass_dimensions(input_mask)
    n_voxels = np.count_nonzero(input_mask)

    return n_voxels, mask_bb


def get_soft_mask_stats(input_mask):
    mask_th = np.where(input_mask > 0.5, 1.0, 0.0)
    mask_bb = cryomask.get_mass_dimensions(mask_th)
    n_voxels = np.count_nonzero(mask_th)

    return n_voxels, mask_bb


def cut_the_best_subtomo(tomogram, motl_path, subtomo_shape, output_file):
    tomo = cryomap.read(tomogram)
    m = cryomotl.Motl.load(motl_path)
    m.update_coordinates()

    max_idx = m.df["score"].idxmax()

    coord = m.df.loc[m.df.index[max_idx], ["x", "y", "z"]].to_numpy() - 1
    shifts = -m.df.loc[m.df.index[max_idx], ["shift_x", "shift_y", "shift_z"]].to_numpy()
    angles = m.df.loc[m.df.index[max_idx], ["phi", "theta", "psi"]].to_numpy()

    subvolume = cryomap.extract_subvolume(tomo, coord, subtomo_shape)
    subvolume_sh = cryomap.shift2(subvolume, shifts)
    # subvolume_rot = cryomap.rotate(subvolume_sh,rotation_angles=angles)

    if output_file is not None:
        cryomap.write(subvolume_sh, output_file, data_type=np.single)

    return subvolume_sh, angles


# get subtomograms out
def create_subtomograms_for_tm(template_list, parent_folder_path):
    temp_df = pd.read_csv(template_list, index_col=0)
    unique_entries = temp_df.groupby(["Structure", "Motl", "Tomogram", "Boxsize"]).groups
    entry_indices = list(unique_entries.values())

    for i, entry in enumerate(unique_entries):
        if np.all(temp_df.loc[entry_indices[i], "Tomo created"]):
            continue
        else:
            motl = create_em_path(parent_folder_path, entry[0], entry[1])
            boxsize = entry[3]
            not_created = temp_df.loc[temp_df["Tomo created"] == False, "Tomo created"].index
            create_idx = np.intersect1d(not_created, entry_indices[i])
            subtomo_name = create_subtomo_name(entry[0], entry[1], entry[2], boxsize)
            _, subtomo_rotation = cut_the_best_subtomo(
                create_tomo_name(parent_folder_path, entry[2]),
                motl,
                (boxsize, boxsize, boxsize),
                create_structure_path(parent_folder_path, entry[0]) + subtomo_name,
            )
            temp_df.loc[create_idx, ["Phi", "Theta", "Psi"]] = np.tile(subtomo_rotation, (create_idx.shape[0], 1))
            temp_df.loc[create_idx, "Tomo created"] = True
            temp_df.loc[create_idx, "Tomo map"] = subtomo_name[0:-3]

    temp_df.to_csv(template_list)

    return temp_df

def get_mask_stats(template_list, indices, parent_folder_path):
    temp_df = pd.read_csv(template_list, index_col=0)

    for i in indices:
        structure_name = temp_df.at[i, "Structure"]
        soft_mask = cryomap.read(create_em_path(parent_folder_path, structure_name, temp_df.at[i, "Mask"]))
        sharp_mask = cryomap.read(create_em_path(parent_folder_path, structure_name, temp_df.at[i, "Tight mask"]))

        solidity = cryomask.compute_solidity(sharp_mask)

        voxels, bbox = get_sharp_mask_stats(sharp_mask)
        voxels_soft, _ = get_soft_mask_stats(soft_mask)

        temp_df.at[i, "Voxels"] = voxels_soft
        temp_df.at[i, "Voxels TM"] = voxels
        temp_df.at[i, "Dim x"] = bbox[0]
        temp_df.at[i, "Dim y"] = bbox[1]
        temp_df.at[i, "Dim z"] = bbox[2]
        temp_df.at[i, "Solidity"] = solidity

        temp_df.to_csv(template_list)


def compute_sharp_mask_overlap(template_list, indices, angle_list_path, parent_folder_path, angles_order="zxz"):
    temp_df = pd.read_csv(template_list, index_col=0)

    for i in indices:
        if not temp_df.at[i, "Done"]:
            continue

        structure_name = temp_df.at[i, "Structure"]
        mask_name = create_em_path(parent_folder_path, structure_name, temp_df.at[i, "Tight mask"])
        mask = cryomap.read(mask_name)
        angle_list = angle_list_path + temp_df.at[i, "Angles"]
        angles = ioutils.rot_angles_load(angle_list, angles_order)
        rotations = srot.from_euler("zxz", angles, degrees=True)

        voxel_count = []

        for j in rotations:
            mask_rot = cryomap.rotate(mask, rotation=j, transpose_rotation=True)
            mask_rot = np.where(mask_rot > 0.1, 1.0, 0.0)
            voxel_count.append(np.count_nonzero(mask_rot * mask))

        output_base = create_output_base_name(i)
        output_folder = create_output_folder_path(parent_folder_path, structure_name, temp_df.at[i, "Output folder"])

        csv_name = output_folder + output_base + ".csv"
        info_df = pd.read_csv(csv_name, index_col=0)
        info_df["Tight mask overlap"] = np.asarray(voxel_count)
        info_df.to_csv(csv_name)


def check_existing_tight_mask_values(template_list, indices, parent_folder_path, angle_list_path, angles_order="zxz"):
    temp_df = pd.read_csv(template_list, index_col=0)

    for i in indices:
        if not temp_df.at[i, "Done"]:
            continue

        structure_name = temp_df.at[i, "Structure"]
        output_base = create_output_base_name(i)
        output_folder = create_output_folder_path(parent_folder_path, structure_name, temp_df.at[i, "Output folder"])

        rot_info = pd.read_csv(output_folder + output_base + ".csv", index_col=0)

        data_found = False

        if "Tight mask overlap" not in rot_info.columns:
            tm_spec = {"Done": True, "Tight mask": temp_df.at[i, "Tight mask"], "Degrees": temp_df.at[i, "Degrees"]}
            done_idx = get_indices(template_list, tm_spec)

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

        if not data_found:
            print(f"Computing sharp mask overlap for index {i}")
            compute_sharp_mask_overlap(
                template_list, [i], angle_list_path, parent_folder_path, angles_order=angles_order
            )


def compute_dist_maps_voxels(template_list, indices, parent_folder_path, morph_footprint=(2, 2, 2)):
    temp_df = pd.read_csv(template_list, index_col=0)

    for i in indices:
        if not temp_df.at[i, "Done"]:
            continue

        structure_name = temp_df.at[i, "Structure"]

        degrees = temp_df.at[i, "Degrees"]
        output_base = create_output_base_name(i)
        output_folder = create_output_folder_path(parent_folder_path, structure_name, temp_df.at[i, "Output folder"])
        scores_map = cryomap.read(output_folder + output_base + "_scores.em")
        cc_mask = cryomask.spherical_mask(np.asarray(scores_map.shape), radius=10)
        scores_map *= cc_mask
        peak_center, _, _ = tmana.create_starting_parameters_2D(scores_map)

        dist_names = ["dist_all", "dist_normals", "dist_inplane"]

        for j, value in enumerate(dist_names):
            dist_map = cryomap.read(output_folder + output_base + "_angles_" + value + ".em")
            dist_map[peak_center[0], peak_center[1], peak_center[2]] = degrees

            if j == 0:
                dist_map = np.where(dist_map <= 2.0 * degrees, 1.0, 0.0)
            else:
                dist_map = np.where(dist_map <= degrees, 1.0, 0.0)

            dist_label = measure.label(dist_map, connectivity=1)
            dist_props = pd.DataFrame(measure.regionprops_table(dist_label, properties=("label", "area", "solidity")))
            peak_label = dist_label[peak_center[0], peak_center[1], peak_center[2]]
            label_vc = dist_props.loc[dist_props["label"] == peak_label, "area"].values
            column_name = "VC " + value
            temp_df.at[i, column_name] = label_vc

            label_sol = dist_props.loc[dist_props["label"] == peak_label, "solidity"].values
            column_name = "Solidity " + value
            temp_df.at[i, column_name] = label_sol

            dist_label = np.where(dist_label == peak_label, 1.0, 0.0)
            cryomap.write(
                dist_label, output_folder + output_base + "_angles_" + value + "_label.em", data_type=np.single
            )

            open_label = morphology.binary_opening(dist_label, footprint=np.ones(morph_footprint), out=None)
            open_label = measure.label(open_label, connectivity=1)
            peak_label = open_label[peak_center[0], peak_center[1], peak_center[2]]
            open_label = np.where(open_label == peak_label, 1.0, 0.0)

            label_vc = np.count_nonzero(open_label)
            column_name = "VCO " + value
            temp_df.at[i, column_name] = label_vc
            cryomap.write(
                open_label, output_folder + output_base + "_angles_" + value + "_label_open.em", data_type=np.single
            )
            # print(column_name, label_vc)

            open_dim = cryomask.get_mass_dimensions(open_label)
            for d, dim in enumerate(["x", "y", "z"]):
                column_name = "O " + value + " " + dim
                temp_df.at[i, column_name] = open_dim[d]

        temp_df.to_csv(template_list)  # to have saved what was finished in case of a crush


def compute_center_peak_stats_and_profiles(template_list, indices, parent_folder_path):
    temp_df = pd.read_csv(template_list, index_col=0)

    for i in indices:
        if not temp_df.at[i, "Done"]:
            continue

        structure_name = temp_df.at[i, "Structure"]

        output_base = create_output_base_name(i)
        output_folder = create_output_folder_path(parent_folder_path, structure_name, temp_df.at[i, "Output folder"])
        scores_map = cryomap.read(output_folder + output_base + "_scores.em")
        cc_mask = cryomask.spherical_mask(np.asarray(scores_map.shape), radius=10)
        masked_map = scores_map * cc_mask
        peak_center, peak_value, _ = tmana.create_starting_parameters_2D(masked_map)
        _, _, line_profiles = tmana.create_starting_parameters_1D(scores_map)

        temp_df.at[i, "Peak value"] = peak_value

        line_pd = pd.DataFrame(data=line_profiles, columns=["x", "y", "z"])
        line_pd.to_csv(output_folder + output_base + "_peak_line_profiles.csv")

        for j, dim in enumerate(["x", "y", "z"]):
            peak_difference = (
                peak_value - (line_profiles[peak_center[j] - 1, j] + line_profiles[peak_center[j] + 1, j]) / 2.0
            )
            temp_df.at[i, "Drop " + dim] = peak_difference
            temp_df.at[i, "Peak " + dim] = peak_center[j]

        for r in range(1, 6):
            cc_mask = cryomask.spherical_mask(np.asarray(scores_map.shape), radius=r, center=peak_center)
            masked_map = scores_map[np.nonzero(scores_map * cc_mask)]
            temp_df.at[i, "Mean " + str(r)] = np.mean(masked_map)
            temp_df.at[i, "Median " + str(r)] = np.median(masked_map)
            temp_df.at[i, "Var " + str(r)] = np.var(masked_map)

        temp_df.to_csv(template_list)


def analyze_rotations(
    tomogram,
    template,
    template_mask,
    input_angles,
    wedge_mask_tomo=None,
    wedge_mask_tmpl=None,
    output_file=None,
    cc_radius=3,
    angular_offset=None,
    starting_angle=None,
    c_symmetry=1,
    angles_order="zxz",
):
    angles = ioutils.rot_angles_load(input_angles, angles_order)
    # angles = angles[0:4,:]

    if starting_angle is None:
        starting_angle = np.asarray([0, 0, 0])

    if np.any(starting_angle):
        rots = srot.from_euler("zxz", angles=angles, degrees=True)
        add_rot = srot.from_euler("zxz", angles=starting_angle, degrees=True)
        new_rot = rots * add_rot
        angles = new_rot.as_euler("zxz", degrees=True)

    if angular_offset is not None and np.any(angular_offset):
        rots = srot.from_euler("zxz", angles=angles, degrees=True)
        add_rot = srot.from_euler("zxz", angles=angular_offset, degrees=True)
        new_rot = rots * add_rot
        angles = new_rot.as_euler("zxz", degrees=True)

    # angles = angles[0:20,:]

    tomo = cryomap.read(tomogram)
    tmpl = cryomap.read(template)
    mask = cryomap.read(template_mask)

    # would be possibly faster to pad tmpl2 and mask after the rotation, but less readible
    if np.any(tomo.shape < tmpl.shape):
        tomo = cryomap.pad(tomo, tmpl.shape)
        output_size = tmpl.shape
    elif np.any(tomo.shape > tmpl.shape):
        tmpl = cryomap.pad(tmpl, tomo.shape)
        mask = cryomap.pad(mask, tomo.shape)
        output_size = tomo.shape
    else:
        output_size = tomo.shape

    cc_mask = cryomask.spherical_mask(np.array(output_size), radius=cc_radius).astype(np.single)

    if wedge_mask_tomo is not None:
        wedge_tomo = cryomap.read(wedge_mask_tomo)
        conj_target, conj_target_sq = cryomap.calculate_conjugates(tomo, wedge_tomo)
    else:
        conj_target, conj_target_sq = cryomap.calculate_conjugates(tomo)

    if wedge_mask_tmpl is not None:
        wedge_tmpl = cryomap.read(wedge_mask_tmpl)

    starting_angles = np.tile(starting_angle, (angles.shape[0], 1))
    ang_dist, cone, inplane = geom.compare_rotations(starting_angles, angles, c_symmetry=c_symmetry)

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

    final_ccc_map = np.full(output_size, -1)
    final_angles_map = np.full(output_size, -1)

    for i, a in enumerate(angles):
        rot_ref = cryomap.rotate(tmpl, rotation_angles=a, spline_order=1).astype(np.single)
        rot_mask = cryomap.rotate(mask, rotation_angles=a, spline_order=1).astype(np.single)

        rot_mask[rot_mask < 0.001] = 0.0  # Cutoff values
        rot_mask[rot_mask > 1.000] = 1.0  # Cutoff values

        if wedge_mask_tmpl is not None:
            rot_ref = np.fft.ifftn(np.fft.fftn(rot_ref) * wedge_tmpl).real

        norm_ref = cryomap.normalize_under_mask(rot_ref, rot_mask)
        masked_ref = norm_ref * rot_mask

        cc_map = cryomap.calculate_flcf(masked_ref, rot_mask, conj_target=conj_target, conj_target_sq=conj_target_sq)
        z_score = (cc_map - np.mean(cc_map)) / np.std(cc_map)
        max_idx = np.argmax((final_ccc_map, cc_map), 0).astype(bool)
        final_ccc_map = np.maximum(final_ccc_map, cc_map)
        final_angles_map[max_idx] = i + 1

        masked_map = cc_map * cc_mask
        z_score_masked = z_score * cc_mask

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

    final_ccc_map = np.clip(final_ccc_map, 0.0, 1.0)
    final_ccc_map_masked = final_ccc_map * cc_mask

    if output_file is not None:
        res_table.to_csv(output_file + ".csv", index=False)
        cryomap.write(file_name=output_file + "_scores.em", data_to_write=final_ccc_map, data_type=np.single)
        # cryomap.write(file_name=output_file + '_scores_masked.em', data_to_write = final_ccc_map_masked, data_type=np.single)
        cryomap.write(file_name=output_file + "_angles.em", data_to_write=final_angles_map, data_type=np.single)

    return res_table, final_ccc_map, final_angles_map, final_ccc_map_masked


def run_analysis(template_list, indices, angle_list_path, wedge_path, parent_folder_path, cc_radius_tol=10):
    temp_df = pd.read_csv(template_list, index_col=0)

    for i in indices:
        structure_name = temp_df.at[i, "Structure"]
        tmpl_name = temp_df.at[i, "Template"]
        tmpl_folder = create_structure_path(parent_folder_path, structure_name)
        template = create_em_path(parent_folder_path, structure_name, tmpl_name)
        mask = create_em_path(parent_folder_path, structure_name, temp_df.at[i, "Mask"])
        angle_list = angle_list_path + temp_df.at[i, "Angles"]

        wedge_tomo = None
        wedge_tmpl = None

        if temp_df.at[i, "Compare"] == "tmpl":
            tomo = template

        elif temp_df.at[i, "Compare"] == "subtomo":
            tomo = create_em_path(parent_folder_path, structure_name, temp_df.at[i, "Tomo map"])
            tomo_number = re.findall(r"\d+", temp_df.at[i, "Tomogram"])[0]

            if temp_df.at[i, "Apply wedge"]:
                wedge_tomo, wedge_tmpl = create_wedge_names(
                    wedge_path, tomo_number, temp_df.at[i, "Boxsize"], temp_df.at[i, "Binning"]
                )
        else:
            tomo = create_em_path(parent_folder_path, temp_df.at[i, "Compare"], temp_df.at[i, "Tomo map"])

        starting_angle = temp_df.loc[[i], ["Phi", "Theta", "Psi"]].to_numpy()

        if temp_df.at[i, "Apply angular offset"]:
            angular_offset = np.full((3,), temp_df.at[i, "Degrees"] / 2.0)
        else:
            angular_offset = np.asarray([0, 0, 0])

        c_symmetry = temp_df.at[i, "Symmetry"]
        output_base = create_output_base_name(i)
        output_folder = create_output_folder_path(parent_folder_path, structure_name, i)

        temp_df.at[i, "Output folder"] = create_output_folder_name(i)

        Path(output_folder).mkdir(parents=True, exist_ok=True)

        _ = analyze_rotations(
            tomogram=tomo,
            template=template,
            template_mask=mask,
            input_angles=angle_list,
            wedge_mask_tomo=wedge_tomo,
            wedge_mask_tmpl=wedge_tmpl,
            output_file=output_folder + "/" + output_base,
            angular_offset=angular_offset,
            starting_angle=starting_angle,
            cc_radius=cc_radius_tol,
            c_symmetry=c_symmetry,
        )[0]

        angles_map = output_folder + "/" + output_base + "_angles.em"
        _, _, _ = tmana.create_angular_distance_maps(angles_map, angle_list, write_out_maps=True)

        temp_df.at[i, "Done"] = True
        temp_df.to_csv(template_list)  # to have saved what was finished in case of a crush


def run_angle_analysis(
    template_list, indices, wedge_path, parent_folder_path, angular_range=359, write_output=False, cc_radius_tol=10
):
    temp_df = pd.read_csv(template_list, index_col=0)

    for i in indices:
        structure_name = temp_df.at[i, "Structure"]
        tmpl_name = temp_df.at[i, "Template"]
        tmpl_folder = create_structure_path(parent_folder_path, structure_name)
        template = create_em_path(parent_folder_path, structure_name, tmpl_name)
        mask = create_em_path(parent_folder_path, structure_name, temp_df.at[i, "Mask"])

        wedge_tomo = None
        wedge_tmpl = None

        if temp_df.at[i, "Compare"] == "tmpl":
            tomo = template

        elif temp_df.at[i, "Compare"] == "subtomo":
            tomo = create_em_path(parent_folder_path, structure_name, temp_df.at[i, "Tomo map"])
            tomo_number = re.findall(r"\d+", temp_df.at[i, "Tomogram"])[0]

            if temp_df.at[i, "Apply wedge"]:
                wedge_tomo, wedge_tmpl = create_wedge_names(
                    wedge_path, tomo_number, temp_df.at[i, "Boxsize"], temp_df.at[i, "Binning"]
                )
        else:
            tomo = create_em_path(parent_folder_path, temp_df.at[i, "Compare"], temp_df.at[i, "Tomo map"])

        starting_angle = temp_df.loc[[i], ["Phi", "Theta", "Psi"]].to_numpy()
        c_symmetry = temp_df.at[i, "Symmetry"]

        results = np.zeros((angular_range, 8, 3))

        n_bins = 100
        final_hist = np.zeros((n_bins, 3))
        for a in range(angular_range):
            angles = np.full((3, 3), a)
            angles[1, 0] = 0  # only cone rotation
            angles[2, 1:] = 0  # only inplane rotation

            for j in range(3):
                res_df, cc_map, _, _ = analyze_rotations(
                    tomogram=tomo,
                    template=template,
                    template_mask=mask,
                    input_angles=angles[j, :].reshape(1, 3),
                    wedge_mask_tomo=wedge_tomo,
                    wedge_mask_tmpl=wedge_tmpl,
                    output_file=None,
                    starting_angle=starting_angle,
                    cc_radius=cc_radius_tol,
                    c_symmetry=c_symmetry,
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

        if write_output:
            output_base = create_output_folder_path(
                parent_folder_path, structure_name, temp_df.at[i, "Output folder"]
            ) + create_output_base_name(i)
            final_df.to_csv(output_base + "_gradual_angles_analysis.csv")
            hist_df = pd.DataFrame(data=final_hist, columns=["ang_dist", "cone_dist", "inplane_dist"])
            hist_df.to_csv(output_base + "_gradual_angles_histograms.csv")


def create_summary_pdf(template_list, indices, parent_folder_path):
    temp_df = pd.read_csv(template_list, index_col=0)

    for i in indices:
        if not temp_df.at[i, "Done"]:
            continue

        structure_name = temp_df.at[i, "Structure"]
        output_base = create_output_base_name(i)
        output_folder = create_output_folder_path(parent_folder_path, structure_name, temp_df.at[i, "Output folder"])

        if temp_df.at[i, "Compare"] == "tmpl":
            title_end = "self"
        elif temp_df.at[i, "Compare"] == "subtomo":
            title_end = "subtomo (" + temp_df.at[i, "Tomo map"] + ")"
        else:
            title_end = temp_df.at[i, "Compare"] + " (" + temp_df.at[i, "Tomo map"] + ")"

        figure_title = structure_name + " (" + temp_df.at[i, "Template"] + ") matched with " + title_end

        extenstion_list = ["_scores.em", "_angles_dist_all.em", "_angles_dist_normals.em", "_angles_dist_inplane.em"]

        tmpl_info = [
            "Symmetry",
            "Apply wedge",
            "Degrees",
            "Apply angular offset",
            "Binning",
            "Pixelsize",
            "Boxsize",
            "Voxels",
            "Voxels TM",
            "Solidity",
        ]

        temp_df = temp_df.round(decimals=4)
        tmpl_dict = pd.DataFrame(temp_df.loc[temp_df.index[i], tmpl_info]).to_dict()
        tmpl_dict = tmpl_dict.get(i)

        peak_info = ["Peak value"]
        peak_dict = pd.DataFrame(temp_df.loc[temp_df.index[i], peak_info]).to_dict()
        peak_dict = peak_dict.get(i)

        dist_names = ["dist_all", "dist_normals", "dist_inplane"]
        dist_dict = {}

        values_temp = np.zeros((3,))
        peak_center = np.zeros((3,))
        bb_dim = np.zeros((3,))
        peak_drop = np.zeros((3,))
        for d, dim in enumerate(["x", "y", "z"]):
            peak_center[d] = temp_df.at[i, "Peak " + dim]
            bb_dim[d] = temp_df.at[i, "Dim " + dim]
            peak_drop[d] = temp_df.at[i, "Drop " + dim]

        peak_dict["Peak center"] = peak_center
        tmpl_dict["Dimensions"] = bb_dim
        peak_dict["Drop"] = peak_drop

        dist_vc = np.zeros((3,))
        dist_sol = np.zeros((3,))
        dist_vco = np.zeros((3,))
        for d, dname in enumerate(dist_names):
            dist_vc[d] = temp_df.at[i, "VC " + dname]
            dist_sol[d] = temp_df.at[i, "Solidity " + dname]
            dist_vco[d] = temp_df.at[i, "VCO " + dname]

        # dist_dict['Dummy'] = 1
        dist_dict["Dist maps Solidity"] = dist_sol
        dist_dict["Dist maps VC"] = dist_vc
        dist_dict["Dist maps VC open"] = dist_vco

        for d, dname in enumerate(dist_names):
            for j, dim in enumerate(["x", "y", "z"]):
                values_temp[j] = temp_df.at[i, "O " + dname + " " + dim]

            dist_dict["Open " + dname] = values_temp.copy()

        values_temp = np.zeros((5,))

        for sts in ["Mean", "Median", "Var"]:
            for r in range(1, 6):
                values_temp[r - 1] = temp_df.at[i, sts + " " + str(r)]
            peak_dict[sts] = values_temp.copy()

        dicts = []
        with np.printoptions(
            precision=4,
            suppress=True,
        ):
            dicts.append([[k, str(v)] for k, v in tmpl_dict.items()])
            dicts.append([[k, str(v)] for k, v in peak_dict.items()])
            dicts.append([[k, str(v)] for k, v in dist_dict.items()])

        cross_slices = []
        tight_mask = cryomap.read(create_em_path(parent_folder_path, structure_name, temp_df.at[i, "Tight mask"]))
        cross_slices.append(cryomap.get_cross_slices(tight_mask))

        for m, ext_name in enumerate(extenstion_list):
            input_map = cryomap.read(output_folder + output_base + ext_name)

            if m == 0:
                cross_slices.append(cryomap.get_cross_slices(input_map, slice_numbers=peak_center, axis=[0, 1, 2]))

            cross_slices.append(
                cryomap.get_cross_slices(input_map, slice_half_dim=5, slice_numbers=peak_center, axis=[0, 1, 2])
            )

        grid_rows_n = 2
        unit_size = 5
        grid_row_ratio = [1.6 * unit_size, 6 * unit_size]

        # check if file for histogram analysis exists or not
        hist_file = output_folder + output_base + "_gradual_angles_histograms.csv"
        add_hist = False
        last_row = 1

        if os.path.isfile(hist_file):
            grid_rows_n += 1
            add_hist = True
            last_row = 2
            grid_row_ratio = [1.6 * unit_size, 0.8 * unit_size, 6 * unit_size]

        fig_height = sum(grid_row_ratio) + 0.4
        widths = [unit_size, unit_size, unit_size, unit_size * 0.05]

        fig = plt.figure(layout="constrained", figsize=(sum(widths), fig_height))
        fig.suptitle(figure_title, fontsize=16, y=1.008)

        grid_base = gridspec.GridSpec(grid_rows_n, 1, figure=fig, height_ratios=grid_row_ratio)
        grid_rows = [gridspec.GridSpecFromSubplotSpec(2, 3, subplot_spec=grid_base[0])]

        if add_hist:
            grid_rows.append(gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=grid_base[1]))

        grid_rows.append(
            gridspec.GridSpecFromSubplotSpec(
                6,
                4,
                subplot_spec=grid_base[last_row],
                height_ratios=[unit_size, unit_size, unit_size, unit_size, unit_size, unit_size],
                width_ratios=widths,
            )
        )

        table_plots = []
        for j in range(3):
            table_plots.append(fig.add_subplot(grid_rows[0][0, j]))

        col_width = [[0.4, 0.6], [0.3, 0.7], [0.4, 0.6]]
        for tbl, dc, cw in zip(table_plots, dicts, col_width):
            tbl_h = tbl.table(colWidths=cw, cellText=dc, bbox=[0, 0, 1, 1])
            tbl_h.auto_set_font_size(False)
            tbl_h.set_fontsize(10)

        rot_info = pd.read_csv(output_folder + output_base + ".csv", index_col=0)
        line_profiles = pd.read_csv(output_folder + output_base + "_peak_line_profiles.csv", index_col=0)

        if "Tight mask overlap" not in rot_info.columns:
            print(i)
            continue

        ad_plt = sns.scatterplot(
            ax=fig.add_subplot(grid_rows[0][1, 0]),
            data=rot_info,
            x="Tight mask overlap",
            y="ccc_masked",
            linewidth=0,
            s=5,
        )
        ad_plt.set(ylabel="CCC", xlabel="Tight mask overlap (in voxels)")
        ad_plt = sns.scatterplot(
            ax=fig.add_subplot(grid_rows[0][1, 1]), data=rot_info, x="ang_dist", y="ccc_masked", linewidth=0, s=5
        )
        ad_plt.set(ylabel=None, xlabel="Angular distance (in degrees)")
        ad_plt = sns.lineplot(ax=fig.add_subplot(grid_rows[0][1, 2]), data=line_profiles[["x", "y", "z"]])
        ad_plt.set(ylabel=None, xlabel="Position (in voxels)")

        if add_hist:
            hist_info = pd.read_csv(hist_file, index_col=0)
            hist_plt = fig.add_subplot(grid_rows[1][0, 0])
            sns.lineplot(ax=hist_plt, data=hist_info, x=np.linspace(0.0, 1.0, num=100), y="ang_dist")
            sns.lineplot(ax=hist_plt, data=hist_info, x=np.linspace(0.0, 1.0, num=100), y="cone_dist")
            sns.lineplot(ax=hist_plt, data=hist_info, x=np.linspace(0.0, 1.0, num=100), y="inplane_dist")
            hist_plt.set(ylim=(0, 250), ylabel="Number of CCC values (bin size 0.1)", xlabel="CCC")

            hist_info = pd.read_csv(output_folder + output_base + "_gradual_angles_analysis.csv", index_col=0)
            hist_plt = fig.add_subplot(grid_rows[1][0, 1])
            sns.lineplot(ax=hist_plt, data=hist_info, x=np.linspace(0.0, 359.0, num=359), y="ccc_masked")
            sns.lineplot(ax=hist_plt, data=hist_info, x=np.linspace(0.0, 359.0, num=359), y="cone_ccc_masked")
            sns.lineplot(ax=hist_plt, data=hist_info, x=np.linspace(0.0, 359.0, num=359), y="inplane_ccc_masked")
            hist_plt.set(ylabel="CCC", xlabel="Rotation (in degrees)")

        for c, slice in enumerate(cross_slices):
            if c <= 1:
                use_annot = False
            else:
                use_annot = True

            f_fmt = ".2f"
            if c == 0:
                data_max = 1.0
                c_scheme = "gray"
            elif c < 3:
                data_max = temp_df.at[i, "Peak value"]
                c_scheme = "viridis"
            else:
                data_max = 180
                c_scheme = "cividis"
                f_fmt = ".1f"

            hm1 = fig.add_subplot(grid_rows[last_row][c, 0])
            hm2 = fig.add_subplot(grid_rows[last_row][c, 1])
            hm3 = fig.add_subplot(grid_rows[last_row][c, 2])
            cb = fig.add_subplot(grid_rows[last_row][c, 3])

            sns.heatmap(
                ax=hm1,
                data=np.flipud(slice[0].T),
                annot=use_annot,
                fmt=f_fmt,
                square=True,
                cmap=c_scheme,
                yticklabels=False,
                xticklabels=False,
                vmin=0,
                vmax=data_max,
                cbar_ax=cb,
            )
            sns.heatmap(
                ax=hm2,
                data=np.flipud(slice[1].T),
                annot=use_annot,
                fmt=f_fmt,
                square=True,
                cmap=c_scheme,
                yticklabels=False,
                xticklabels=False,
                vmin=0,
                vmax=data_max,
                cbar=False,
            )
            sns.heatmap(
                ax=hm3,
                data=np.flipud(slice[2].T),
                annot=use_annot,
                fmt=f_fmt,
                square=True,
                cmap=c_scheme,
                yticklabels=False,
                xticklabels=False,
                vmin=0,
                vmax=data_max,
                cbar=False,
            )

        # plt.tight_layout() # or layout = "constrained" in figure
        plt.savefig(output_folder + output_base + "_summary.pdf", transparent=True, bbox_inches="tight")
        plt.close()


###########################################################################################################################################
####################### Following functions are not used in the current analysis but might come handy later ###############################
###########################################################################################################################################


# Check what kind of descriptors skimage can offer
def get_shape_stats(template_list, indices, shape_type, parent_folder_path):
    temp_df = pd.read_csv(template_list, index_col=0)

    for i in indices:
        structure_name = temp_df.at[i, "Structure"]
        sharp_mask = cryomap.read(create_em_path(parent_folder_path, structure_name, temp_df.at[i, "Tight mask"]))

        mask_label = measure.label(sharp_mask, connectivity=1)
        mask_stats = pd.DataFrame(
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

        output_base = (
            create_structure_path(parent_folder_path, structure_name)
            + temp_df.at[i, "Output folder"]
            + "/id_"
            + str(i)
            + "_shape_stats_"
        )
        mask_stats.to_csv(output_base + shape_type + ".csv")


def plot_scores_and_peaks(peak_files, plot_title=None, output_file=None):
    n_rows = len(peak_files)
    row_size = 4 * n_rows
    fig, axs = plt.subplots(n_rows, 4, figsize=(row_size, 100), gridspec_kw={"width_ratios": [20, 20, 20, 1]})

    if plot_title is not None:
        fig.suptitle(plot_title, fontsize=28, y=1.005)

    peak_center, peak_height, _ = tmana.create_starting_parameters_2D(peak_files[0])

    for i, p in enumerate(peak_files):
        _, _, peaks = tmana.create_starting_parameters_2D(p, peak_center=peak_center)
        data_min = np.amin(p)

        sns.heatmap(
            ax=axs[i][0],
            data=np.flipud(peaks[:, :, 0].T),
            square=True,
            cmap="viridis",
            annot=False,
            yticklabels=False,
            xticklabels=False,
            vmin=data_min,
            vmax=peak_height,
            cbar_ax=axs[i][3],
        )
        sns.heatmap(
            ax=axs[i][1],
            data=np.flipud(peaks[:, :, 1].T),
            square=True,
            cmap="viridis",
            annot=False,
            yticklabels=False,
            xticklabels=False,
            vmin=data_min,
            vmax=peak_height,
            cbar=False,
        )
        sns.heatmap(
            ax=axs[i][2],
            data=np.flipud(peaks[:, :, 2].T),
            square=True,
            cmap="viridis",
            annot=False,
            yticklabels=False,
            xticklabels=False,
            vmin=data_min,
            vmax=peak_height,
            cbar=False,
        )

    plt.tight_layout()

    if output_file is not None:
        plt.savefig(output_file, transparent=True, bbox_inches="tight")


def compute_peak_shapes(template_list, indices, parent_folder_path):
    temp_df = pd.read_csv(template_list, index_col=0)

    for i in indices:
        if not temp_df.at[i, "Done"]:
            continue

        if temp_df.at[i, "Structure"] == "membrane":
            continue

        structure_name = temp_df.at[i, "Structure"]

        output_base = create_output_base_name(i)
        output_folder = create_ouptut_folder_path(parent_folder_path, structure_name, temp_df.at[i, "Output folder"])
        scores_map = cryomap.read(output_folder + output_base + "_scores.em")

        t_map, tp_shape, peak_value, t_th_map, t_surf = tmana.evaluate_scores_map(
            scores_map, label_type="ellipsoid", threshold_type="triangle"
        )
        g_map, gp_shape, _, g_th_map, g_surf = tmana.evaluate_scores_map(
            scores_map, label_type="ellipsoid", threshold_type="gauss"
        )
        h_map, hp_shape, _, h_th_map, h_surf = tmana.evaluate_scores_map(
            scores_map, label_type="ellipsoid", threshold_type="hard"
        )

        for t, p in enumerate(["x", "y", "z"]):
            temp_df.at[i, "TP " + p] = np.round(tp_shape[t], 3)
            temp_df.at[i, "GP " + p] = np.round(gp_shape[t], 3)
            temp_df.at[i, "HP " + p] = np.round(hp_shape[t], 3)

        temp_df.at[i, "Peak value"] = peak_value

        temp_df.to_csv(template_list)  # to have saved what was finished in case of a crush

        plot_scores_and_peaks(
            [scores_map, t_th_map, t_surf, t_map, g_th_map, g_surf, g_map, h_th_map, h_surf, h_map],
            plot_title=structure_name + " id" + str(i),
            output_file=output_folder + "peaks.png",
        )


## Function to change the output folder base name
def rename_folders(template_list, indices, parent_folder_path):
    temp_df = pd.read_csv(template_list, index_col=0)

    for i in indices:
        structure_path = create_structure_path(parent_folder_path, temp_df.at[i, "Structure"])
        current_folder_path = structure_path + temp_df.at[i, "Output folder"]
        new_folder_name = create_output_folder_name(i)
        new_folder_path = structure_path + new_folder_name

        os.rename(current_folder_path, new_folder_path)
        temp_df.at[i, "Output folder"] = new_folder_name
        temp_df.to_csv(template_list)  # to have saved what was finished in case of a crush


# Function to change the names of TM results -> facilitate reading later on
def rename_scores_angles(template_list, indices, parent_folder_path):
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


def correct_bbox(template_list, indices):
    temp_df = pd.read_csv(template_list, index_col=0)

    for i in indices:
        if not temp_df.at[i, "Done"]:
            continue

        list_to_correct = ["Dim", "O dist_all", "O dist_normals", "O dist_inplane"]

        for l in list_to_correct:
            for d in ["x", "y", "z"]:
                temp_df.at[i, l + " " + d] += 1

        temp_df.to_csv(template_list)


def recompute_dist_maps(template_list, indices, parent_folder_path, angle_list_path):
    temp_df = pd.read_csv(template_list, index_col=0)

    for i in indices:
        if not temp_df.at[i, "Done"]:
            continue

        structure_name = temp_df.at[i, "Structure"]

        output_base = create_output_base_name(i)
        output_folder = create_output_folder_path(parent_folder_path, structure_name, temp_df.at[i, "Output folder"])
        angles_map = output_folder + output_base + "_angles.em"
        angle_list = angle_list_path + temp_df.at[i, "Angles"]
        c_symmetry = temp_df.at[i, "Symmetry"]
        _, _, _ = tmana.create_angular_distance_maps(angles_map, angle_list, write_out_maps=True, c_symmetry=c_symmetry)
