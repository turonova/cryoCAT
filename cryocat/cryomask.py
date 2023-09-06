import numpy as np
from cryocat import cryomaps
from scipy.spatial.transform import Rotation as srot
from skimage import filters
from scipy import ndimage
from skimage import measure
import pandas as pd
import decimal
from skimage import morphology
import numbers

def add_gaussian(input_mask, sigma):
    if sigma == 0:
        return input_mask
    else:
        return filters.gaussian(input_mask, sigma=sigma)


def write_out(input_mask, output_name):
    if output_name is not None:
        cryomaps.write(input_mask, output_name, data_type=np.single)


def rotate(input_mask, angles):
    if not np.any(angles):
        return input_mask
    else:
        return cryomaps.rotate(input_mask, rotation_angles=angles)


def postprocess(input_mask, gaussian, angles, output_name):
    mask = add_gaussian(input_mask, gaussian)
    mask = rotate(mask, angles)
    write_out(mask, output_name)

    return mask


def union(mask_list, output_name=None):
    final_mask = np.zeros(cryomaps.read(mask_list[0]).shape)

    for m in mask_list:
        mask = cryomaps.read(m)
        final_mask += mask

    final_mask = np.clip(final_mask, 0.0, 1.0)

    write_out(final_mask, output_name)

    return final_mask


def intersection(mask_list, output_name=None):
    final_mask = np.ones(cryomaps.read(mask_list[0]).shape)

    for m in mask_list:
        mask = cryomaps.read(m)
        final_mask *= mask

    final_mask = np.clip(final_mask, 0.0, 1.0)
    write_out(final_mask, output_name)

    return final_mask


def subtraction(mask_list, output_name=None):
    final_mask = cryomaps.read(mask_list[0]).shape

    for m in mask_list[1:]:
        mask = cryomaps.read(m)
        final_mask -= mask

    final_mask = np.clip(final_mask, 0.0, 1.0)
    write_out(final_mask, output_name)

    return final_mask


def difference(mask_list, output_name=None):
    union_mask = union(mask_list)
    inter_mask = intersection(mask_list)

    final_mask = union_mask - inter_mask
    final_mask = np.clip(final_mask, 0.0, 1.0)
    write_out(final_mask, output_name)

    return final_mask


def spherical_mask(mask_size, radius=None, center=None, gaussian=0, output_name=None):
    
    if center is None:
        center = mask_size // 2
        if isinstance(center,numbers.Number):           
            center = (center, center, center)


    if radius is None:
        radius = np.amin(mask_size) // 2

    x, y, z = np.mgrid[0 : mask_size[0] : 1, 0 : mask_size[1] : 1, 0 : mask_size[2] : 1]
    mask = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2 + (z - center[2]) ** 2)
    mask[mask > radius] = 0
    mask[mask > 0] = 1
    mask[center[0], center[1], center[2]] = 1

    mask = postprocess(mask, gaussian, np.asarray([0, 0, 0]), output_name)

    return mask


def ellipsoid_mask(
    mask_size,
    radii=None,
    center=None,
    gaussian=0,
    output_name=None,
    angles=np.asarray([0, 0, 0]),
):
    if isinstance(mask_size, tuple):
        if len(mask_size) == 3:
            mask_shape = np.asarray(mask_size)
        else:
            raise ValueError(
                "Mask size have to be a single number or tuple of length 3!"
            )
    else:
        mask_shape = np.full((3,), mask_size).astype(int)

    if center is None:
        center = mask_shape // 2
    else:
        center = np.asarray(center)

    if radii is None:
        radii = mask_shape // 2
    else:
        radii = np.asarray(radii)

    # Build a grid and get its points as a list
    xi = tuple(np.linspace(1, s, s) - np.floor(0.5 * s) for s in mask_shape)

    # Build a list of points forming the grid
    xi = np.meshgrid(*xi, indexing="ij")
    points = np.array(xi).reshape(3, -1)[::-1]

    # Find grid center
    grid_center = 0.5 * mask_shape - center
    grid_center = np.tile(grid_center.reshape(3, 1), (1, points.shape[1]))

    # Reorder coordinates back to ZYX to match the order of numpy array axis
    points = points[:, ::-1]
    grid_center = grid_center[::-1]
    radii = radii[::-1]
    radii = np.tile(radii.reshape(3, 1), (1, points.shape[1]))

    # Draw the ellipsoid
    # dx**2 + dy**2 + dz**2 = r**2
    # dx**2 / r**2 + dy**2 / r**2 + dz**2 / r**2 = 1
    ellipsoid = (points - grid_center) ** 2
    ellipsoid = ellipsoid / radii**2
    # Sum dx, dy, dz / r**2
    distance = np.sum(ellipsoid, axis=0).reshape(mask_shape)

    mask = distance <= 1

    mask = postprocess(mask, gaussian, angles, output_name)

    return mask


def molmap_tight_mask(
    input_map,
    threshold=0.0,
    dilation_size=0,
    gaussian=0,
    output_name=None,
    angles=np.asarray([0, 0, 0]),
):
    model = cryomaps.read(input_map)

    if dilation_size == 0:
        mask = np.where(model > threshold, 1.0, 0.0)
    else:
        mask = ndimage.binary_dilation(model, iterations=dilation_size)

    mask = postprocess(mask, gaussian, angles, output_name)

    return mask


def map_tight_mask(
    input_map,
    threshold=None,
    dilation_size=0,
    gaussian=0,
    output_name=None,
    angles=np.asarray([0, 0, 0]),
    n_regions=1,
):
    mask = cryomaps.read(input_map)

    if threshold is None:
        threshold = 3.0 * np.std(mask)
        if np.median(mask) > 0.0:
            threshold *= -1

    if threshold < 0.0:
        mask = np.where(mask < threshold, 1.0, 0.0)
    else:
        mask = np.where(mask > threshold, 1.0, 0.0)

    labeled_mask = measure.label(mask, connectivity=1)
    info_table = pd.DataFrame(
        measure.regionprops_table(
            labeled_mask,
            properties=["label", "area"],
        )
    ).set_index("label")
    info_table = info_table.reset_index()

    label_ids = (
        info_table.sort_values(by="area", ascending=False)
        .head(n_regions)["label"]
        .values
    )
    # label_id = info_table.iloc[info_table['area'].idxmax()]['label']
    mask = np.where(np.isin(labeled_mask, [label_ids]), 1.0, 0.0)

    if dilation_size > 0:
        mask = ndimage.binary_dilation(mask, iterations=dilation_size)

    mask = postprocess(mask, gaussian, angles, output_name)

    return mask


def get_bounding_box(input_mask):
    mask = cryomaps.read(input_mask)
    epsilon = 0.00001
    i, j, k = np.asarray(mask > epsilon).nonzero()
    if i.shape[0] == 0:
        return np.zeros((3,)).astype(int), np.zeros((3,)).astype(int)
    start_ids = np.array([min(i), min(j), min(k)])
    end_ids = np.array([max(i), max(j), max(k)])

    return start_ids, end_ids


def get_mass_dimensions(input_mask):
    mask = cryomaps.read(input_mask)

    start_ids, end_ids = get_bounding_box(mask)

    return end_ids - start_ids + 1


def get_mass_center(input_mask):
    mask = cryomaps.read(input_mask)
    start_ids, end_ids = get_bounding_box(mask)

    mask_center = (start_ids + end_ids) / 2

    for i in range(3):
        mask_center[i] = (
            decimal.Decimal(mask_center[i]).to_integral_value(
                rounding=decimal.ROUND_HALF_UP
            )
            + 1
        )

    return mask_center.astype(int)


def shrink_full_mask(input_mask, shrink_factor, output_name=None):
    dim_x, dim_y, dim_z = input_mask.shape
    filled_mask = np.zeros(input_mask.shape)

    for z in range(dim_z):
        for y in range(dim_y):
            not_zero = np.flatnonzero(input_mask[:, y, z])
            if not_zero.size != 0:
                s_idx = not_zero[0] + shrink_factor
                e_idx = not_zero[-1] + 1 - shrink_factor
                if e_idx - s_idx > 0:
                    filled_mask[s_idx:e_idx, y, z] = 1

    for z in range(dim_z):
        for x in range(dim_x):
            not_zero = np.flatnonzero(filled_mask[x, :, z])
            if not_zero.size != 0:
                s_idx = not_zero[0] + shrink_factor
                e_idx = not_zero[-1] + 1 - shrink_factor
                if e_idx - s_idx > 0:
                    filled_mask[x, s_idx:e_idx, z] += 1

    for y in range(dim_y):
        for x in range(dim_x):
            not_zero = np.flatnonzero(filled_mask[x, y, :])
            if not_zero.size != 0:
                s_idx = not_zero[0] + shrink_factor
                e_idx = not_zero[-1] + 1 - shrink_factor
                if e_idx - s_idx > 0:
                    filled_mask[x, y, s_idx:e_idx] += 1

    filled_mask = np.where(filled_mask == 3, 1, 0)

    filled_mask = morphology.binary_opening(filled_mask, footprint=np.ones((2, 2, 2)))
    filled_mask = morphology.binary_closing(filled_mask)

    write_out(filled_mask, output_name)

    return filled_mask


def fill_hollow_mask(input_mask, output_name=None):
    dim_x, dim_y, dim_z = input_mask.shape
    filled_mask = np.zeros(input_mask.shape)

    for z in range(dim_z):
        for y in range(dim_y):
            not_zero = np.flatnonzero(input_mask[:, y, z])
            if not_zero.size != 0:
                filled_mask[not_zero[0] : not_zero[-1] + 1, y, z] = 1

    for z in range(dim_z):
        for x in range(dim_x):
            not_zero = np.flatnonzero(filled_mask[x, :, z])
            if not_zero.size != 0:
                filled_mask[x, not_zero[0] : not_zero[-1] + 1, z] += 1

    for y in range(dim_y):
        for x in range(dim_x):
            not_zero = np.flatnonzero(filled_mask[x, y, :])
            if not_zero.size != 0:
                filled_mask[x, y, not_zero[0] : not_zero[-1] + 1] += 1

    filled_mask = np.where(filled_mask > 0, 1, 0)

    filled_mask = morphology.binary_opening(filled_mask, footprint=np.ones((2, 2, 2)))
    filled_mask = morphology.binary_closing(filled_mask)

    write_out(filled_mask, output_name)

    return filled_mask



def compute_solidity(input_mask):

    mask_label = measure.label(input_mask)
    props = pd.DataFrame(measure.regionprops_table(mask_label,properties=['solidity']))

    return props.at[0,'solidity']

def mask_overlap(mask1,mask2,threshold=1.9):

    mask_overlap=np.where((mask1+mask2) <= threshold, 0, 1)
    
    return np.sum(mask_overlap)