import numpy as np
import pandas as pd
import gc
import re
import warnings

from skimage import measure
from skimage import morphology

from cryocat.core import cryomap
from cryocat.utils import geom
from cryocat.utils import ioutils
from cryocat.core import cryomotl
from cryocat.core import cryomask

from lmfit import models
import skimage
from scipy.spatial import KDTree
from sklearn.cluster import DBSCAN


def scores_extract_particles(
    scores_map,
    angles_map,
    angles_list,
    tomo_id,
    particle_diameter,
    object_id=None,
    scores_threshold=None,
    sigma_threshold=None,
    cluster_size=None,
    n_particles=None,
    output_path=None,
    output_type="emmotl",
    angles_order="zxz",
    symmetry="c1",
    angles_numbering=0,
    tomo_mask=None,
):
    """Extract particles from template-matching scores maps produced by GAPSTOP(TM) or STOPGAP.

    Voxels above a score threshold are collected, deduplicated using a greedy
    distance filter (keeping the highest-scoring position within each
    ``particle_diameter``-radius neighbourhood), clustered with DBSCAN, and
    assembled into a :class:`cryocat.core.cryomotl.Motl` with orientations
    looked up from the corresponding angles map.  Particle coordinates are
    stored in 1-based indexing.

    Parameters
    ----------
    scores_map : str or array-like
        Path to the scores map file or a pre-loaded array.
    angles_map : str or array-like
        Path to the angles map file or a pre-loaded array.  Each voxel stores
        the index of the best-matching angle in ``angles_list``.
    angles_list : str or array-like
        Path to the rotation-angles file or a pre-loaded (N, 3) array of
        Euler angles (phi, theta, psi).
    tomo_id : int
        Tomogram identifier written into the output motl.
    particle_diameter : float
        Particle diameter in voxels.  Used both as the exclusion radius for
        the greedy deduplication step and as twice the ``eps`` parameter for
        DBSCAN clustering.
    object_id : int, optional
        Object identifier written into the output motl.  Defaults to 1.
    scores_threshold : float, optional
        Absolute score threshold.  Voxels with scores at or below this value
        are discarded.  Takes priority over ``sigma_threshold``; if set,
        ``sigma_threshold`` is ignored.  Defaults to None.
    sigma_threshold : float, optional
        Threshold expressed as mean + ``sigma_threshold`` * std of the scores
        map.  Only used when ``scores_threshold`` is None.  If both are None
        the threshold is determined automatically via
        :meth:`cryocat.analysis.tmana.compute_scores_map_threshold_triangle`.
        Defaults to None.
    cluster_size : int, optional
        Minimum number of candidate positions required for a DBSCAN cluster to
        be retained.  Clusters with fewer members are discarded entirely.
        Defaults to None (all clusters kept).
    n_particles : int, optional
        Maximum number of particles to return, taken in descending score order.
        Defaults to None (no limit).
    output_path : str, optional
        Path to write the output motl.  No file is written when None.
        Defaults to None.
    output_type : str, {"emmotl", "stopgap", "relion"}
        Format of the output file.  Only used when ``output_path`` is not None.
        Defaults to "emmotl".
    angles_order : str, {"zxz", "zzx"}
        Euler-angle convention of ``angles_list``.  Use "zzx" for STOPGAP
        angle lists and "zxz" for GAPSTOP(TM) lists.  Defaults to "zxz".
    symmetry : str, default="c1"
        Cyclic symmetry to apply.  A random multiple of 360/N degrees is added
        to phi for each particle when N > 1.  Only C symmetries are supported;
        any other symmetry string issues a warning and falls back to "c1".
    angles_numbering : int, default=0
        Index offset applied to values read from ``angles_map`` before
        indexing into ``angles_list``.  STOPGAP angle maps are 1-based, so
        set this to 1; GAPSTOP(TM) maps are 0-based (default 0).
    tomo_mask : str or array-like, optional
        Path to a binary tomogram mask or a pre-loaded array.  When provided,
        the scores map is multiplied by this mask before thresholding.
        Defaults to None.

    Returns
    -------
    motl : Motl or None
        :class:`cryocat.core.cryomotl.Motl` containing extracted particle
        coordinates, scores, and orientations.  Returns ``None`` when no
        voxel exceeds the threshold.

    Raises
    ------
    UserWarning
        If a non-C symmetry string is supplied; symmetry is set to "c1".
    ValueError
        If ``output_type`` is not one of "emmotl", "stopgap", or "relion".
    """

    if symmetry.lower().startswith("c"):
        symmetry = int(re.findall(r"\d+", symmetry)[-1])
    else:
        warnings.warn(
            f"Only C symmetry is supported. Provided {symmetry} " f"is currently not supported and will be ignored."
        )
        symmetry = 1

    # load the scores map
    scores_map = cryomap.read(scores_map)

    # load the angles map
    angles_map = cryomap.read(angles_map)

    # Read angle list.
    anglist = ioutils.rot_angles_load(angles_list, angles_order=angles_order)

    # load and apply a tomogram mask if any:
    if tomo_mask is not None:
        tomo_mask = cryomap.read(tomo_mask)
        scores_map = scores_map * tomo_mask

    if object_id is None:
        object_id = 1

    if scores_threshold is not None:
        threshold = scores_threshold
    elif sigma_threshold is None:
        threshold = compute_scores_map_threshold_triangle(scores_map)
    else:
        # Set threshold for scores map by sigma value
        score_mean = scores_map.mean()
        score_std = scores_map.std(ddof=1)
        threshold = score_mean + sigma_threshold * score_std

    # Threshold and sort indices/scores
    t_idx = np.where(scores_map > threshold)

    # original piece - not clear whether this is really working
    # if n_particles is not None:
    #    k = min(n_particles, len(t_idx[0]))
    # else:
    k = len(t_idx[0])

    # Check for early termination
    if k == 0:
        return None

    k = min(k, len(scores_map[t_idx])) - 1
    s_idx = np.argpartition(-scores_map[t_idx], k)[: k + 1]
    s_idx = s_idx[np.argsort(-scores_map[t_idx][s_idx])]  # Sort for later

    # Sorted indices. s_ind[0] = x, s_ind[1] = y, s_ind[2] = z
    s_ind = np.array([t_idx[0][s_idx], t_idx[1][s_idx], t_idx[2][s_idx]])
    # n_vox = len(s_idx)

    # Create a list of tuples where each tuple is (coord, score)
    # and sort it by score in descending order
    scored_coords = sorted(zip(s_ind.T, scores_map[s_ind[0], s_ind[1], s_ind[2]]), key=lambda x: x[1], reverse=True)

    # Build a KD-tree with the coordinates
    tree = KDTree([coord for coord, score in scored_coords])

    # Remove any points within the specified particle diameter of a higher score point
    coord_to_score = {tuple(coord): score for coord, score in scored_coords}
    remaining_coords = set(coord_to_score.keys())
    filtered_coords = []

    for coord, score in scored_coords:
        if tuple(coord) not in remaining_coords:
            continue
        filtered_coords.append((coord, score))
        nearby_coords = tree.query_ball_point(coord, particle_diameter)
        for nearby_coord in nearby_coords:
            nearby_coord_tuple = tuple(scored_coords[nearby_coord][0])
            if nearby_coord_tuple in remaining_coords and coord_to_score[nearby_coord_tuple] <= score:
                remaining_coords.remove(nearby_coord_tuple)

    # Extract the coordinates from the filtered_coords list
    filtered_coords, filtered_scores = zip(*filtered_coords)
    filtered_coords = np.array(filtered_coords)
    filtered_scores = np.array(filtered_scores)

    # Use DBSCAN to cluster points
    clusterer = DBSCAN(eps=particle_diameter / 2, min_samples=1)
    cluster_labels = clusterer.fit_predict(filtered_coords)

    # Keep track of hits in case of number of particles
    filtered_hit_idx = np.zeros(len(filtered_coords), dtype=bool)

    # Count number of hits
    c = 0
    for cluster_id in np.unique(cluster_labels):
        if cluster_id == -1:
            continue

        # Check cluster size
        if cluster_size is not None:
            c_size = np.sum(cluster_labels == cluster_id)
            if c_size < cluster_size:
                continue

        filtered_hit_idx[cluster_labels == cluster_id] = True
        c += np.sum(cluster_labels == cluster_id)

        # Check for early termination
        # if n_particles is not None and c >= n_particles:
        #    break

    # Remaining positions
    rpos = filtered_coords[filtered_hit_idx]
    if n_particles is not None:
        rpos = rpos[0 : min(rpos.shape[0], n_particles), :]
        filtered_scores = filtered_scores[0 : min(rpos.shape[0], n_particles)]

    # Fill orientation and scores
    # Parse angle index
    ang_idx = angles_map[rpos[:, 0], rpos[:, 1], rpos[:, 2]].astype(int) - angles_numbering

    phi = anglist[ang_idx, 0]
    theta = anglist[ang_idx, 1]
    psi = anglist[ang_idx, 2]

    if symmetry > 1:
        add_phi = np.linspace(0, 360, symmetry + 1)
        add_phi = add_phi[:-1]
        phi = phi + np.random.choice(add_phi, size=phi.shape[0])

    ##### Generate motivelist #####
    print("Generating motivelist...")

    motl = cryomotl.Motl()
    motl.fill(
        {
            "x": rpos[:, 0] + 1,
            "y": rpos[:, 1] + 1,
            "z": rpos[:, 2] + 1,
            "score": filtered_scores,
            "class": 1,
            "tomo_id": tomo_id,
            "object_id": object_id,
            "phi": phi,
            "theta": theta,
            "psi": psi,
            "subtomo_id": np.arange(1, rpos.shape[0] + 1),
        }
    )

    del s_ind, scored_coords
    gc.collect()

    if output_path is not None:
        if output_type == "emmotl":
            motl.write_out(output_path)
        elif output_type == "stopgap":
            sg_motl = cryomotl.StopgapMotl(motl.df)
            sg_motl.write_out(output_path=output_path)
        elif output_type == "relion":
            rel_motl = cryomotl.RelionMotl(motl.df)
            rel_motl.write_out(output_path=output_path)
        else:
            raise ValueError(f"The output motl type {output_type} is not currently supported.")

    return motl


def compute_scores_map_threshold_triangle(scores_map):
    """Compute a threshold from a scores map using a triangle method on sorted values.

    The input array is flattened and sorted in ascending order.  A straight
    line is drawn from the first positive value to the maximum value.  The
    threshold is the sorted value whose perpendicular distance from that line
    is greatest — analogous to the triangle/chord method used for histogram
    thresholding, but applied directly to the empirical CDF of the scores
    rather than to binned counts.

    Parameters
    ----------
    scores_map : array_like
        Input array of any shape containing score or intensity values.  The
        array is flattened and sorted internally; the original shape is not
        modified.

    Returns
    -------
    float
        Threshold value.  Score values strictly above this threshold are
        considered signal in the calling code.

    Notes
    -----
    - Because the method operates on sorted values, the "peak" is always the
      maximum of the array (last element after sorting), not a histogram mode.
    - The flip branch (reversing the sorted array when the peak lies closer to
      the low end) is inherited from the histogram variant of the algorithm but
      is never triggered on sorted data where the maximum is by definition at
      the far right.  It is retained for completeness.
    - This implementation omits an additive constant present in some ImageJ
      versions; the omission does not affect the threshold location.
    """

    sp = np.sort(scores_map, axis=None)
    nbins = len(sp)

    # Find peak, lowest and highest gray levels.
    arg_peak_height = np.argmax(sp)
    peak_height = sp[arg_peak_height]
    arg_low_level, arg_high_level = np.where(sp > 0)[0][[0, -1]]

    # Flip is True if left tail is shorter.
    flip = arg_peak_height - arg_low_level < arg_high_level - arg_peak_height
    if flip:
        sp = sp[::-1]
        arg_low_level = nbins - arg_high_level - 1
        arg_peak_height = nbins - arg_peak_height - 1

    # If flip == True, arg_high_level becomes incorrect
    # but we don't need it anymore.
    del arg_high_level

    # Set up the coordinate system.
    width = arg_peak_height - arg_low_level
    x1 = np.arange(width)
    y1 = sp[x1 + arg_low_level]

    # Normalize.
    norm = np.sqrt(peak_height**2 + width**2)
    peak_height /= norm
    width /= norm

    # Maximize the length.
    # The ImageJ implementation includes an additional constant when calculating
    # the length, but here we omit it as it does not affect the location of the
    # minimum.
    length = peak_height * x1 - width * y1
    arg_level = np.argmax(length) + arg_low_level

    if flip:
        arg_level = nbins - arg_level - 1

    return sp[arg_level]


def create_starting_parameters_1D(input_map, peak_tolerance=20):
    """Locate the highest-scoring position within a central region and extract 1D profiles.

    A spherical mask of radius ``peak_tolerance`` is applied to restrict the
    peak search to the centre of the map.  The coordinates of the maximum
    within the masked region are returned together with three 1D intensity
    profiles along each array axis through that position.

    Parameters
    ----------
    input_map : ndarray
        3D array representing the map (e.g. a cross-correlation or density map).
    peak_tolerance : int, optional
        Radius of the spherical search mask in voxels.  Only the peak within
        this sphere around the map centre is considered.  Default is 20.

    Returns
    -------
    peak_center : tuple of int
        Array indices (dim0, dim1, dim2) of the detected peak, i.e. the
        highest-valued voxel inside the masked region.
    peak_height : float
        Global maximum of the unmasked ``input_map``.
    profiles : ndarray of shape (N, 3)
        1D intensity profiles through ``peak_center`` along each array axis.
        Column 0 varies along dim0 (x), column 1 along dim1 (y), column 2
        along dim2 (z).  N equals the length of the corresponding axis.

    Notes
    -----
    - The peak location is determined within the masked region, but
      ``peak_height`` is the global maximum of the full unmasked map.
    - The spherical mask is centred on the map centre, not on the peak.
    """

    peak_mask = cryomask.spherical_mask(np.asarray(input_map.shape), radius=peak_tolerance)
    masked_map = input_map * peak_mask
    peak_center = np.unravel_index(np.argmax(masked_map), shape=masked_map.shape)
    peak_height = np.amax(input_map)

    x_profile = input_map[:, peak_center[1], peak_center[2]]
    y_profile = input_map[peak_center[0], :, peak_center[2]]
    z_profile = input_map[peak_center[0], peak_center[1], :]

    profiles = np.vstack((x_profile, y_profile, z_profile))

    return peak_center, peak_height, profiles.T


def create_starting_parameters_2D(input_map, peak_tolerance=20, peak_center=None):
    """Extract three orthogonal 2D slices and peak parameters from a 3D map.

    Three planes — XY (fixed z), YZ (fixed x), and XZ (fixed y) — are
    extracted through the detected or supplied peak position and stacked into
    a single array.  Requires a cubic input volume (all three dimensions equal)
    so that the three 2D slices share the same shape.

    Parameters
    ----------
    input_map : array_like
        3D cubic array representing the map or volume.
    peak_tolerance : int, optional
        Radius in voxels of the spherical search mask used when ``peak_center``
        is not provided.  Default is 20.
    peak_center : tuple of int, optional
        Array indices (x, y, z) of the peak.  When supplied, the peak search is
        skipped and this position is used directly.  Defaults to None (automatic
        detection within the spherical mask).

    Returns
    -------
    peak_center : tuple of int
        Array indices (x, y, z) of the peak position used for slicing.
    peak_height : float
        When ``peak_center`` is None: global maximum of ``input_map``.
        When ``peak_center`` is provided: value of the *masked* map at that
        position, which is 0 if the supplied point lies outside the spherical
        mask.
    slices : ndarray of shape (N, N, 3)
        Three orthogonal slices stacked along the last axis:
        ``slices[:, :, 0]`` = XY plane (fixed z = peak_center[2]),
        ``slices[:, :, 1]`` = YZ plane (fixed x = peak_center[0]),
        ``slices[:, :, 2]`` = XZ plane (fixed y = peak_center[1]).
        N is the common edge length of the cubic volume.
    """

    peak_mask = cryomask.spherical_mask(np.asarray(input_map.shape), radius=peak_tolerance)
    masked_map = input_map * peak_mask
    if peak_center is None:
        peak_center = np.unravel_index(np.argmax(masked_map), shape=masked_map.shape)
        peak_height = np.amax(input_map)
    else:
        peak_height = masked_map[peak_center[0], peak_center[1], peak_center[2]]

    xy_plane = input_map[:, :, peak_center[2]]
    yz_plane = input_map[peak_center[0], :, :]
    xz_plane = input_map[:, peak_center[1], :]

    slices = np.stack((xy_plane, yz_plane, xz_plane), axis=2)

    return peak_center, peak_height, slices


def compute_gaussian_threshold(input_map):
    """Estimate the peak intensity of a 3D map by Gaussian fitting of 1D profiles.

    Three 1D profiles through the detected peak (one per array axis) are each
    fitted with a Gaussian model.  The mean of the three fitted peak heights
    (``amplitude / (sigma * sqrt(2π))``) is returned as the threshold estimate.

    Parameters
    ----------
    input_map : array_like
        3D array representing the map or volume to analyse.

    Returns
    -------
    threshold : float
        Mean Gaussian peak height averaged over the three orthogonal 1D profiles
        through the map's central peak.

    Notes
    -----
    - Peak detection uses :func:`create_starting_parameters_1D` with
      ``peak_tolerance=20``.
    - Gaussian fitting is performed with ``lmfit.models.GaussianModel``.
    - The Gaussian centre along each axis is constrained to ±1 voxel around
      the detected peak position.
    """

    pc, ph, profiles = create_starting_parameters_1D(input_map, peak_tolerance=20)

    heights = []
    for i in range(3):
        rt_line = profiles[:, i]
        x = np.linspace(0, rt_line.shape[0], rt_line.shape[0])
        y = rt_line
        mod = models.GaussianModel()

        # params = mod.make_params(center=24, sigma=0.5)
        params = mod.guess(rt_line, x)

        # you can place min/max bounds on parameters
        params["amplitude"].min = 0
        params["sigma"].min = 0
        params["center"].min = pc[i] - 1
        params["center"].max = pc[i] + 1

        # pars = mod.guess(y, x=x)
        out = mod.fit(y, params, x=x)

        heights.append(out.params["height"].value)

    return np.mean(np.asarray(heights))


def get_ellipsoid_label(input_map, peak_coordinates, map_threshold=0.0):
    """Fit an ellipsoid to the connected region around a peak and return the filled volume.

    The input map is binarised by treating voxels equal to ``map_threshold`` as
    background (label 2) and all others as foreground (label 1).  The connected
    component containing ``peak_coordinates`` is isolated, its surface is
    extracted with marching cubes, an ellipsoid is fitted to those surface
    vertices, and a filled ellipsoid mask is returned.

    Parameters
    ----------
    input_map : array_like
        3D array representing the volumetric map to segment.
    peak_coordinates : tuple of int
        Array indices (dim0, dim1, dim2) of a voxel inside the target region.
    map_threshold : float, optional
        Value used to distinguish background from foreground during
        binarisation.  Voxels with this exact value become background (label 2)
        and all others become foreground (label 1) before connected-component
        labelling.  Default is 0.0.

    Returns
    -------
    fitted_label : ndarray
        Binary 3D array (same shape as ``input_map``) filled inside the fitted
        ellipsoid (1) and zero outside.
    radii_sorted : ndarray of shape (3,)
        Full axis lengths (diameter, not radius) of the fitted ellipsoid,
        reordered so that element 0 corresponds to the spatial axis (x, y, or z)
        that the first principal axis most closely aligns with, element 1 to the
        second, and element 2 to the third.
    surface_fit : ndarray
        Binary 3D array marking the ellipsoid surface voxels obtained from
        marching cubes (value 1 on the surface, 0 elsewhere).
    th_map : ndarray
        Binary 3D array containing only the connected component that includes
        ``peak_coordinates`` (1 inside the component, 0 elsewhere).

    Notes
    -----
    - Connected-component labelling uses face connectivity (connectivity=1).
    - Ellipsoid fitting and volume filling are performed by
      :func:`cryocat.utils.geom.fit_ellipsoid` and
      :func:`cryocat.utils.geom.fill_ellipsoid`.
    """

    # shift the thresholding, otherwise only 1 label is found
    th_map = np.where(input_map == map_threshold, 2.0, 1.0)
    labeled_th_map = measure.label(th_map, connectivity=1)
    central_label = labeled_th_map[peak_coordinates[0], peak_coordinates[1], peak_coordinates[2]]
    th_map = np.where(labeled_th_map == central_label, 1.0, 0.0)

    ellipsoid_verts, _, _, _ = measure.marching_cubes(th_map, level=0.5)
    idx = np.round(ellipsoid_verts).astype(int)
    surface_fit = np.zeros(th_map.shape)
    surface_fit[idx[:, 0], idx[:, 1], idx[:, 2]] = 1.0

    _, radii, radii_dir, ell_params = geom.fit_ellipsoid(ellipsoid_verts)

    # dist = np.zeros(3,)
    # for i in range(3):
    #    dist[i] = np.linalg.norm(ellipsoid_verts[ellipsoid_verts[:, i].argsort()][-1,:]
    #                             - ellipsoid_verts[ellipsoid_verts[:, i].argsort()][0,:],
    #                             axis=0)

    # sorted_idx = np.argsort(dist)
    # radii_sorted = radii[sorted_idx]

    sorted_idx = np.argmax(np.abs(radii_dir), axis=0)
    radii_sorted = radii[sorted_idx] * 2.0

    fitted_label = geom.fill_ellipsoid(th_map.shape, ell_params)

    return fitted_label, radii_sorted, surface_fit, th_map


def get_central_plane_labels(input_map, peak_coordinates, map_threshold=0.0):
    """Label central-plane ellipses in a 3D map and estimate ellipsoid half-lengths.

    Three orthogonal planes (XY at fixed z, YZ at fixed x, XZ at fixed y)
    through ``peak_coordinates`` are extracted from the binarised map.  In each
    plane the connected component containing the peak is identified with
    ``skimage.measure.regionprops``, an ellipse is fitted to it, and the
    resulting filled ellipse mask is written back into the corresponding plane
    of the 3D output mask.

    .. note::
        Requires a cubic input volume (all three dimensions equal) because the
        YZ and XZ planes are stored in a buffer dimensioned to match the XY plane.

    Parameters
    ----------
    input_map : array_like
        3D cubic array to process.
    peak_coordinates : tuple of int
        Array indices (x, y, z) specifying a voxel inside the target region.
    map_threshold : float, optional
        Value used to separate background from foreground.  Voxels equal to
        this value become background (label 2) and all others become foreground
        (label 1) before connected-component labelling.  Default is 0.0.

    Returns
    -------
    label_mask : ndarray
        Binary 3D array (same shape as ``input_map``) that is 1 at voxels
        belonging to the three in-plane ellipse masks and 0 elsewhere.  Where
        planes overlap the mask is clipped to 1.
    ellipsoid_half_lengths : tuple of float
        Estimated half-lengths (x, y, z) of the ellipsoid.  Each half-length
        is the average of the two semi-axis measurements obtained from the two
        orthogonal planes that contain the corresponding spatial axis.

    Notes
    -----
    - Ellipse properties (major/minor axis lengths, orientation, centroid) are
      derived from ``skimage.measure.regionprops_table``.
    - ``skimage.draw.ellipse`` is used to rasterise (fill) the fitted ellipse
      into a binary mask; it does not perform fitting itself.
    - The accumulation of three plane masks into ``label_mask`` uses ``+=``
      followed by ``np.clip``, so overlapping regions are retained as 1.
    """

    # shift the thresholding, otherwise only 1 label is found
    th_map = np.where(input_map == map_threshold, 2.0, 1.0)
    # labeled_th_map = measure.label(th_map, connectivity = 1)
    # central_label = labeled_th_map[peak_coordinates[0],peak_coordinates[1],
    #                                peak_coordinates[2]]
    # th_map = np.where(labeled_th_map == central_label, 2.0, 1.0)

    # works only for cubic volumes!!!
    planes = np.zeros((input_map.shape[0], input_map.shape[1], 3))
    planes[:, :, 0] = th_map[:, :, peak_coordinates[2]]
    planes[:, :, 1] = th_map[peak_coordinates[0], :, :]
    planes[:, :, 2] = th_map[:, peak_coordinates[1], :]

    label_mask = np.zeros(input_map.shape)
    ellipse_masks = np.zeros(planes.shape)
    size_x, size_y, size_z = (0.0, 0.0, 0.0)

    for i in range(3):
        plane_label = measure.label(planes[:, :, i], connectivity=1)
        plane_props = pd.DataFrame(
            measure.regionprops_table(
                plane_label, properties=["label", "centroid", "axis_major_length", "axis_minor_length", "orientation"]
            )
        )

        if i < 2:
            central_label = plane_label[peak_coordinates[i], peak_coordinates[i + 1]]
        else:
            central_label = plane_label[peak_coordinates[0], peak_coordinates[i]]

        plane_props = plane_props[plane_props["label"] == central_label].reset_index()

        ellipse_indices = skimage.draw.ellipse(
            plane_props.at[0, "centroid-0"],
            plane_props.at[0, "centroid-1"],
            plane_props.at[0, "axis_major_length"] * 0.5,
            plane_props.at[0, "axis_minor_length"] * 0.5,
            rotation=plane_props.at[0, "orientation"],
        )
        ellipse_indices_x = np.clip(ellipse_indices[0], 0, input_map.shape[0] - 1)
        ellipse_indices_y = np.clip(ellipse_indices[1], 0, input_map.shape[1] - 1)

        if plane_props.at[0, "orientation"] < 0.0:
            major_axis = plane_props.at[0, "axis_major_length"]
            minor_axis = plane_props.at[0, "axis_minor_length"]
        else:
            major_axis = plane_props.at[0, "axis_minor_length"]
            minor_axis = plane_props.at[0, "axis_major_length"]

        ellipse_masks[ellipse_indices_x, ellipse_indices_y, i] = 1.0

        if i == 0:
            size_x += minor_axis
            size_y += major_axis
        elif i == 1:
            size_y += minor_axis
            size_z += major_axis
        else:
            size_x += minor_axis
            size_z += major_axis

    label_mask[:, :, peak_coordinates[2]] += ellipse_masks[:, :, 0]
    label_mask[peak_coordinates[0], :, :] += ellipse_masks[:, :, 1]
    label_mask[:, peak_coordinates[1], :] += ellipse_masks[:, :, 2]

    label_mask = np.clip(label_mask, 0.0, 1.0)
    ellipsoid_half_lengths = (size_x * 0.5, size_y * 0.5, size_z * 0.5)

    return label_mask, ellipsoid_half_lengths


def get_central_label(map, peak_coordinates):
    """Isolate the connected foreground region around a peak and measure its extent.

    Zero-valued voxels are treated as background (mapped to label 2) and
    non-zero voxels as foreground (mapped to label 1) before connected-component
    labelling.  The component that contains ``peak_coordinates`` is returned as
    a binary mask, and the span of that component along three 1D lines through
    the peak (one per axis) is reported as the size.

    Parameters
    ----------
    map : array_like
        3D array.  Zero values are treated as background; all other values as
        foreground.
    peak_coordinates : tuple of int
        Array indices (x, y, z) of a voxel inside the target connected region.

    Returns
    -------
    labeled_mask : ndarray
        Binary 3D array (same shape as ``map``) that is 1 inside the connected
        component containing ``peak_coordinates`` and 0 elsewhere.
    size_xyz : tuple of int
        Span of the labeled region along each axis measured on the 1D lines
        through the peak: (size_x, size_y, size_z).  Concretely, size_x is the
        number of consecutive labeled voxels along dim0 at
        (dim1=peak_y, dim2=peak_z), and analogously for y and z.

    Notes
    -----
    - Connectivity is face-based (connectivity=1).
    - Sizes reflect the extent along the axis *lines through the peak*, not the
      full bounding-box extent of the connected component.
    """

    # shift the thresholding, otherwise only 1 label is found
    th_map = np.where(map == 0.0, 2.0, 1.0)
    labeled_mask = measure.label(th_map, connectivity=1)
    central_label = labeled_mask[peak_coordinates[0], peak_coordinates[1], peak_coordinates[2]]
    labeled_mask = np.where(labeled_mask == central_label, 1.0, 0.0)

    profile_x = np.nonzero(labeled_mask[:, peak_coordinates[1], peak_coordinates[2]])[0]
    profile_y = np.nonzero(labeled_mask[peak_coordinates[0], :, peak_coordinates[2]])[0]
    profile_z = np.nonzero(labeled_mask[peak_coordinates[0], peak_coordinates[1], :])[0]
    size_x = profile_x[-1] - profile_x[0] + 1
    size_y = profile_y[-1] - profile_y[0] + 1
    size_z = profile_z[-1] - profile_z[0] + 1

    size_xyz = (size_x, size_y, size_z)
    return labeled_mask, size_xyz


def evaluate_scores_map(input_map, label_type="plane", threshold_type="gauss"):
    """Threshold a 3D scores map, label the central region, and return geometry estimates.

    The peak position is located with :func:`create_starting_parameters_2D`.  A
    threshold is computed according to ``threshold_type``, the map is binarised,
    and the selected labelling strategy extracts the peak region and estimates
    its dimensions.

    Parameters
    ----------
    input_map : array_like
        3D scores map to analyse.
    label_type : str, optional
        Strategy for labelling the thresholded region:

        - ``"ellipsoid"`` — fit an ellipsoid to the central connected component
          (calls :func:`get_ellipsoid_label`).
        - ``"plane"`` — extract three orthogonal central-plane ellipses
          (calls :func:`get_central_plane_labels`).
        - any other string — return the central connected component as-is
          (calls :func:`get_central_label`).

        Default is ``"plane"``.
    threshold_type : str, optional
        Thresholding method:

        - ``"gauss"`` — Gaussian peak height minus one standard deviation of
          the map.
        - ``"triangle"`` — triangle threshold plus one standard deviation of
          the map.
        - ``"hard"`` — half the detected peak height.

        Default is ``"gauss"``.

    Returns
    -------
    labeled_map : ndarray
        ``input_map`` multiplied element-wise by the binary label mask, i.e.
        the original scores retained only inside the labelled region.
    sizes : tuple of float
        Estimated region dimensions.  For ``"ellipsoid"`` and ``"plane"``:
        half-lengths along (x, y, z).  For other label types: full extents
        along the axis lines through the peak (see :func:`get_central_label`).
    peak_height : float
        Global maximum of ``input_map`` as returned by
        :func:`create_starting_parameters_2D`.
    thresholded_map : ndarray
        For ``"plane"`` and central-label types: ``input_map`` with all values
        at or below the threshold set to zero.  For ``"ellipsoid"``: the binary
        connected-component map returned by :func:`get_ellipsoid_label`
        (overwrites the threshold map).
    surface : ndarray or list
        For ``"ellipsoid"``: binary 3D array of the fitted ellipsoid surface.
        For all other label types: empty list.

    Raises
    ------
    ValueError
        If ``threshold_type`` is not one of ``"gauss"``, ``"triangle"``, or
        ``"hard"``.
    """

    pc, ph, slices = create_starting_parameters_2D(input_map)

    if threshold_type == "triangle":
        th = compute_scores_map_threshold_triangle(input_map)
        th = th + np.std(input_map)
    elif threshold_type == "gauss":
        th = compute_gaussian_threshold(input_map)
        th = th - np.std(input_map)
    elif threshold_type == "hard":
        th = ph / 2.0
    else:
        raise ValueError("Unknown type of threshold!")

    th_map = np.where(input_map > th, 1.0, 0.0)
    # th_map_close = binary_closing(th_map)
    th_map = input_map * th_map

    if label_type == "ellipsoid":
        labeled_mask, sizes, surface, th_map = get_ellipsoid_label(th_map, pc)
        labeled_map = labeled_mask * input_map
    elif label_type == "plane":
        labeled_mask, sizes = get_central_plane_labels(th_map, pc)
        labeled_map = labeled_mask * input_map
        surface = []
    else:
        labeled_mask, sizes = get_central_label(th_map, pc)
        labeled_map = labeled_mask * input_map
        surface = []

    return labeled_map, sizes, ph, th_map, surface


def filter_dist_maps(dist_maps, th_mask, min_angles_voxel_count):
    """Remove small connected regions from a stack of 3D distance maps.

    Each distance map in ``dist_maps`` is masked with ``th_mask``, connected
    regions are labelled, and any region with fewer than
    ``min_angles_voxel_count`` voxels is zeroed out.  Removing a small region
    from one map also removes the corresponding voxels from ``th_mask``,
    propagating the constraint to subsequent maps.  A second pass then applies
    the final mask to all maps for consistency.

    Parameters
    ----------
    dist_maps : ndarray of shape (X, Y, Z, N)
        Stack of N distance maps to filter.  Modified in place.
    th_mask : ndarray of shape (X, Y, Z)
        Binary mask.  Voxels that are zero here are excluded from labelling.
        The mask is progressively narrowed as small regions are removed.
    min_angles_voxel_count : int
        Minimum connected-region size (in voxels) to retain.

    Returns
    -------
    dist_maps : ndarray
        Filtered distance maps (same object as input, modified in place).
    th_mask : ndarray
        Updated binary mask after removal of all small regions across all maps.

    Notes
    -----
    - Connected-component labelling uses face connectivity (connectivity=1).
    - Because the mask is updated iteratively during the first pass, a region
      that was borderline-large in map *i* may be eliminated when map *i+1* is
      processed.  The second pass re-applies the final mask uniformly.
    """

    for j in range(dist_maps.shape[3]):
        dist_maps[:, :, :, j] *= th_mask
        dist_label = measure.label(dist_maps[:, :, :, j], connectivity=1)
        dist_props = pd.DataFrame(measure.regionprops_table(dist_label, properties=("label", "area")))
        too_small_dist = dist_props.loc[dist_props["area"] < min_angles_voxel_count, "label"].values
        th_mask = np.where(np.isin(dist_label, too_small_dist), 0.0, dist_label)
        th_mask = np.where(th_mask > 0.0, 1.0, 0.0)
        dist_maps[:, :, :, j] *= th_mask

    for j in range(dist_maps.shape[3]):
        dist_maps[:, :, :, j] *= th_mask

    return dist_maps, th_mask


def create_angular_distance_maps(
    angles_map, angles_list, output_file_base=None, write_out_maps=True, c_symmetry=1, angles_order="zxz"
):
    """Compute per-voxel angular distance maps relative to the first entry in the angles list.

    Each voxel of ``angles_map`` stores a 1-based index into ``angles_list``.
    For every orientation in the list the total angular distance, the distance
    of the rotation axis (normals), and the in-plane rotation distance relative
    to ``angles_list[0]`` are computed.  These distances are then mapped back
    onto the volume via the index map to produce three 3D distance volumes.

    Parameters
    ----------
    angles_map : str or ndarray
        Path to the orientation-index map or a pre-loaded integer array.
        Values are 1-based indices (subtracted by 1 internally before lookup).
    angles_list : str or ndarray
        Path to the rotation-angles file or a pre-loaded (N, 3) array of
        Euler angles.  The angle convention is given by ``angles_order``.
    output_file_base : str, optional
        Base path (without extension) for the three output ``.em`` files.
        When None and ``angles_map`` is a file path, the path without the
        last three characters is used.  When None and ``angles_map`` is an
        array, ``write_out_maps`` is silently set to False.
    write_out_maps : bool, default=True
        Whether to save the three distance maps to disk as ``.em`` files
        (``*_dist_all.em``, ``*_dist_normals.em``, ``*_dist_inplane.em``)
        in single precision.
    c_symmetry : int, default=1
        Cyclic symmetry order passed to :func:`cryocat.utils.geom.compare_rotations`
        when computing angular distances.
    angles_order : str, default='zxz'
        Euler-angle convention used in ``angles_list``.

    Returns
    -------
    ang_dist_map : ndarray
        3D map of total angular distances from ``angles_list[0]``.
    dist_normals_map : ndarray
        3D map of rotation-axis (normal) distances from ``angles_list[0]``.
    dist_inplane_map : ndarray
        3D map of in-plane rotation distances from ``angles_list[0]``.

    Notes
    -----
    - All distances are measured relative to the *first* angle in
      ``angles_list`` (index 0 after the 1-based offset is removed), not
      relative to a zero rotation.
    - If ``output_file_base`` cannot be determined and ``write_out_maps`` is
      True, writing is silently disabled (the ``ValueError`` in the code is
      constructed but not raised; this is a known limitation).
    """

    if output_file_base is None:
        if isinstance(angles_map, str):
            output_file_base = angles_map[:-3]
        elif write_out_maps:
            ValueError("The output_file_base was not specified -> " "the maps will not be written out!")
            write_out_maps = False

    angles_map = cryomap.read(angles_map).astype(int)

    map_shape = angles_map.shape
    angles = ioutils.rot_angles_load(angles_list, angles_order)

    zero_rotations = np.tile(angles[0, :], (angles.shape[0], 1))
    dist_all, dist_normals, dist_inplane = geom.compare_rotations(zero_rotations, angles, c_symmetry)

    angles_array = angles_map.flatten() - 1

    ang_dist_map = dist_all[angles_array].reshape(map_shape)
    dist_normals_map = dist_normals[angles_array].reshape(map_shape)
    dist_inplane_map = dist_inplane[angles_array].reshape(map_shape)

    if write_out_maps:
        cryomap.write(ang_dist_map, output_file_base + "_dist_all.em", data_type=np.single)
        cryomap.write(dist_normals_map, output_file_base + "_dist_normals.em", data_type=np.single)
        cryomap.write(dist_inplane_map, output_file_base + "_dist_inplane.em", data_type=np.single)

    return ang_dist_map, dist_normals_map, dist_inplane_map


def select_peaks(
    scores_map,
    angles_map,
    angles_file,
    peak_number=None,
    create_dist_maps=False,
    dist_maps_list=["_dist_all", "_dist_normals", "_dist_inplane"],
    dist_maps_name_base=None,
    write_dist_maps=False,
    min_peak_voxel_count=7,
    min_angles_voxel_count=7,
    template_mask=None,
    template_radius=2,
    edge_masking=None,
    tomo_mask=None,
    output_motl_name=None,
    tomo_number=None,
    angles_order="zxz",
):
    """Select peaks from a template-matching scores map using angular-distance constraints.

    The algorithm:

    1. Thresholds the scores map with the triangle method and optionally
       applies a tomogram mask and edge mask.
    2. Removes thresholded regions that are smaller than
       ``min_peak_voxel_count`` voxels.
    3. Loads or computes angular-distance maps and filters out small connected
       regions from them (``min_angles_voxel_count``).
    4. Iterates over candidate voxels in descending score order.  A candidate
       is accepted if (a) its particle mask does not overlap with any already-
       accepted particle, (b) the thresholded scores region inside the mask
       meets ``min_peak_voxel_count``, and (c) every distance map also has a
       connected region of at least ``min_angles_voxel_count`` voxels at that
       position.  Accepted particles are stamped into an occupancy volume and
       their footprint removed from subsequent candidate checks.

    Parameters
    ----------
    scores_map : str or ndarray
        Path to the CCC scores map or a pre-loaded array.
    angles_map : str or ndarray
        Path to the angle-index map or a pre-loaded array.  Values are 1-based
        indices into ``angles_file`` (converted to 0-based internally).
    angles_file : str or ndarray
        Path to the rotation-angles file or a pre-loaded (N, 3) array in
        (phi, theta, psi) order.
    peak_number : int, optional
        Maximum number of peaks to select.  Defaults to None (select all
        passing candidates).
    create_dist_maps : bool, optional
        When True, angular-distance maps are computed on the fly from
        ``angles_map`` and ``angles_file``.  When False, they are read from
        disk using ``dist_maps_name_base``.  Defaults to False.
    dist_maps_list : list of str, optional
        Subset of distance maps to use.  Each entry must be one of
        ``"_dist_all"``, ``"_dist_normals"``, ``"_dist_inplane"``.
        Defaults to all three.
    dist_maps_name_base : str, optional
        Base path for the distance map files (without the suffix and
        ``.em`` extension).  Required when ``create_dist_maps`` is False.
        Defaults to None.
    write_dist_maps : bool, optional
        Whether to save freshly computed distance maps to disk.  Only
        relevant when ``create_dist_maps`` is True.  Defaults to False.
    min_peak_voxel_count : int, optional
        Minimum connected-region size (voxels) in the thresholded scores map
        required for a candidate to be accepted.  Defaults to 7.
    min_angles_voxel_count : int, optional
        Minimum connected-region size (voxels) in each distance map required
        for a candidate to be accepted.  Defaults to 7.
    template_mask : str or ndarray, optional
        Path to a particle mask or a pre-loaded array.  Used to define the
        volume carved out around each accepted peak.  Should be a solid mask
        (no hollow interiors).  When None, a spherical mask of radius
        ``template_radius`` is used.  Defaults to None.
    template_radius : int, optional
        Radius in voxels of the fallback spherical particle mask, used only
        when ``template_mask`` is None.  Defaults to 2.
    edge_masking : int or ndarray of shape (3,), optional
        Width in voxels of the border to exclude on each side of the
        tomogram.  A single integer applies the same width to all three axes;
        a length-3 array specifies per-axis widths.  Defaults to None (no
        edge masking).
    tomo_mask : ndarray, optional
        Binary mask with the same shape as the scores map.  Zero-valued
        regions are excluded from peak search.  Defaults to None.
    output_motl_name : str, optional
        Path to write the output motl (EM format).  No file is written when
        None.  Defaults to None.
    tomo_number : int, optional
        Tomogram identifier stored in ``tomo_id`` column of the output motl.
        Defaults to None.
    angles_order : str, optional
        Euler-angle convention of ``angles_file``.  Defaults to ``"zxz"``.

    Returns
    -------
    output_motl : Motl
        :class:`cryocat.core.cryomotl.Motl` containing the selected peak
        coordinates, orientations, and scores.
    empty_label : ndarray
        Occupancy volume of the same shape as the scores map.  Each accepted
        particle contributes its (possibly rotated) particle mask to this
        array via accumulation; values ≥ 1 indicate occupied voxels.

    Raises
    ------
    ValueError
        If ``edge_masking`` is neither a single integer nor an array of
        shape (3,).
    ValueError
        If an entry in ``dist_maps_list`` is not one of the recognised
        suffixes.
    ValueError
        If ``create_dist_maps`` is False and ``dist_maps_name_base`` is None.
    """

    # load the angles
    angles = ioutils.rot_angles_load(angles_file, angles_order=angles_order)
    angles_map = (cryomap.read(angles_map) - 1).astype(int)

    # get threshold and threshold map
    scores_map = cryomap.read(scores_map)
    th = compute_scores_map_threshold_triangle(scores_map)
    th_map = np.where(scores_map >= th, 1.0, 0.0)

    if tomo_mask is not None:
        th_map *= tomo_mask

    if edge_masking is not None:
        edge_mask = np.zeros(th_map.shape)

        if isinstance(edge_masking, int):
            edge_masking = np.full((3,), edge_masking)
        elif edge_masking.shape[0] != 3:
            raise ValueError("The edge mask has to be single number or 3 numbers - " "one for each dimension.")
        edge_mask[
            edge_masking[0] : -edge_masking[0], edge_masking[1] : -edge_masking[1], edge_masking[2] : -edge_masking[2]
        ] = 1
        th_map *= edge_mask

    n_dist_maps = len(dist_maps_list)
    dist_maps = np.zeros((th_map.shape[0], th_map.shape[1], th_map.shape[2], n_dist_maps))

    if create_dist_maps:
        temp_dist_maps = create_angular_distance_maps(
            angles_map, angles_file, output_file_base=dist_maps_name_base, write_out_maps=write_dist_maps
        )
        for j, d_name in enumerate(dist_maps_list):
            if d_name == "_dist_all":
                dist_maps[:, :, :, j] = temp_dist_maps[0]
            elif d_name == "_dist_normals":
                dist_maps[:, :, :, j] = temp_dist_maps[1]
            elif d_name == "_dist_inplane":
                dist_maps[:, :, :, j] = temp_dist_maps[2]
            else:
                raise ValueError("The dist_maps_list contains unknown dist map " f"specifier: {d_name}!")
    elif dist_maps_name_base is None:
        raise ValueError("The dist_maps_name_base was not specified!")
    else:
        for j, d_name in enumerate(dist_maps_list):
            dist_maps[:, :, :, j] = cryomap.read(dist_maps_name_base + d_name + ".em")

    th_map_d = th_map

    labels = measure.label(th_map, connectivity=1)
    props = pd.DataFrame(measure.regionprops_table(labels, properties=("label", "area")))

    too_small_peaks = props.loc[props["area"] < min_peak_voxel_count, "label"].values
    th_map = np.where(np.isin(labels, too_small_peaks), 0.0, labels)
    th_map = np.where(th_map > 0.0, 1.0, 0.0)

    dist_maps, th_map_d = filter_dist_maps(dist_maps, th_map_d, min_angles_voxel_count)

    for j in range(n_dist_maps):
        dist_temp = np.zeros(th_map.shape)
        dist_label = measure.label(dist_maps[:, :, :, j], connectivity=1)
        dist_props = pd.DataFrame(measure.regionprops_table(dist_label, properties=("label", "bbox")))
        labels, xs, xe, ys, ye, zs, ze = dist_props[
            ["label", "bbox-0", "bbox-3", "bbox-1", "bbox-4", "bbox-2", "bbox-5"]
        ].T.to_numpy()
        for l in range(labels.shape[0]):
            label_cut = dist_label[xs[l] : xe[l], ys[l] : ye[l], zs[l] : ze[l]]
            label_cut = np.where(label_cut == labels[l], 1.0, 0.0)
            label_open = morphology.binary_opening(label_cut, footprint=np.ones((2, 2, 2)), out=None)
            dist_temp[xs[l] : xe[l], ys[l] : ye[l], zs[l] : ze[l]] = np.where(
                label_open == 1, dist_maps[xs[l] : xe[l], ys[l] : ye[l], zs[l] : ze[l], j], 0.0
            )
        dist_maps[:, :, :, j] = dist_temp

    dist_maps, th_map_d = filter_dist_maps(dist_maps, th_map_d, min_angles_voxel_count)

    th_map *= th_map_d

    scores_th = np.ndarray.flatten(scores_map * th_map)
    nz_idx = np.flatnonzero(scores_th)
    remaining_idx = nz_idx[np.argsort(scores_th[nz_idx], axis=None)][::-1]
    selected_peaks = []
    n_selected_peaks = 0

    if template_mask is None:
        particle_mask = cryomask.spherical_mask(2 * template_radius + 2, radius=template_radius)
    else:
        particle_mask = cryomap.read(template_mask)

    if peak_number is None:
        peak_number = remaining_idx.size

    c_idx = 0

    empty_label = np.zeros(th_map.shape)
    removed_idx = []

    c_coord = (np.ceil(np.asarray(particle_mask.shape) / 2)).astype(int)

    while n_selected_peaks < peak_number and remaining_idx.size != 0:
        idx_3d = np.unravel_index(remaining_idx[c_idx], th_map.shape)
        ls, le, ms, me = cryomap.get_start_end_indices(idx_3d, empty_label.shape, particle_mask.shape)
        cut_coord = c_coord - ms

        if template_mask is not None:
            p_particle = cryomap.rotate(
                particle_mask, rotation_angles=angles[angles_map[idx_3d[0], idx_3d[1], idx_3d[2]]]
            )
            p_particle = np.where(p_particle >= 0.5, 1.0, 0.0)
            p_particle = p_particle[ms[0] : me[0], ms[1] : me[1], ms[2] : me[2]]
        else:
            p_particle = particle_mask[ms[0] : me[0], ms[1] : me[1], ms[2] : me[2]]

        overlap_voxels = np.count_nonzero(empty_label[ls[0] : le[0], ls[1] : le[1], ls[2] : le[2]] * p_particle)

        if overlap_voxels == 0 and np.all(cut_coord < me):
            th_label = measure.label(th_map[ls[0] : le[0], ls[1] : le[1], ls[2] : le[2]] * p_particle)
            th_label_id = th_label[cut_coord[0], cut_coord[1], cut_coord[2]]

            if th_label_id == 0:
                peak_area = 0
                angle_size = 0
                print(idx_3d)
            else:
                peak_area = np.count_nonzero(np.where(th_label == th_label_id, 1.0, 0.0))
                angle_size = min_angles_voxel_count
                for j in range(n_dist_maps):
                    dist_label = measure.label(dist_maps[ls[0] : le[0], ls[1] : le[1], ls[2] : le[2], j] * p_particle)
                    dist_label_id = dist_label[cut_coord[0], cut_coord[1], cut_coord[2]]
                    if dist_label_id == 0:
                        angle_size = 0
                        print(idx_3d)
                        break
                    else:
                        ## Add opening
                        label_open = np.where(dist_label == dist_label_id, 1.0, 0.0)
                        # label_open = morphology.binary_opening(label_open,
                        #                                        footprint=np.ones((2,2,2)),
                        #                                        out=None)
                        # label_open = measure.label(label_open)
                        # open_label_id = label_open[cut_coord[0],cut_coord[1],cut_coord[2]]
                        # label_open = np.where(label_open==open_label_id,1.0,0.0)
                        # if open_label_id == 0:
                        #    angle_size = 0
                        #    break
                        angle_size = np.minimum(angle_size, np.count_nonzero(label_open))
                        if angle_size < min_angles_voxel_count:
                            break

            if angle_size >= min_angles_voxel_count and peak_area >= min_peak_voxel_count:
                empty_label[ls[0] : le[0], ls[1] : le[1], ls[2] : le[2]] += p_particle
                th_map[ls[0] : le[0], ls[1] : le[1], ls[2] : le[2]] = np.where(
                    p_particle == 1, 0.0, th_map[ls[0] : le[0], ls[1] : le[1], ls[2] : le[2]]
                )
                for j in range(n_dist_maps):
                    dist_maps[ls[0] : le[0], ls[1] : le[1], ls[2] : le[2], j] = np.where(
                        p_particle == 1, 0.0, dist_maps[ls[0] : le[0], ls[1] : le[1], ls[2] : le[2], j]
                    )

                selected_peaks.append(
                    (
                        idx_3d,
                        angles[angles_map[idx_3d[0], idx_3d[1], idx_3d[2]]],
                        scores_map[idx_3d[0], idx_3d[1], idx_3d[2]],
                    )
                )
                n_selected_peaks += 1
                non_zero = np.flatnonzero(empty_label)
                remaining_idx = np.setdiff1d(remaining_idx, non_zero, assume_unique=True)
                removed_idx = []
                c_idx = 0
            else:
                removed_idx.append(remaining_idx[c_idx])
                c_idx += 1

        else:
            removed_idx.append(remaining_idx[c_idx])
            c_idx += 1

        if c_idx == remaining_idx.size:
            remaining_idx = np.setdiff1d(remaining_idx, np.asarray(removed_idx), assume_unique=True)
            removed_idx = []
            c_idx = 0

    motl_df = cryomotl.Motl.create_empty_motl_df()
    dim, angles, score = zip(*selected_peaks)
    motl_df[["x", "y", "z"]] = np.array(dim)
    motl_df[["phi", "theta", "psi"]] = np.array(angles)
    motl_df["score"] = score
    motl_df = motl_df.fillna(0)

    if tomo_number is not None:
        motl_df["tomo_id"] = tomo_number

    motl_df["subtomo_id"] = range(1, len(selected_peaks) + 1)
    motl_df["class"] = 1

    output_motl = cryomotl.Motl(motl_df)

    if output_motl_name is not None:
        output_motl.write_to_emfile(output_motl_name)

    print(f"Number of selected peaks: {output_motl.df.shape[0]}")

    return output_motl, empty_label
