import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
from scipy.spatial import KDTree
from cryocat import tgeometry as tg
from cryocat import surfsamp
from cryocat import geom
import scipy.ndimage as ndimage
import skimage
from skimage.segmentation import watershed
from skimage.morphology import convex_hull_image


# shape_layer is shape data layer, e.g. viewer.layers[x].data where z is layer id
def get_oversampling(shape_layer, sampling_distance):
    """Create oversample points from the convex hull of a point cloud representing your surface.

    Parameters
    ----------
    shape_layer : list
        A list containing point cloud coordinates ordered by z, y, x, with points sharing the same z-value organized into arrays.
    sampling_distance : int
        Distance between sample points in pixels.

    Returns
    -------
    ndarray
        A numpy array of point cloud coordinates and normals.

    Raises
    ------
    TypeError
        If shape_layer is not a list.
    """
    if isinstance(shape_layer, list):
        # create array from the list
        mask_points = np.concatenate(shape_layer, axis=0)
    else:
        mask_points = shape_layer

    # get convex hull
    hull = ConvexHull(mask_points)

    tri_points = []
    normals = []

    for i in range(len(hull.simplices)):
        tp = hull.points[hull.simplices[i, :]]
        mp = tg.get_mesh(tp, sampling_distance)
        n_tp = hull.equations[i][0:3]

        for p in mp:
            if tg.point_inside_triangle(p, tp):
                tri_points.append(p)
                normals.append(n_tp)

    tri_points = np.array(tri_points)
    normals = np.array(normals)

    return tri_points, normals


# save shape layer into text file
def save_shapes(shape_layer, output_file_name):
    """This function saves the shape layer data into a csv file.

    Parameters
    ----------
    shape_layer : list
        A list of numpy arrays where each array represents a shape layer. Each array is expected to have three columns representing the x, y, and z coordinates respectively.
    output_file_name : str
        The name of the output file where the shape layer data will be saved.

    Returns
    -------
    None

    Notes
    -----
    The function creates a pandas DataFrame from the shape layer data and saves it into a csv file. The DataFrame has four columns: 's_id', 'x', 'y', and 'z'. The 's_id' column represents the shape layer id, 'x', 'y', and 'z' columns represent the coordinates of the shape layer.
    """
    x = []
    y = []
    z = []
    s_id = []

    for i in range(len(shape_layer)):
        x.append(shape_layer[i][:, 2])
        y.append(shape_layer[i][:, 1])
        z.append(shape_layer[i][:, 0])
        s_id.append(np.full(shape_layer[i].shape[0], i))

    x = np.concatenate(x, axis=0)
    y = np.concatenate(y, axis=0)
    z = np.concatenate(z, axis=0)
    s_id = np.concatenate(s_id, axis=0)

    # dictionary of lists
    dict = {"s_id": s_id, "x": x, "y": y, "z": z}

    df = pd.DataFrame(dict)
    # saving the dataframe
    df.to_csv(output_file_name, index=False)


def load_shapes(input_file_name):
    """load_shapes(input_file_name: str) -> List[np.array]:

    This function loads a surface point cloud from a csv file. The point cloud coordinates are ordered by z, y, x.
    Points sharing the same z-value are organized into arrays.

    Parameters:
    -----------
    input_file_name : str
        The path to your csv point cloud file.

    Returns:
    --------
    shape_list : list
        A list containing point cloud coordinates ordered by z, y, x, with points sharing the same z-value organized into arrays.
    """
    df = pd.read_csv(input_file_name)
    s_id = df["s_id"].unique()

    shape_list = []

    for sf in s_id:
        a = df.loc[df["s_id"] == sf]
        sp_array = np.zeros((a.shape[0], 3))
        sp_array[:, 2] = (a["x"]).values
        sp_array[:, 1] = (a["y"]).values
        sp_array[:, 0] = (a["z"]).values
        shape_list.append(sp_array)

    return shape_list


def save_shapes_as_point_cloud(shape_layer, output_file_name):
    if isinstance(shape_layer, list):
        point_cloud = np.concatenate(shape_layer, axis=0)
    else:
        point_cloud = shape_layer

    np.savetxt(output_file_name, point_cloud, fmt="%5.6f")


def load_shapes_as_point_cloud(input_file_name):
    point_cloud = np.loadtxt(input_file_name)
    return point_cloud


# visualize points
def visualize_points(viewer, points, p_size):
    viewer.add_points(points, size=p_size)


# visualize normals
def visualize_normals(viewer, points, normals):
    """This function visualizes the normals of a given set of points.

    Parameters
    ----------
    viewer : object
        The viewer object where the vectors will be added.
    points : ndarray
        An array of points in 3D space.
    normals : ndarray
        An array of normal vectors corresponding to the points.

    Returns
    -------
    None

    Notes
    -----
    The function modifies the viewer object by adding vectors representing the normals.
    The vectors are represented as lines with a specified edge width and length.
    """
    vectors = np.zeros((points.shape[0], 2, 3), dtype=np.float32)
    vectors[:, 0] = points
    vectors[:, 1] = normals
    viewer.add_vectors(vectors, edge_width=1, length=10)


def expand_points(points, normals, distances, tb_distances=None):
    """Moves sample points within a specified distance in its normal direction.

    Parameters
    ----------
    points : ndarray
        Array of sample coordinates.
    normals : ndarray
        Array of sample normals.
    distances : int
        Moving distance in pixels.
    tb_distances : int, optional
        Moving distance of top and bottom surfaces if needed. Defaults to None for similar movements as the other surfaces.

    Returns
    -------
    ndarray
        Array of moved sample coordinates.
    ndarray
        Array of moved sample normals.
    """
    moved_points = []
    tb_points = []
    if tb_distances == None:
        tb_distances = distances
    for i in range(len(normals)):
        if abs(normals[i, 0]) == 1 and normals[i, 1] == 0 and normals[i, 2] == 0:
            tb_points.append(i)
        else:
            moved_points.append(i)

    post_points = points.copy()
    if tb_distances == 0:  # 0 for nonmove top and bottom
        post_points[moved_points] = (
            post_points[moved_points] + distances * normals[moved_points]
        )
    elif tb_distances != 0:  # other value for move top and bottom specific distance
        post_points[moved_points] = (
            post_points[moved_points] + distances * normals[moved_points]
        )
        post_points[tb_points] = (
            post_points[tb_points] + tb_distances * normals[tb_points]
        )

    if tb_points == []:
        return post_points, normals
    else:
        # removing the distinct points after shifting
        drop_lim = [
            max(post_points[tb_points][:, 0]),
            min(post_points[tb_points][:, 0]),
        ]
        drop_list = [
            i
            for i in range(len(post_points))
            if post_points[i, 0] > drop_lim[0] or post_points[i, 0] < drop_lim[1]
        ]
        cleaned_points = np.delete(post_points, drop_list, axis=0)
        cleaned_normals = np.delete(normals, drop_list, axis=0)
        return cleaned_points, cleaned_normals


# remove sample points with specific normals value
# TODO more precise removing method
def rm_points(points, normals, rm_surface):
    """Removes sample points based on their normal direction. Only considers cases for top and bottom points.

    Parameters
    ----------
    points : ndarray
        Array of sample coordinates.
    normals : ndarray
        Array of sample normals.
    rm_surface : int
        Specifies the faces for removal: assign 1 for the top face, -1 for the bottom face, and 0 for both.
        The face with the lower z value represents the bottom face, while the one with the higher z value signifies the top face.

    Returns
    -------
    cleaned_points : ndarray
        Array of cleaned sample coordinates.
    cleaned_normals : ndarray
        Array of cleaned sample normals.
    """
    removed_points = []
    if rm_surface == 1:
        tob = 1
    elif rm_surface == 0:
        tob = 1
        normals[:, 0] = np.absolute(normals[:, 0])
    elif rm_surface == -1:
        tob = -1

    for i in range(len(normals)):
        if normals[i, 0] == tob and normals[i, 1] == 0 and normals[i, 2] == 0:
            removed_points.append(i)

    cleaned_points = np.delete(points, removed_points, axis=0)
    cleaned_normals = np.delete(normals, removed_points, axis=0)

    return cleaned_points, cleaned_normals


# replace the normal with the closest postion from convexHull
# Substitute the normal in one motl with the adjacent normal from oversampling points.
# samp_dist : int Distance between sample points in pixels.
def reset_normals(
    shapes_data, motl, samp_dist=1, shift_dist=None, tb_dist=0, bin_factor=None
):
    """Replaces the angle of the motl with the angle from adjacent oversampling points.

    Parameters
    ----------
    shapes_data : list
        A list containing point cloud coordinates ordered by z, y, x, with points sharing the same z-value organized into arrays.
    motl : object
        The motl for resetting angles.
    samp_dist : int
        Distance between sample points in pixels for resetting angles. Defaults to 1.
    shift_dist : int
        Shifting distance for resetting angles. Defaults to None for no shifting.
    tb_dist : int
        Shifting distance for moving the points at top and bottom surface. Defaults to 0 for no top and bottom movement.
    bin_factor : int
        Bin_factor for shape_data. Defaults to None for no binning differents between shape and motl.

    Returns
    -------
    motl : pddataframe
        dataframe of motl list with replaced euler angles

    Notes
    -----
    This function gets the oversampling, shifts points to the normal direction, reorganizes x,y,z into z,y,x to match with tri_points, adds binning to tri_points, searches for the closest point to motl_points, replaces normals in motlist to new normals, gets Euler angles from coordinates, creates pandas from angles, and replaces angles.
    """
    # get the oversampling
    tri_points, tri_normals = surfsamp.get_oversampling(shapes_data, samp_dist)
    # shifting points to the normal direction
    if shift_dist != None:
        tri_points, tri_normals = surfsamp.expand_points(
            tri_points, tri_normals, shift_dist, tb_dist
        )  # negative for shifting in opposite direction

    # reorganize x,y,z into z,y,x to match with tri_points
    # flip here rather than oversampling because oversampling will be show in napari which needed to be order in z, y, x
    motl_points = np.flip(motl.get_coordinates(), axis=1)
    if bin_factor != None:
        sample_points = tri_points * bin_factor  # add binning to tri_points
    else:
        sample_points = tri_points
    # searching for the closest point to motl_points
    kdtree = KDTree(sample_points)
    _, points = kdtree.query(motl_points, 1)

    ## replacing eular angles in motlist to new angles
    n_normals = tri_normals[points]
    # create panda frames from normals
    pd_normals = pd.DataFrame(
        {"x": n_normals[:, 2], "y": n_normals[:, 1], "z": n_normals[:, 0]}
    )
    # get Euler angles from normals
    phi, psi, theta = tg.normals_to_euler_angles(pd_normals)
    # replace angles in motl
    motl.fill(
        {
            "angles": pd.DataFrame(
                {
                    "phi": phi,
                    "psi": psi,
                    "theta": theta,
                }
            )
        }
    )

    return motl


def get_sampling_pandas(
    shapes_data, overSample_dist, shift_dist=None, tb_dist=0, rm_surface=None
):
    """This function generates a pandas dataframe of point cloud coordinates and euler angles from given shape data.

    Parameters
    ----------
    shapes_data : list
        A list containing point cloud coordinates ordered by z, y, x, with points sharing the same z-value organized into arrays.
    overSample_dist : int
        Distance between sample points in pixels.
    shift_dist : int, optional
        Shifting distance of all points in pixels. Negative for shifting in opposite direction. Defaults to None.
    tb_dist : int, optional
        Shifting distance of top and bottom points. Defaults to 0.
    rm_surface : int, optional
        Set to 1 to remove top and bottom points. Defaults to None.

    Returns
    -------
    pddataframe
        Dataframe of point cloud coordinates and euler angles.
    """
    # get the oversampling
    tri_points, tri_normals = surfsamp.get_oversampling(shapes_data, overSample_dist)
    # shifted points to the normal direction
    if shift_dist != None:
        tri_points, tri_normals = surfsamp.expand_points(
            tri_points, tri_normals, shift_dist, tb_dist
        )  # negative for shifting in opposite direction, set 4 argument to 0 to omit moving of top and bottom layer.
    # remove unecessary points base on condition
    if rm_surface != None:
        tri_points, tri_normals = surfsamp.rm_points(
            tri_points, tri_normals, rm_surface
        )
    # create panda frames from points
    pd_points = pd.DataFrame(
        {"x": tri_points[:, 2], "y": tri_points[:, 1], "z": tri_points[:, 0]}
    )
    # create panda frames from normals
    pd_normals = pd.DataFrame(
        {"x": tri_normals[:, 2], "y": tri_normals[:, 1], "z": tri_normals[:, 0]}
    )
    # get Euler angles from coordinates
    phi, psi, theta = tg.normals_to_euler_angles(pd_normals)
    # create pandas
    pd_angles = pd.DataFrame({"phi": phi, "psi": psi, "theta": theta})

    return pd_points, pd_angles


def get_surface_area_from_hull(mask_points, rm_faces=0):
    # consider adding the mask surface area here as well
    """Calculates the area of the convex hull surface for a point cloud.

    Args:
        mask_points (numpy.ndarray): An array of points representing the surface.
        rm_faces (int): An integer representing the faces to be removed. Can be 0, 1, or -1. Defaults to 0 for not remove anything. 1 for removing face with larger z axis, -1 for removing face with smaller z.

    Raises:
        ValueError: If the target top/bottom surfaces do not exist.

    Returns:
        float: The updated area of the convex hull after removing the specified faces.
    """
    # get convexhull
    hull = ConvexHull(mask_points)
    faces = hull.equations

    # find indices of surface that were belongs to top or bottom
    if rm_faces == 0:
        updated_area = hull.area
    else:
        if rm_faces == 1:
            tb_faces = [
                i for i, num in enumerate(faces) if sum(num[0:3] == [1, 0, 0]) == 3
            ]
        elif rm_faces == -1:
            tb_faces = [
                i for i, num in enumerate(faces) if sum(num[0:3] == [-1, 0, 0]) == 3
            ]
        elif rm_faces == 2:
            tb_faces = [
                i for i, num in enumerate(faces) if sum(abs(num[0:3]) == [1, 0, 0]) == 3
            ]
        if tb_faces == []:
            raise ValueError("The target top/bottom surfaces doesn't exist")
        all_faces_points_in = hull.simplices  # indices of points for surface
        tb_faces_points_in = all_faces_points_in[tb_faces]
        face_points_coord = [[mask_points[j] for j in i] for i in tb_faces_points_in]
        face_points_array = np.asarray(face_points_coord)
        tb_areas = geom.area_triangle(face_points_array)
        total_tb_area = sum(tb_areas)
        updated_area = hull.area - total_tb_area

    return updated_area


def angles_clean(
    motl, vertices, normals, angle_threshold=90, normal_vector=[0, 0, 1], keep_top=None
):
    """This function cleans the given list of motl based on the angle threshold and normal vector. It calculates the difference in angles between the motl's z vector and the surface's z vector. It keeps the particles with an angle difference smaller than the threshold or with a tm score in the top 10.

    Parameters
    ----------
    motl : object
        The motl object to be cleaned.
    vertices : array_like
        The vertices of the surface.
    normals : array_like
        The normals of the surface.
    angle_threshold : int, optional
        The threshold for the angle difference, by default 90.
    normal_vector : list, optional
        The normal vector to be applied to the rotation, by default [0,0,1].
    keep_top : int, optional
        The number of top-scoring particles to retain, regardless of their angles. Default is None.

    Returns
    -------
    object
        The cleaned motl list.
    """
    clean_motl_list = motl
    coord = motl.get_coordinates()
    rotation = motl.get_rotations()
    # transfer from euler angle to normal vector
    motl_z_vector = rotation.apply(normal_vector)
    # searching for the closest point to motl_points
    kdtree = KDTree(vertices)
    dist, points = kdtree.query(coord, 1)
    surface_z_vector = normals[points]
    # calculate angles difference between two vectors
    diff_angles = (
        np.arccos(np.sum(motl_z_vector * surface_z_vector, axis=1)) * 180 / np.pi
    )
    if keep_top != None:
        top_score = motl.df.score >= np.sort(motl.df.score)[-keep_top]
        # keep particles with a angle different smaller than the threshold or with a tm score in top 10
        clean_motl_list.df = motl.df[
            np.logical_or(diff_angles <= angle_threshold, top_score)
        ]
    else:
        clean_motl_list.df = motl.df[diff_angles <= angle_threshold]

    return clean_motl_list


def mask_clean(motl, mask):
    """This function cleans the given motl by applying a mask to it. If a coordinate is in the mask, it is kept in the output clean_motl, otherwise it is removed.

    Parameters
    ----------
    motl : object
        The motl object to be cleaned.
    mask : ndarray
        A 3D numpy array representing the mask to be applied. It should have the same dimensions as the motl object.

    Returns
    -------
    clean_motl : object
        The cleaned motl object. It has the same type as the input motl object, but only contains the coordinates that are in the mask.

    """
    clean_motl = motl
    coords = motl.get_coordinates()
    clean_array = np.zeros(len(coords))
    for i, coord in enumerate(coords):
        is_in_mask = mask[int(coord[0]), int(coord[1]), int(coord[2])]
        if is_in_mask == 1:
            clean_array[i] = True
        else:
            clean_array[i] = False
    clean_boolmask = pd.array(clean_array, dtype="boolean")
    clean_motl.df = motl.df[clean_boolmask]

    return clean_motl


# only test for capsule mask
def surface_closing(mask, overlad_level=1, thickness_iter=15):
    mask_cvhullz = mask.copy()
    mask_cvhullz_fill = mask.copy()
    for i in range(mask.shape[2]):
        points = np.transpose(np.where(mask[:, :, i]))
        if points.size == 0:
            mask_cvhullz[:, :, i] = mask[:, :, i]
        else:
            frame_cvhull = convex_hull_image(mask[:, :, i])
            inn_frame = ndimage.binary_erosion(frame_cvhull, iterations=thickness_iter)
            out_mask = np.logical_and(frame_cvhull, np.invert(inn_frame))
            mask_cvhullz[:, :, i] = out_mask
            mask_cvhullz_fill[:, :, i] = frame_cvhull

    mask_cvhullx = mask.copy()
    mask_cvhullx_fill = mask.copy()
    for i in range(mask.shape[0]):
        points = np.transpose(np.where(mask[i, :, :]))
        if points.size == 0:
            mask_cvhullx[i, :, :] = mask[i, :, :]
        else:
            frame_cvhull = convex_hull_image(mask[i, :, :])
            inn_frame = ndimage.binary_erosion(frame_cvhull, iterations=thickness_iter)
            out_mask = np.logical_and(frame_cvhull, np.invert(inn_frame))
            mask_cvhullx[i, :, :] = out_mask
            mask_cvhullx_fill[i, :, :] = frame_cvhull

    mask_cvhully = mask.copy()
    mask_cvhully_fill = mask.copy()
    for i in range(mask.shape[1]):
        points = np.transpose(np.where(mask[:, i, :]))
        if points.size == 0:
            mask_cvhully[:, i, :] = mask[:, i, :]
        else:
            frame_cvhull = convex_hull_image(mask[:, i, :])
            inn_frame = ndimage.binary_erosion(frame_cvhull, iterations=thickness_iter)
            out_mask = np.logical_and(frame_cvhull, np.invert(inn_frame))
            mask_cvhully[:, i, :] = out_mask
            mask_cvhully_fill[:, i, :] = frame_cvhull

    mask_cvhull = mask_cvhullz + mask_cvhullx + mask_cvhully
    mask_frame_cvhull = mask_cvhullz + mask_cvhullx + mask_cvhully > overlad_level
    mask_fill_cvhull = mask_cvhullz_fill + mask_cvhullx_fill + mask_cvhully_fill > 0
    return mask_frame_cvhull, mask_cvhull, mask_fill_cvhull


def process_mask(mask, radius, mode):
    """Performs morphological operations on a given mask.

    Parameters
    ----------
    mask : ndarray
        The input mask on which the morphological operations are to be performed.
    radius : int
        The radius of the structuring element used for the morphological operations.
    mode : str
        The type of morphological operation to be performed. Options are 'closing', 'opening', 'dilation', 'erosion'.

    Returns
    -------
    pro_mask : ndarray
        The mask after the morphological operation.

    Notes
    -----
    The mask is padded with zeros on all sides by the specified radius to avoid touching the boundary during dilation.
    """
    # padding around mask to avoid touching boundary during dilation
    npad = ((radius, radius), (radius, radius), (radius, radius))
    mask_pad = np.pad(mask, pad_width=npad, mode="constant", constant_values=0)
    if mode == "closing":
        pro_mask = skimage.morphology.isotropic_closing(mask_pad, radius=radius)
    if mode == "opening":
        pro_mask = skimage.morphology.isotropic_opening(mask_pad, radius=radius)
    if mode == "dilation":
        pro_mask = skimage.morphology.isotropic_dilation(mask_pad, radius=radius)
    if mode == "erosion":
        pro_mask = skimage.morphology.isotropic_erosion(mask_pad, radius=radius)
    pro_mask = pro_mask[radius:-radius, radius:-radius, radius:-radius]
    return pro_mask.astype(float)


def marker_from_fill_mask(fill_mask, ero_radius=60, clos_radius=20, mode="None"):
    # TODO: including create marker from centroid
    """This function generates a marker for mask_fill_gap by performing erosion and closing operations on the mask.

    Parameters
    ----------
    fill_mask : ndarray
        The fill mask to process.
    ero_radius : int, optional
        The radius for the erosion operation, by default 60.
    clos_radius : int, optional
        The radius for the closing operation, by default 20.
    mode : str, optional
        The mode for the mask processing, by default 'None'.

    Returns
    -------
    core_mark : ndarray
        The mark for mask_fill_gap.
    """
    core_mark = process_mask(fill_mask, ero_radius, mode="erosion")
    core_mark = process_mask(core_mark, clos_radius, mode="closing") * 2
    core_mark[0:, 0:, 0] = 1
    core_mark[0, 0:, 0:] = 1
    core_mark[0:, 0, 0:] = 1
    core_mark[0:, -1, 0:] = 1
    core_mark[-1, 0:, 0:] = 1
    core_mark[0:, 0:, -1] = 1

    return core_mark


def mask_fill_gap(mask, marker_ero=10):
    chull_mask, sum_mask, fill_mask = surface_closing(mask)
    core_mark = marker_from_fill_mask(fill_mask, ero_radius=marker_ero)
    chull_mask_close = process_mask(mask, 60, mode="closing")
    chull_mask_dila = process_mask(chull_mask_close, 20, mode="dilation")
    core_mask = watershed(chull_mask_dila, core_mark)
    core_mask[core_mask == 1] = 0
    core_mask[core_mask == 2] = 1

    # dilating and erosing core mask to generate capsule mask
    core_mask_op = process_mask(core_mask, 50, mode="opening")
    core_mask_dila = process_mask(core_mask_op, 10, mode="dilation")
    core_mask_eros = process_mask(core_mask_op, 30, mode="erosion")
    capsule_mask = np.logical_xor(core_mask_dila, core_mask_eros)
    return capsule_mask, core_mask_op
