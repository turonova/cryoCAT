import numpy as np
import pandas as pd
import skimage
import scipy.ndimage as ndimage
from skimage.segmentation import watershed
from skimage.morphology import convex_hull_image
from scipy.spatial import ConvexHull
from scipy.spatial import KDTree
import warnings

from cryocat import geom
from cryocat import cryomap
from cryocat import cryomotl
from cryocat import cryomask
from cryocat import tgeometry as tg


class SamplePoints:
    def __init__(self):

        self.vertices = None
        self.normals = None
        self.faces = None
        self.area = None
        self.shape_list = None
        self.shape_mask = None
        self.sample_distance = None

    @classmethod
    def load(cls, shape_input):
        samp = cls()
        if isinstance(shape_input, str):
            if shape_input.endswith(".mrc"):
                samp.shape_mask = cryomap.read(shape_input)
            elif shape_input.endswith("csv"):
                samp.shape_list = cls.load_shapes(input_file_name=shape_input)
            elif shape_input.endswith("em"):
                motl = cryomotl.Motl.load(shape_input)
                samp.vertices = motl.get_coordinates()
                angles = motl.get_angles()
                samp.normals = geom.euler_angles_to_normals(angles)
                # convert to unit vectors
                magnitudes = np.linalg.norm(samp.normals, axis=1, keepdims=True)
                samp.normals = samp.normals / magnitudes
            else:
                raise ValueError(
                    "The input file name", input, "is neither mrc, csv or em file!"
                )
        elif isinstance(shape_input, list):
            samp.shape_list = shape_input
        elif isinstance(shape_input, np.ndarray):
            samp.shape_mask = shape_input
        elif isinstance(shape_input, object):
            samp.vertices = shape_input.get_coordinates()
            angles = shape_input.get_angles()
            samp.normals = geom.euler_angles_to_normals(angles)
            # convert to unit vectors
            magnitudes = np.linalg.norm(samp.normals, axis=1, keepdims=True)
            samp.normals = samp.normals / magnitudes
        else:
            raise ValueError(
                "The input is neither string, shape list, 3d volume or motl file!"
            )
        return samp

    def boundary_sampling(self, sampling_distance=1):
        """This method performs boundary sampling on the object. For input of mrc mask,
        it uses the marching_cubes method from the skimage module to compute the vertices, faces,
        normals, and values. For input of manual labeling, it uses the convex hull to compute the vertices, normals, and faces.
        The input object will be updated with the information from boundary sampling

        Parameters
        ----------
        sampling_distance : int, optional
            The distance between sample points, by default 1

        Returns
        -------
        None
        """
        if self.shape_mask is not None and self.shape_list is not None:
            raise ValueError(
                "Both shape_mask and shape_list are set. Only one should be defined."
            )
        elif self.shape_mask is not None:
            self.vertices, self.faces, self.normals, _ = skimage.measure.marching_cubes(
                self.shape_mask, step_size=sampling_distance
            )
            # make sure the normal vectors are unit vectors
            magnitudes = np.linalg.norm(self.normals, axis=1, keepdims=True)
            self.normals = self.normals / magnitudes
            # calculate area
            self.area = skimage.measure.mesh_surface_area(self.vertices, self.faces)
            self.sample_distance = sampling_distance
        elif self.shape_list is not None:
            self.vertices, self.normals, self.faces, self.area = (
                SamplePoints.get_oversampling(self.shape_list, sampling_distance)
            )
            # make sure the normal vectors are unit vectors
            magnitudes = np.linalg.norm(self.normals, axis=1, keepdims=True)
            self.normals = self.normals / magnitudes
            self.sample_distance = sampling_distance

    def write(self, path, input_dict=None):
        """Writes the vertices and normals to a `motl_list`.

        Parameters
        ----------
        path : str
            The destination path for the output file.
        input_dict : dict, optional
            A dictionary with keys from :attr:`cryocat.cryomotl.Motl.motl_columns` and new values to be assigned.
            Coordinates and angles are derived from the class. Subtomograms are sequentially ordered starting from 1, and the class is set to 1 by default. All other fields will assign 0 if not specified in this dictionay

        Returns
        -------
        None

        Notes
        -----
        This method creates pandas DataFrames from the object's vertices and normals, converts the normals to Euler angles,
        and writes the resulting data to a file at the specified path.
        """
        # create panda frames from vertices
        pd_points = pd.DataFrame(
            {
                "x": self.vertices[:, 0],
                "y": self.vertices[:, 1],
                "z": self.vertices[:, 2],
            }
        )
        # create panda frames from normals
        pd_normals = pd.DataFrame(
            {"x": self.normals[:, 0], "y": self.normals[:, 1], "z": self.normals[:, 2]}
        )
        # get Euler angles from coordinates
        angles = geom.normals_to_euler_angles(pd_normals, output_order="zzx")
        # create pandas
        pd_angles = pd.DataFrame(
            {"phi": angles[:, 0], "psi": angles[:, 1], "theta": angles[:, 2]}
        )
        # input_dict = {"coord": pd_points, "angles": pd_angles}
        # only save coord and angles, add value and faces in future. The number of faces was inconsistence with particles number which might be a issue for writing in motl_list.
        motl = cryomotl.Motl()
        motl.fill(pd_points)
        motl.fill(pd_angles)
        motl.df["class"] = float(1)
        motl.fill({"subtomo_id": np.arange(1, motl.df.shape[0] + 1, dtype="float")})

        # fill other fields base on the input_dict
        if input_dict is not None:
            if any(
                key == "x"
                or key == "y"
                or key == "z"
                or key == "coord"
                or key == "phi"
                or key == "psi"
                or key == "theta"
                or key == "angles"
                for key in input_dict.keys()
            ):
                raise ValueError(
                    "Coordinates and angles are filled from the class. Please assign values only to motl_columns unrelated to them."
                )
            else:
                motl.fill(input_dict)

        motl.write_out(path)

    def shift_points(self, distances):
        """Shift the vertices of a geometric object along their normal vectors.

        Parameters
        ----------
        distances : int
            Moving distances value in pixels to shift each vertex.

        Notes
        -----
        This function modifies the vertices of the object in place by moving each
        vertex along its corresponding normal vector. The normal vectors are first converted
        to unit normal vectors before applying the shift.
        """

        self.vertices = self.vertices + distances * self.normals

    def shift_points_in_groups(self, shift_dict):
        """Shift points in groups based on a dictionary of shift values.

        Parameters
        ----------
        shift_dict : dict
            A dictionary where keys are tuples representing normal vectors and values are the shift magnitudes in pixels to be applied to the vertices corresponding to those normal vectors.

        Raises
        ------
        UserWarning
            If a key in `shift_dict` does not match any normal vectors in the point cloud, a warning is issued indicating that only matching vectors have been moved.

        Notes
        -----
        This function modifies the `vertices` attribute of the object by shifting the points in the direction of their normal vectors, scaled by the corresponding value in `shift_dict`.
        """
        for key, value in shift_dict.items():
            match = (self.normals == list(key)).all(axis=1)
            if match.any():
                self.vertices[match] = (
                    self.vertices[match] + value * self.normals[match]
                )
            else:
                warnings.warn(
                    f"Vectors '{key}' were not matched to any normal vectors in the point cloud. Only the matching vectors have been moved."
                )

    # remove sample points with specific normals value
    # TODO more precise removing method, removing point base on normal values/eular angles
    def rm_points(self, rm_surface):
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
            self.normals[:, 0] = np.absolute(self.normals[:, 0])
        elif rm_surface == -1:
            tob = -1

        for i in range(len(self.normals)):
            if (
                self.normals[i, 0] == tob
                and self.normals[i, 1] == 0
                and self.normals[i, 2] == 0
            ):
                removed_points.append(i)

        cleaned_points = np.delete(self.vertices, removed_points, axis=0)
        cleaned_normals = np.delete(self.normals, removed_points, axis=0)

        return cleaned_points, cleaned_normals

    # replace the normal with the closest postion from convexHull
    # Substitute the normal in one motl with the adjacent normal from oversampling points.
    def reset_normals(self, motl, bin_factor=None):
        """Replaces the angle of the motl with the angle from adjacent oversampling points.

        Parameters
        ----------
        motl : object
            The motl for resetting angles.
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
        motl_points = motl.get_coordinates()
        if bin_factor != None:
            sample_points = self.vertices * bin_factor  # add binning to points
        else:
            sample_points = self.vertices
        # searching for the closest point to motl_points
        kdtree = KDTree(sample_points)
        _, points = kdtree.query(motl_points, 1)

        ## replacing eular angles in motlist to new angles
        n_normals = self.normals[points]
        # create panda frames from normals
        pd_normals = pd.DataFrame(
            {"x": n_normals[:, 0], "y": n_normals[:, 1], "z": n_normals[:, 2]}
        )
        # get Euler angles from normals
        angles = geom.normals_to_euler_angles(pd_normals, output_order="zzx")
        # replace angles in motl
        pd_angles = pd.DataFrame(
            {"phi": angles[:, 0], "psi": angles[:, 1], "theta": angles[:, 2]}
        )
        motl.fill(pd_angles)

        return motl

    def angles_clean(
        self, motl, angle_threshold=90, normal_vector=[0, 0, 1], keep_top=None
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
        # angles = motl.get_angles()
        rotation = motl.get_rotations()
        # transfer from euler angle to normal vector
        # motl_z_vector = geom.euler_angles_to_normals(angles)
        motl_z_vector = rotation.apply(normal_vector)
        # searching for the closest point to motl_points
        kdtree = KDTree(self.vertices)
        _, points = kdtree.query(coord, 1)
        surface_z_vector = self.normals[points]
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

    # TODO add parallization to speed up
    # two motl list or two class?
    def inner_and_outer_pc(self, thickness):
        """This function calculates the inner and outer vertices and normals of a mask. Ideally for irregular capsules and membranes.

        Parameters
        ----------
        mask : ndarray
            The input mask for which the inner and outer vertices and normals are to be calculated.
        thickness : float
            The distance to between two sets of points in pixels, should be the thickness of your mask region.

        Returns
        -------
        outer_vertices : ndarray
            The outer vertices of the mask.
        outer_normals : ndarray
            The normals corresponding to the outer vertices of the mask.
        inner_vertices : ndarray
            The inner vertices of the mask.
        inner_normals : ndarray
            The normals corresponding to the inner vertices of the mask.

        """
        # find centroid of points
        cent = np.mean(self.vertices, axis=0)
        vertices = self.vertices
        normals = self.normals
        point_spacing = self.sample_distance
        inner_class = SamplePoints()
        outer_class = SamplePoints()

        is_inner = np.zeros(vertices.shape[0], dtype=bool)
        for i in range(vertices.shape[0]):
            if is_inner[i]:  # Skip if already marked as inner
                continue

            coord = vertices[i]
            # find distance of all points to line between iterate points to centroid
            line_vec = cent - coord
            line_len = np.linalg.norm(line_vec)  # Length of the line segment
            distances = (
                np.linalg.norm(np.cross(line_vec, vertices - cent), axis=1) / line_len
            )  # Distance from other points to line

            # check if projections of all points on line fall within line segment
            projection_lengths = np.dot(vertices - coord, line_vec) / line_len
            is_point_on_line = (projection_lengths > thickness * 0.5) & (
                projection_lengths < line_len
            )

            # Points close to the line
            is_dist_small = distances <= point_spacing
            # Normal direction check:
            dot_products = np.dot(normals, line_vec / line_len)  # Normalize line_vec
            # Clip values to avoid numerical errors outside [-1,1]
            angles = np.degrees(np.arccos(np.clip(dot_products, -1, 1)))
            is_normal_diff = angles < np.radians(90)  # 90 degrees threshold

            # record idx of coord that has a distance to line less then 2 and within the range of line segment
            is_inner_point = np.logical_and(
                is_dist_small, is_point_on_line, is_normal_diff
            )

            if np.any(is_inner_point):
                is_inner = np.logical_or(is_inner, is_inner_point)

        inner_class.vertices = vertices[is_inner]
        inner_class.normals = normals[is_inner]
        outer_class.vertices = vertices[~is_inner]
        outer_class.normals = normals[~is_inner]
        return outer_class, inner_class

    # shape_layer is shape data layer, e.g. viewer.layers[x].data where z is layer id

    @staticmethod
    def get_oversampling(shape_layer, sampling_distance):
        """Create oversample points from the convex hull of a point cloud representing your surface.

        Parameters
        ----------
        shape_layer : list
            A list containing point cloud coordinates ordered by x, y, z, with points sharing the same z-value organized into arrays.
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
        faces = hull.equations
        area = hull.area

        return tri_points, normals, faces, area

    @staticmethod
    def load_shapes(input_file_name):
        """load_shapes(input_file_name: str) -> List[np.array]:

        This function loads a surface point cloud from a csv file. The point cloud coordinates are ordered by x, y, z.
        Points sharing the same z-value are organized into arrays.

        Parameters:
        -----------
        input_file_name : str
            The path to your csv point cloud file.

        Returns:
        --------
        shape_list : list
            A list containing point cloud coordinates ordered by x, y, z, with points sharing the same z-value organized into arrays.
        """
        df = pd.read_csv(input_file_name)
        s_id = df["s_id"].unique()

        shape_list = []

        for sf in s_id:
            a = df.loc[df["s_id"] == sf]
            sp_array = np.zeros((a.shape[0], 3))
            sp_array[:, 0] = (a["x"]).values
            sp_array[:, 1] = (a["y"]).values
            sp_array[:, 2] = (a["z"]).values
            shape_list.append(sp_array)

        return shape_list

    @staticmethod
    def load_shapes_as_point_cloud(input_file_name):
        point_cloud = np.loadtxt(input_file_name)
        return point_cloud

    @staticmethod
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
                    i
                    for i, num in enumerate(faces)
                    if sum(abs(num[0:3]) == [1, 0, 0]) == 3
                ]
            if tb_faces == []:
                raise ValueError("The target top/bottom surfaces doesn't exist")
            all_faces_points_in = hull.simplices  # indices of points for surface
            tb_faces_points_in = all_faces_points_in[tb_faces]
            face_points_coord = [
                [mask_points[j] for j in i] for i in tb_faces_points_in
            ]
            face_points_array = np.asarray(face_points_coord)
            tb_areas = geom.area_triangle(face_points_array)
            total_tb_area = sum(tb_areas)
            updated_area = hull.area - total_tb_area

        return updated_area

    @staticmethod
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
    @staticmethod
    def slice_cvhull_closing(mask, overlad_level=1, thickness_iter=15):
        """This function performs surface closing operation on a given mask. It iterates through and applies the convex hull to each slice of the mask in three axis.
        The function returns the mask after applying the surface closing operation, the convex hull mask, and the capsule mask.

        Parameters
        ----------
        mask : ndarray
            The input mask on which the surface closing operation is to be performed.
        overlad_level : int, optional
            The threshold level for the overlap of the convex hull masks in different axis, by default 1 which keeps all pixels from three axis.
        thickness_iter : int, optional
            The number of iterations for the binary erosion operation used to generate the capsule mask, by default 15.

        Returns
        -------
        capmask_frame_cvhull : ndarray
            The capsule mask after applying the closing operation.
        mask_cvhull : ndarray
            The sum convex hull mask from three axis.
        mask_fill_cvhull : ndarray
            The mask with filled center.
        """
        mask_cvhullz = mask.copy()
        mask_cvhullz_fill = mask.copy()
        for i in range(mask.shape[2]):
            points = np.transpose(np.where(mask[:, :, i]))
            if points.size == 0:
                mask_cvhullz[:, :, i] = mask[:, :, i]
            else:
                frame_cvhull = convex_hull_image(mask[:, :, i])
                inn_frame = ndimage.binary_erosion(
                    frame_cvhull, iterations=thickness_iter
                )
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
                inn_frame = ndimage.binary_erosion(
                    frame_cvhull, iterations=thickness_iter
                )
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
                inn_frame = ndimage.binary_erosion(
                    frame_cvhull, iterations=thickness_iter
                )
                out_mask = np.logical_and(frame_cvhull, np.invert(inn_frame))
                mask_cvhully[:, i, :] = out_mask
                mask_cvhully_fill[:, i, :] = frame_cvhull

        mask_cvhull = mask_cvhullz + mask_cvhullx + mask_cvhully
        capmask_slice_cvhull = (
            mask_cvhullz + mask_cvhullx + mask_cvhully + mask > overlad_level
        )
        mask_fill_cvhull = mask_cvhullz_fill + mask_cvhullx_fill + mask_cvhully_fill > 0
        return capmask_slice_cvhull, mask_cvhull, mask_fill_cvhull

    @staticmethod
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

    @staticmethod
    def marker_from_fill_mask(
        fill_mask, ero_radius=30, clos_radius=20, boundary_thickness=5
    ):
        # TODO: including create marker from centroid
        """This function generates a marker for mask_fill_gap by performing erosion and closing operations on the mask.

        Parameters
        ----------
        fill_mask : ndarray
            The fill mask to process.
        ero_radius : int, optional
            The radius for the erosion operation, by default 30.
        clos_radius : int, optional
            The radius for the closing operation, by default 20.
        boundary_thickness : int, optional
            The thickness of boundary marker, by default 5


        Returns
        -------
        core_mark : ndarray
            The marker for mask_fill_gap.
        """
        core_mark = SamplePoints.process_mask(fill_mask, ero_radius, mode="erosion")
        core_mark = (
            SamplePoints.process_mask(core_mark, clos_radius, mode="closing") * 2
        )
        core_mark[:boundary_thickness, :, :] = 1
        core_mark[-boundary_thickness:, :, :] = 1
        core_mark[:, :boundary_thickness, :] = 1
        core_mark[:, -boundary_thickness:, :] = 1
        core_mark[:, :, :boundary_thickness] = 1
        core_mark[:, :, -boundary_thickness:] = 1
        return core_mark

    @staticmethod
    def marker_from_centroid(mask, centroid_mark_size=5, boundary_thickness=5):
        """This function generates a marker from the centroid of a given mask.

        Parameters
        ----------
        mask : ndarray
            The input mask from which the centroid is calculated.
        centroid_mark_size : int, optional
            The radius of the sphere centroid marker, by default 5.
        boundary_thickness : int, optional
            The thickness of boundary marker, by default 5

        Returns
        -------
        cent_mark : ndarray
            The generated marker from the centroid of the mask. The marker is a 3D array with the same shape as the input mask. The centroid is marked with a sphere with defined radius, and the edges of the array are marked with 1.

        Notes
        -----
        The centroid is calculated using the centroid() function. The centroid marker is a sphere with radius of centroid_mark_size centered at the centroid. The edges of the array are also marked.
        """
        centroid = skimage.measure.centroid(mask)
        cent_mark = np.zeros(mask.shape)
        cent_mark = (
            cryomask.spherical_mask(
                mask.shape, radius=centroid_mark_size, center=centroid
            )
            * 2
        )
        cent_mark[:boundary_thickness, :, :] = 1
        cent_mark[-boundary_thickness:, :, :] = 1
        cent_mark[:, :boundary_thickness, :] = 1
        cent_mark[:, -boundary_thickness:, :] = 1
        cent_mark[:, :, :boundary_thickness] = 1
        cent_mark[:, :, -boundary_thickness:] = 1
        return cent_mark

    @staticmethod
    def mask_fill_gap(mask, marker_ero=10, level=1):
        """This function is used to fill gaps in a capsule mask using watershed flooding from the marker and the edge of volume.

        Parameters
        ----------
        mask : ndarray
            Binary image mask to be processed.
        marker_ero : int, optional
            Radius for the erosion operation used to generate the capsule mask, by default 10.
        level : int, optional
            overcross threshold for the convex hull masks in different axis, by default 1.

        Returns
        -------
        tuple
            Returns a tuple containing the capsule mask and the processed core mask.

        Notes
        -----
        The function performs a series of morphological operations on the input mask, including closing, dilation, erosion, and opening. It uses the watershed algorithm to segment the mask into different regions. The function then generates a capsule mask by performing a logical XOR operation on the dilated and eroded core mask.
        """
        cap_mask, _, fill_mask = SamplePoints.slice_cvhull_closing(
            mask, overlad_level=level, thickness_iter=15
        )
        core_mark = SamplePoints.marker_from_fill_mask(fill_mask, ero_radius=marker_ero)
        chull_mask_close = SamplePoints.process_mask(cap_mask, 20, mode="closing")
        core_mask = watershed(chull_mask_close.astype(int), core_mark.astype(int))
        core_mask[core_mask == 1] = 0
        core_mask[core_mask == 2] = 1

        # dilating and erosing core mask to generate capsule mask
        core_mask_dila = SamplePoints.process_mask(core_mask, 10, mode="dilation")
        core_mask_eros = SamplePoints.process_mask(core_mask, 10, mode="erosion")
        capsule_mask = np.logical_xor(core_mask_dila, core_mask_eros)
        return capsule_mask, core_mask
