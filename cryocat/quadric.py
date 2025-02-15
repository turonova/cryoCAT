import numpy as np
import pandas as pd
from scipy.optimize import fsolve
from cryocat import cryomotl


class QuadricsM:
    def __init__(self, input_data, quadric="ellipsoid", feature_id="object_id"):

        self.dict = {}
        self.f_id = feature_id

        if isinstance(input_data, cryomotl.Motl):
            input_motl = cryomotl.Motl.load(input_data)
            for f in input_motl.get_unique_values(feature_id=feature_id):
                fm = input_motl.get_motl_subset(feature_values=f, feature_id=feature_id, reset_index=True)
                tomo_id = fm.df["tomo_id"].values[0]
                self.dict[(tomo_id, f)] = Quadric.load(fm.get_coordinates(), quadric_type=quadric)
        elif isinstance(input_data, str):
            df = pd.read_csv(input_data)
            # todo write iteration over rows
            for _, ds in df.iterrows():
                f = ds[feature_id]
                tomo_id = ds["tomo_id"]
                self.dict[(tomo_id, f)] = Quadric.load(ds, quadric_type=quadric)
        else:
            raise ValueError(f"The input type {type(input_data)} is currently not supported.")

    def write_out(self, output_name):

        full_df = pd.DataFrame()

        for key, value in self.dict.items():
            df = value.get_props_as_df()
            df["tomo_id"] = key[0]
            df[self.f_id] = key[1]
            full_df = pd.concat([full_df, df], ignore_index=True)

        full_df.to_csv(output_name, index=False)

    def distance_point_surface(self, tomo_id, feature_id, points):

        distances = []
        for p in points:
            distances.append(self.dict[(tomo_id, feature_id)].distance_point_surface(p))

        return distances

    def distance_point_center(self, tomo_id, feature_id, points):

        distances = []
        for p in points:
            distances.append(self.dict[(tomo_id, feature_id)].distance_point_center(p))

        return distances

    def find_closest_quadric(self, tomo_id, points):

        f_ids = []
        for key in self.dict.keys():
            if key[0] == tomo_id:
                f_ids.append(key[1])

        num_points = points.shape[0]
        closest_ids = np.full(num_points, -1)  # -1 indicates no intersection found
        closest_distances = np.full(num_points, np.inf)  # Start with infinity for closest distance

        for i, p in enumerate(points):
            for f in f_ids:
                distance = self.dict[(tomo_id, f)].distance_point_center(p)
                if distance < closest_distances[i]:  # Closest absolute distance
                    closest_distances[i] = distance
                    closest_ids[i] = f

        return closest_ids


class Quadric:
    def __init__(self):
        self.columns = []

    @classmethod
    def load(cls, input_data, quadric_type="ellipsoid"):
        if quadric_type == "ellipsoid":
            return Ellipsoid(input_data=input_data)
        else:
            raise ValueError(f"The quadric type {quadric_type} is currently not supported.")


class Ellipsoid(Quadric):

    def __init__(self, input_data=None):

        # define columns
        self.columns = [
            "cx",
            "cy",
            "cz",
            "rx",
            "ry",
            "rz",
            "ev1x",
            "ev1y",
            "ev1z",
            "ev2x",
            "ev2y",
            "ev2z",
            "ev3x",
            "ev3y",
            "ev3z",
            "p1",
            "p2",
            "p3",
            "p4",
            "p5",
            "p6",
            "p7",
            "p8",
            "p9",
            "p10",
        ]

        if isinstance(input_data, dict):
            self.from_dict(input_data)
        elif isinstance(input_data, (np.ndarray, list)):
            self.from_array_like(input_data)
        elif isinstance(input_data, str):
            self.from_df(pd.read_csv(input_data))
        elif isinstance(input_data, (pd.DataFrame, pd.Series)):
            self.from_df(input_data)
        else:
            print("Warning: the ellipsoid is initialized with zero values!")
            self.center = np.zeros((3,))
            self.radii = np.zeros((3,))
            self.e_vec1 = np.zeros((3,))
            self.e_vec2 = np.zeros((3,))
            self.e_vec3 = np.zeros((3,))
            self.params = np.zeros((10,))

    def from_dict(self, input_dict):
        attr_count = 0
        for key, value in input_dict.items():
            if key in ["center", "radii", "e_vec1", "e_vec2", "e_vec3", "params"]:
                setattr(self, key, value)
                attr_count += 1

        if hasattr(self.params):
            if attr_count != 6:
                self.compute_props()
        else:
            raise ValueError("The dictionary does not contain the parameteric description")

    def from_df(self, df):
        self.center = df[["cx", "cy", "cz"]].values
        self.radii = df[["rx", "ry", "rz"]].values
        self.e_vec1 = df[["ev1x", "ev1y", "ev1z"]].values
        self.e_vec2 = df[["ev2x", "ev2y", "ev2z"]].values
        self.e_vec3 = df[["ev3x", "ev3y", "ev3z"]].values
        self.params = df[["p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9", "p10"]]

    def from_array_like(self, array_like):

        if isinstance(array_like, list):
            al_size = len(array_like)
            al_shape = 0
        elif isinstance(array_like, np.ndarray):
            al_size = array_like.size
            array_like = np.atleast_2d(array_like)
            al_shape = array_like.shape
        else:
            raise ValueError("The input is neither list or np.ndarray")

        if al_size == 10:
            self.params = array_like
            self.compute_props()
        elif al_size == 25:
            self.center = array_like[:3]
            self.radii = array_like[3:6]
            self.e_vec1 = array_like[6:9]
            self.e_vec2 = array_like[9:12]
            self.e_vec3 = array_like[12:15]
            self.params = array_like[15:]
        elif al_shape != 0 and al_shape[1] == 3:  # the coordinates are passed
            self.fit_into_coord(array_like)
        else:
            print(f"The input data does not have correct shape.")

    def compute_props(self):

        A = np.array(
            [
                [self.params[0], self.params[3], self.params[4], self.params[6]],
                [self.params[3], self.params[1], self.params[5], self.params[7]],
                [self.params[4], self.params[5], self.params[2], self.params[8]],
                [self.params[6], self.params[7], self.params[8], self.params[9]],
            ]
        )

        self.center = np.linalg.solve(-A[:3, :3], self.params[6:9])

        translation_matrix = np.eye(4)
        translation_matrix[3, :3] = self.center.T

        R = translation_matrix.dot(A).dot(translation_matrix.T)

        evals, evecs = np.linalg.eig(R[:3, :3] / -R[3, 3])
        evecs = evecs.T

        self.e_vec1 = evecs[0, :]
        self.e_vec2 = evecs[1, :]
        self.e_vec3 = evecs[2, :]

        self.radii = np.sqrt(1.0 / np.abs(evals))
        self.radii *= np.sign(evals)

    @staticmethod
    def load(input_data=None, feature_id="object_id"):

        if isinstance(input_data, cryomotl.Motl):
            el_params = Ellipsoid.load_from_motl(input_data, feature_id=feature_id)
        elif isinstance(input_data, str):
            # TODO add check that the format is correct
            el_params = pd.read_csv(input_data)
        elif isinstance(input_data, np.ndarray):
            # TODO add check that the format is correct
            el_params = pd.DataFrame(data=input_data, columns=columns)
        elif not input_data:
            el_params = pd.DataFrame(columns=columns)
        else:
            # TODO add possibility to specify just by parameters directly
            raise ValueError("Invalid type of parametric surface")

        return el_params

    @staticmethod
    def load_from_motl(input_motl, feature_id="object_id"):

        in_motl = cryomotl.Motl.load(input_motl)
        features = in_motl.get_unique_values(feature_id=feature_id)
        el_params_all = pd.DataFrame()

        for f in features:
            fm = in_motl.get_motl_subset(feature_values=f, feature_id=feature_id)
            coord = fm.get_coordinates()
            el_params = pd.DataFrame()
            center, radii, evecs, v = Ellipsoid.fit_into_coord(coord)
            el_params["tomo_id"] = [fm.df.iloc[0]["tomo_id"]]  # assuming that each object has unified tomo_id
            el_params[feature_id] = [f]
            el_params[["cx", "cy", "cz"]] = [center]
            el_params[["rx", "ry", "rz"]] = [radii]
            el_params[["ev1x", "ev1y", "ev1z"]] = [evecs[0, :]]
            el_params[["ev2x", "ev2y", "ev2z"]] = [evecs[1, :]]
            el_params[["ev3x", "ev3y", "ev3z"]] = [evecs[2, :]]
            el_params[["p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9", "p10"]] = [v]
            el_params_all = pd.concat([el_params_all, el_params])

        el_params_all.reset_index(drop=True, inplace=True)

        return el_params_all

    def get_props_as_ndarray(self):

        output_array = np.array((25,))
        output_array[:3] = self.center
        output_array[3:6] = self.radii
        output_array[6:9] = self.e_vec1
        output_array[9:12] = self.e_vec2
        output_array[12:15] = self.e_vec3
        output_array[15:] = self.params

        return output_array

    def get_props_as_dict(self):

        output_dict = {}
        output_dict["center"] = self.center
        output_dict["radii"] = self.radii
        output_dict["e_vec1"] = self.e_vec1
        output_dict["e_vec2"] = self.e_vec2
        output_dict["e_vec3"] = self.e_vec3
        output_dict["params"] = self.params

        return output_dict

    def get_props_as_list(self):

        output_list = []
        output_list.append(self.center)
        output_list.append(self.radii)
        output_list.append(self.e_vec1)
        output_list.append(self.e_vec2)
        output_list.append(self.e_vec3)
        output_list.append(self.params)

        return output_list

    def get_props_as_df(self):

        df = pd.DataFrame(columns=self.columns)
        df[["cx", "cy", "cz"]] = [self.center]
        df[["rx", "ry", "rz"]] = [self.radii]
        df[["ev1x", "ev1y", "ev1z"]] = [self.e_vec1]
        df[["ev2x", "ev2y", "ev2z"]] = [self.e_vec1]
        df[["ev3x", "ev3y", "ev3z"]] = [self.e_vec1]
        df[["p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9", "p10"]] = [self.params]

        return df

    def fit_into_coord(self, coord):
        """Fit an ellipsoid to a set of 3D coordinates. It is based on
        http://www.mathworks.com/matlabcentral/fileexchange/24693-ellipsoid-fit

        Parameters
        ----------
        coord : ndarray
            An array of shape (N, 3) where each row represents the x, y, z coordinates.

        Notes
        -----
        This function fits an ellipsoid to a set of points by solving a linear least squares problem to estimate the
        parameters of the ellipsoid's equation in its algebraic form. It stores the parametric coefficients into
        self.params and compute self.center, self.radii, and self.e_vec1-3.
        """

        x = coord[:, 0]
        y = coord[:, 1]
        z = coord[:, 2]
        D = np.array(
            [
                x * x + y * y - 2 * z * z,
                x * x + z * z - 2 * y * y,
                2 * x * y,
                2 * x * z,
                2 * y * z,
                2 * x,
                2 * y,
                2 * z,
                1 - 0 * x,
            ]
        )
        d2 = np.array(x * x + y * y + z * z).T  # rhs for LLSQ
        u = np.linalg.solve(D.dot(D.T), D.dot(d2))
        a = np.array([u[0] + 1 * u[1] - 1])
        b = np.array([u[0] - 2 * u[1] - 1])
        c = np.array([u[1] - 2 * u[0] - 1])
        v = np.concatenate([a, b, c, u[2:]], axis=0).flatten()
        self.params = v
        self.compute_props()

    def intersection_line_segment(self, line_segment):
        """
        Compute the intersection between a line segment (defined by a 3D point and direction) and ellipsoid.

        Parameters
        ----------
        ray: `geom.LineSegment`
            An instance of a line_segment object. See `geom.LineSegment` for more details.

        Returns
        -------
        p1: np.ndarray
            (3,) array with the closest intersection point, or NaN if no intersection exists.
        p2: np.ndarray
            (3,) array with the second intersection point, or NaN if no intersection exists.
        d1: float
            Distance between line_segment.point and the closest intersection point (p1), or NaN. Is always positive.
        d2: float
            Distance between line_segment.point and the second intersection point (p2), or NaN. Is always positive.
        is_inside: bool
            Returns True if point lies inside the ellipsoid.

        """
        p1, p2, d1, d2, is_inside = self.intersection_ray(line_segment)

        if d1 != np.nan:
            if d1 > line_segment.length:  # no intersection
                p1, p2, d1, d2, is_inside = np.nan, np.nan, np.nan, np.nan, is_inside
            elif d2 != np.nan:
                if d2 > line_segment.length:  # only one intersection
                    p1, p2, d1, d2, is_inside = p1, np.nan, d1, np.nan, is_inside

        return p1, p2, d1, d2, is_inside

    def intersection_ray(self, ray):
        """
        Compute the intersection between a ray (defined by a 3D point and direction) and ellipsoid.

        Parameters
        ----------
        ray: `geom.Line`
            An instance of a ray object. See `geom.Line` for more details.

        Returns
        -------
        p1: np.ndarray
            (3,) array with the closest intersection point, or NaN if no intersection exists or is not in the direction
            of the ray.
        p2: np.ndarray
            (3,) array with the second intersection point, or NaN if no intersection exists or is not in the direction
            of the ray.
        d1: float
            Distance between ray.point and the closest intersection point (p1), or NaN. Is always positive.
        d2: float
            Distance between ray.point and the second intersection point (p2), or NaN. Is always positive.
        is_inside: bool
            Returns True if point lies inside the ellipsoid.

        Notes
        -----
        Unlike for intersection_line function, this function returns only those intersection points that lie in
        the positive direction of the ray.
        """
        p1, p2, d1, d2, is_inside = self.intersection_line(ray)

        if p2 != np.nan:  # both intersection points exist

            if d1 < 0 and d2 > 0:
                return p2, np.nan, d2, np.nan, True
            elif d1 > 0 and d2 < 0:
                return p1, np.nan, d1, np.nan, True
            elif d1 < 0 and d2 < 0:
                return np.nan, np.nan, np.nan, np.nan, False
            else:
                return p1, p2, d1, d2, False

        elif p1 != np.nan:  # one intersection exist
            if d1 < 0:  # closest point in the wrong direction -> no intersection
                return np.nan, np.nan, np.nan, np.nan, False
            else:
                return p1, np.nan, d1, np.nan, False
        else:
            return p1, p2, d1, d2, is_inside

    def intersection_line(self, line):
        """
        Compute the intersection between a line (defined by a 3D point and direction) and ellipsoid.

        Parameters
        ----------
        line: `geom.line`
            An instance of a line object. See `geom.line` for more details.

        Returns
        -------
        p1: np.ndarray
            (3,) array with the closest intersection point, or NaN if no intersection exists.
        p2: np.ndarray
            (3,) array with the second intersection point, or NaN if no intersection exists.
        d1: float
            Distance between line.point and the closest intersection point (p1), or NaN. It can be negative if the
            closest point lies in the opposite direction.
        d2: float
            Distance between line.point and the second intersection point (p2), or NaN. It can be negative if the
            second intersection point lies in the opposite direction.
        is_inside: bool
            Returns True if a point (that defined the line) lies inside the ellipsoid.

        Notes
        -----
        In case of two intersection points, the function return the one that is closest as the first point, disregarding
        whether the distance is negative or positive, i.e. whether the closest point lies in the direction that
        defined the line. The reasoning behind this is that for a line one does not consider the directionality
        (i.e. the direction and a point are used only to define a line) and thus the intersection points are sorted
        by their distance regardless of their sign.
        """
        # Extract line parameters
        x, y, z = line.p[0], line.p[1], line.p[2]
        n1, n2, n3 = line.dir[0], line.dir[1], line.dir[2]

        # Extract ellipsoid parameters
        a, b, c, d, e, f, g, h, i, j = self.params

        # Calculate coefficients A, B, C for the quadratic equation
        A = a * n1**2 + b * n2**2 + c * n3**2 + 2 * d * n1 * n2 + 2 * e * n1 * n3 + 2 * f * n3 * n2
        B = 2 * (
            a * x * n1
            + b * y * n2
            + c * z * n3
            + d * x * n2
            + d * y * n1
            + e * x * n3
            + e * z * n1
            + f * z * n2
            + f * y * n3
            + g * n1
            + h * n2
            + i * n3
        )
        C = j + a * x**2 + b * y**2 + c * z**2 + 2 * (i * z + h * y + g * x + d * x * y + e * x * z + f * z * y)

        # Discriminant
        D = B**2 - 4 * A * C

        is_inside = False

        if D < 0:  # No intersection
            p1, p2 = np.nan, np.nan
            t1, t2 = np.nan, np.nan
        elif D == 0:  # One intersection point
            t1 = -B / (2 * A)
            p1 = line.p + t1 * line.dir
            d1 = np.sign(t1) * np.linalg.norm(p1 - line.p)  # assigning the correct sign
            p2 = np.nan
            d2 = np.nan
        else:  # Two intersection points
            t1 = (-B + np.sqrt(D)) / (2 * A)
            t2 = (-B - np.sqrt(D)) / (2 * A)

            p1 = line.p + t1 * line.dir
            p2 = line.p + t2 * line.dir

            d1 = np.sign(t1) * np.linalg.norm(p1 - line.p)  # assigning the correct sign
            d2 = np.sign(t2) * np.linalg.norm(p2 - line.p)  # assigning the correct sign

            ps = np.column_stack((p1, p2))
            distances = np.asarray([d1, d2])

            if d1 < 0 and d2 > 0:
                is_inside = True
            elif d1 > 0 and d2 < 0:
                is_inside = True

            pi = np.argmin(abs(distances))
            p1 = ps[:, pi]
            p2 = ps[:, 1 - pi]
            d1 = distances[pi]
            d2 = distances[1 - pi]

        return p1, p2, d1, d2, is_inside

    def distance_point_center(self, point):

        return np.linalg.norm(point - self.center)

    def distance_point_surface(self, point):
        """
        Computes the shortest distance from a point to the surface of an ellipsoid.

        Parameters:
        -----------
        p : ndarray (3,)
            The 3D point in space.

        Returns:
        --------
        float
            The shortest distance from the point to the ellipsoid surface.
        """
        e_vecs = np.concatenate([self.e_vec1, self.e_vec2, self.e_vec3]).reshape(3, 3)  # 3x3 eigenvector matrix

        # Transform point to local ellipsoid coordinates
        p_local = np.dot(e_vecs.T, (point - self.center))

        # Function to solve for lambda (scaling factor)
        def scale_equation(lmbda):
            scaled = p_local / (1 + lmbda)
            return np.sum((scaled / self.radii) ** 2) - 1

        # Solve for λ numerically
        lambda_solution = fsolve(scale_equation, 0)[0]

        # Compute closest point on ellipsoid in local space
        closest_local = p_local / (1 + lambda_solution)

        # Transform back to global coordinates
        closest_global = np.dot(e_vecs, closest_local) + self.center

        # Compute Euclidean distance from point to closest surface point
        return np.linalg.norm(point - closest_global)
