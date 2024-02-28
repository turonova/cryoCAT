import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib import gridspec
from matplotlib import cm


def convert_to_radial(coordinates, replace_nan=True):
    r = np.linalg.norm(coordinates, axis=1)
    phi = np.arctan2(coordinates[:, 1], coordinates[:, 0])

    if replace_nan:
        np.nan_to_num(phi, copy=False, nan=0.0)

    return r, phi


def convert_to_spherical(coordinates):
    coord = np.around(coordinates, decimals=14)

    x = coord[:, 0]
    y = coord[:, 1]
    z = coord[:, 2]
    r = np.linalg.norm(coord, axis=1)

    theta = np.arccos(z / r)  # inclination
    phi = np.arctan2(y, x)  # azimuth

    # r2  = np.linalg.norm(coord[:,:2], axis=1)
    # phi = np.sign(y)*np.arccos(x/r2)

    np.nan_to_num(phi, copy=False, nan=0.0)
    np.nan_to_num(theta, copy=False, nan=0.0)

    return r, theta, phi


def project_lambert(coord):
    _, theta, phi = convert_to_spherical(coord)

    projection_xy_local = np.zeros((phi.shape[0], 2))
    projection_theta_r_local = np.zeros((phi.shape[0], 2))

    # dividing by sqrt(2) so that we're projecting onto a unit circle
    projection_xy_local[:, 0] = coord[:, 0] * (np.sqrt(2 / (1 + coord[:, 2])))
    projection_xy_local[:, 1] = coord[:, 1] * (np.sqrt(2 / (1 + coord[:, 2])))

    np.nan_to_num(projection_xy_local, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    # sperhical coordinates -- CAREFUL as per this wikipedia page: https://en.wikipedia.org/wiki/Lambert_azimuthal_equal-area_projection
    # the symbols for inclination and azimuth ARE INVERTED WITH RESPEST TO THE SPHERICAL COORDS!!!
    projection_theta_r_local[:, 0] = phi
    # HACK: doing math.pi - angle in order for the +z to be projected to 0,0
    projection_theta_r_local[:, 1] = 2 * np.cos((np.pi - theta) / 2)

    return projection_theta_r_local, projection_xy_local


def project_stereo(coord):
    _, theta, phi = convert_to_spherical(coord)

    projection_xy_local = np.zeros((phi.shape[0], 2))
    projection_theta_r_local = np.zeros((phi.shape[0], 2))
    projection_xy_local[:, 0] = coord[:, 0] / (1 - coord[:, 2])
    projection_xy_local[:, 1] = coord[:, 1] / (1 - coord[:, 2])

    # https://en.wikipedia.org/wiki/Stereographic_projection uses a different standard from the page on spherical coord Spherical_coordinate_system
    projection_theta_r_local[:, 0] = phi
    # HACK: doing math.pi - angle in order for the +z to be projected to 0,0
    projection_theta_r_local[:, 1] = np.sin(np.pi - theta) / (1 - np.cos(np.pi - theta))

    return projection_theta_r_local, projection_xy_local


def project_equidistant(coord):
    _, theta, phi = convert_to_spherical(coord)

    projection_xy_local = np.zeros((phi.shape[0], 2))
    projection_theta_r_local = np.zeros((phi.shape[0], 2))

    # https://en.wikipedia.org/wiki/Azimuthal_equidistant_projection
    # TODO: To be checked, but this looks like it should -- a straight down projection.

    projection_xy_local[:, 0] = (np.pi / 2 + phi) * np.sin(theta)
    projection_xy_local[:, 1] = -(np.pi / 2 + phi) * np.cos(theta)

    projection_theta_r_local[:, 0] = convert_to_radial(projection_xy_local)[1]  # phi
    projection_theta_r_local[:, 1] = convert_to_radial(projection_xy_local)[
        0
    ]  # np.cos(( np.pi + theta )/2) #np.cos( theta  - np.pi/2)

    return projection_theta_r_local, projection_xy_local


def project_points_on_sphere(coord, projection_type="stereo"):
    if projection_type == "stereo":
        theta_r, xy_proj = project_stereo(coord)
    elif projection_type == "lambert":
        theta_r, xy_proj = project_lambert(coord)
    elif projection_type == "equidistant":
        theta_r, xy_proj = project_equidistant(coord)

    return theta_r, xy_proj


def create_projection(coord, projection_type="stereo", split_into_hemispheres=True):
    if split_into_hemispheres:
        coord_pos = coord[coord[:, 2] >= 0]
        coord_neg = coord[coord[:, 2] < 0]

        coord_neg[:, 2] *= -1

        theta_r_pos, xy_proj_pos = project_points_on_sphere(coord_pos, projection_type)
        theta_r_neg, xy_proj_neg = project_points_on_sphere(coord_neg, projection_type)

        return theta_r_pos, xy_proj_pos, theta_r_neg, xy_proj_neg
    else:
        theta_r, xy_proj = project_points_on_sphere(coord, projection_type)

        return theta_r, xy_proj, [], []


def plot_polar_nn_distances(
    coordinates, distances, max_radius=None, marker_size=3, colormap="viridis_r", graph_title=None, output_file=None
):
    coord_sorted = coordinates[coordinates[:, 2].argsort()]
    dist_sorted = distances[coordinates[:, 2].argsort()]

    theta_r_pos, _, theta_r_neg, _ = create_projection(coord_sorted)

    dist_neg = dist_sorted[0 : theta_r_neg.shape[0]]
    dist_pos = dist_sorted[theta_r_neg.shape[0] :]

    if max_radius is None:
        max_radius = np.amax(np.hstack((theta_r_pos[:, 1], theta_r_neg[:, 1])))

    fig = plt.figure(figsize=(13, 5))
    gs = gridspec.GridSpec(1, 3, width_ratios=[12, 12, 1])
    ax1 = plt.subplot(gs[0], projection="polar")
    ax1.set_yticklabels([])
    ax1.scatter(theta_r_pos[:, 0], theta_r_pos[:, 1], c=dist_pos, s=marker_size)
    ax1.set_xlabel("Northern hemisphere")

    ax2 = plt.subplot(gs[1], projection="polar")
    ax2.set_yticklabels([])
    ax2.scatter(theta_r_neg[:, 0], theta_r_neg[:, 1], c=dist_neg, s=marker_size)
    ax2.set_xlabel("Southern hemisphere")

    ax3 = plt.subplot(gs[2])

    norm = colors.Normalize(vmin=np.amin(distances), vmax=np.amax(distances))
    sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
    plt.colorbar(sm, cax=ax3)

    if graph_title is not None:
        fig.suptitle(graph_title)
    plt.show()

    if output_file is not None:
        fig.savefig(output_file, dpi=fig.dpi)


def fill_wedge(r1, r2, theta1, theta2, theta_step, **kargs):
    # draw annular sector in polar coordinates
    theta = np.linspace(theta1, theta2, theta_step)
    cr1 = np.full_like(theta, r1)
    cr2 = np.full_like(theta, r2)
    plt.fill_between(theta, cr1, cr2, **kargs)


def create_smooth_polar_histogram(ax, histogram, hist_norm_value=None, colormap="viridis_r"):
    # unpack histogram
    h, x_bins, y_bins = histogram

    # create colormap
    space = np.linspace(0.0, 1.0, 100)
    rgb = cm.get_cmap(colormap)(space)

    # set normalization to hist max unless specified differently
    if hist_norm_value is None:
        hist_norm_value = np.amax(h)

    wedge_draw_step = int(360 / (len(x_bins) - 1))
    # fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
    for yi, r_start in enumerate(y_bins[:-1]):
        r_end = y_bins[yi + 1]
        for xi, theta_start in enumerate(x_bins[:-1]):
            theta_end = x_bins[xi + 1]
            color = rgb[int(h[xi, yi] / hist_norm_value * (len(space) - 1))]
            fill_wedge(r_start, r_end, theta_start, theta_end, wedge_draw_step, color=color)


def plot_orientational_distribution(
    coordinates,
    projection="stereo",
    graph_title=None,
    theta_bin=73,
    radius_bin=33,
    max_radius=None,
    colormap="viridis_r",
    output_file=None,
):
    theta_r_pos, _, theta_r_neg, _ = create_projection(coordinates, projection_type=projection)

    if max_radius is None:
        max_radius = np.amax(np.hstack((theta_r_pos[:, 1], theta_r_neg[:, 1])))

    radius_bins = np.linspace(0, max_radius, radius_bin)

    if theta_r_pos.shape[0] > 0:
        theta_bins_pos = np.linspace(np.amin(theta_r_pos[:, 0]), np.amin(theta_r_pos[:, 0]) + 2 * np.pi, theta_bin)
        hist_pos = np.histogram2d(theta_r_pos[:, 0], theta_r_pos[:, 1], bins=(theta_bins_pos, radius_bins))
        max_pos = np.amax(hist_pos[0])
    else:
        max_pos = 0

    if theta_r_neg.shape[0] > 0:
        theta_bins_neg = np.linspace(np.amin(theta_r_neg[:, 0]), np.amin(theta_r_neg[:, 0]) + 2 * np.pi, theta_bin)
        hist_neg = np.histogram2d(theta_r_neg[:, 0], theta_r_neg[:, 1], bins=(theta_bins_neg, radius_bins))
        max_neg = np.amax(hist_neg[0])
    else:
        max_neg = 0

    hist_max = max(max_pos, max_neg)
    # hist_max = np.amax(np.vstack((hist_pos[0], hist_neg[0])))

    fig = plt.figure(figsize=(13, 5))
    gs = gridspec.GridSpec(1, 3, width_ratios=[12, 12, 1])
    ax1 = plt.subplot(gs[0], projection="polar")
    ax1.set_yticklabels([])
    if theta_r_pos.shape[0] > 0:
        create_smooth_polar_histogram(ax1, hist_pos, hist_norm_value=hist_max)
    ax1.set_xlabel("Northern hemisphere")

    ax2 = plt.subplot(gs[1], projection="polar")
    ax2.set_yticklabels([])
    if theta_r_neg.shape[0] > 0:
        create_smooth_polar_histogram(ax2, hist_neg, hist_norm_value=hist_max)
    ax2.set_xlabel("Southern hemisphere")

    ax3 = plt.subplot(gs[2])

    norm = colors.Normalize(vmin=0, vmax=hist_max)
    sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
    plt.colorbar(sm, cax=ax3, aspect=1)

    if graph_title is not None:
        fig.suptitle(graph_title)

    plt.show()

    if output_file is not None:
        fig.savefig(output_file, dpi=fig.dpi)
