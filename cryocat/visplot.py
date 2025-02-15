import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib import gridspec
from matplotlib import cm
import matplotlib.ticker as mticker


def get_colors_from_palette(num_colors, pallete_name="tab10"):
    """Generate a list of color codes in hexadecimal format from a specified color palette.

    Parameters
    ----------
    num_colors : int
        The number of distinct colors to generate.
    pallete_name : str, optional
        The name of the color palette to use (default is "tab10").

    Returns
    -------
    list of str
        A list containing the hexadecimal color codes.

    Examples
    --------
    >>> get_colors_from_palette(3)
    ['#1f77b4', '#ff7f0e', '#2ca02c']
    >>> get_colors_from_palette(5, pallete_name="viridis")
    ['#440154', '#3b528b', '#21918c', '#5ec962', '#fde725']
    """

    # Generate a colormap with the desired number of distinct colors
    cmap = plt.cm.get_cmap(pallete_name, num_colors)

    # Convert RGBA colors to hex format
    color_classes_hex = [colors.rgb2hex(cmap(i)) for i in range(num_colors)]

    return color_classes_hex


def convert_color_scheme(num_colors, color_scheme=None):

    if color_scheme is None:
        return get_colors_from_palette(num_colors)
    elif isinstance(color_scheme, "str"):
        return get_colors_from_palette(num_colors, pallete_name=color_scheme)
    elif isinstance(color_scheme, list):
        return color_scheme[:num_colors]


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
    ax1.scatter(theta_r_pos[:, 0], theta_r_pos[:, 1], c=dist_pos, s=marker_size,cmap=colormap)
    ax1.set_xlabel("Northern hemisphere")

    ax2 = plt.subplot(gs[1], projection="polar")
    ax2.set_yticklabels([])
    ax2.scatter(theta_r_neg[:, 0], theta_r_neg[:, 1], c=dist_neg, s=marker_size,cmap=colormap)
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


def plot_class_occupancy(
    occupancy_dic, color_scheme=None, ax=None, show_legend=True, graph_title=None, output_file=None
):

    ax_provided = True
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        ax_provided = False
    color_classes = convert_color_scheme(num_colors=len(occupancy_dic), color_scheme=color_scheme)

    for i, c in enumerate(sorted(occupancy_dic)):
        ax.plot(range(1, len(occupancy_dic[c]) + 1), occupancy_dic[c], label="Class " + str(c), color=color_classes[i])

    ax.set_xlabel("Iteration")
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.set_xlim(1, len(occupancy_dic[c]))
    ax.set_ylabel("Number of particles")
    ax.set_ylim(
        0,
    )
    if graph_title is None:
        ax.set_title("Class occupancy progress")
    else:
        ax.set_title(graph_title)
    if show_legend:
        ax.legend(loc="upper left")

    if not ax_provided:
        plt.tight_layout()
        plt.show()

    if output_file is not None and not ax_provided:
        fig.savefig(output_file, dpi=fig.dpi)


def plot_class_stability(
    subtomo_changes, color_scheme=None, ax=None, show_legend=True, graph_title=None, output_file=None
):

    ax_provided = True
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        ax_provided = False

    color_classes = convert_color_scheme(num_colors=len(subtomo_changes), color_scheme=color_scheme)

    for cls, values in sorted(subtomo_changes.items()):
        ax.plot(range(1, len(values) + 1), values, label="Class " + str(cls), color=color_classes[cls - 1])

    ax.set_xlabel("Iteration")
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.set_xlim(1, len(values))
    ax.set_ylabel("Particles changing their class")
    ax.set_ylim(
        0,
    )
    if graph_title is None:
        ax.set_title("Stability of classes")
    else:
        ax.set_title(graph_title)
    if show_legend:
        ax.legend(ncol=2)

    if not ax_provided:
        plt.tight_layout()
        plt.show()

    if output_file is not None and not ax_provided:
        fig.savefig(output_file, dpi=fig.dpi)


def plot_classification_convergence(
    occupancy_dic, subtomo_changes_dic, color_scheme=None, graph_title=None, output_file=None
):

    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    plot_class_occupancy(occupancy_dic, color_scheme=color_scheme, ax=ax1, show_legend=False)
    plot_class_stability(subtomo_changes_dic, color_scheme=color_scheme, ax=ax2, show_legend=True)

    # Adjust layout
    plt.tight_layout()

    # Show the plot
    if graph_title is not None:
        fig.suptitle(graph_title, y=1.02)

    plt.show()

    if output_file is not None:
        fig.savefig(output_file, dpi=fig.dpi)


def plot_alignment_stability(input_dfs, labels=None, graph_title="Alignment stability", output_file=None):

    x_axis = np.arange(input_dfs[0].shape[0])

    n_rows = 3
    n_cols = 4

    # Set labels
    if labels is None:
        labels = input_dfs[0].columns
        show_legend = False
    else:
        show_legend = True

    # Create subplots
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(20, 10))

    for i in range(n_rows):  # number of rows
        for j in range(n_cols):  # number of columns
            current_ax = axes[i, j]
            df_idx = i * n_cols + j
            if df_idx >= input_dfs[0].shape[1]:
                continue
            for dfi, df in enumerate(input_dfs):
                current_ax.plot(x_axis, df.iloc[:, df_idx], label=f"{labels[dfi]}")
                current_ax.set_xlabel("Iteration")
                current_ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
                current_ax.set_ylabel(input_dfs[0].columns[df_idx])
                if show_legend:
                    current_ax.legend()

    # Show the plot
    fig.suptitle(graph_title)
    plt.show()

    if output_file is not None:
        fig.savefig(output_file, dpi=fig.dpi)


def scatter_with_histogram(
    data_x,
    data_y,
    bins_x=None,
    bins_y=None,
    colors_x=None,
    colors_y=None,
    edges_x=None,
    edges_y=None,
    axis_title_x=None,
    axis_title_y=None,
    output_file=None,
):

    if not bins_x:
        bins_x = 30
    if not bins_y:
        bins_y = 30

    x_bins = np.linspace(min(data_x), max(data_x), bins_x)
    y_bins = np.linspace(min(data_y), max(data_y), bins_y)

    def assign_colors(colors, edges):
        if not colors:
            colors = ["cornflowerblue"]
        else:
            if isinstance(colors, list):
                pass
            elif isinstance(colors, str) and edges:  # assuming colormap
                cmap = plt.get_cmap(colors, len(edges))
                colors = [cmap(i) for i in range(len(edges))]
            elif isinstance(colors, str):
                colors = [colors]
            else:
                raise ValueError(
                    f"The colors have to be either a list of colors, name of a single color or name of a colormap ."
                )

        return colors

    colors_x = assign_colors(colors=colors_x, edges=edges_x)
    colors_y = assign_colors(colors=colors_y, edges=edges_y)

    def get_color(value, colors, edges):
        if not edges:
            colors[0]
        else:
            for i, e in enumerate(edges):
                if value < e:
                    return colors[i]

    # Create figure and grid layout
    fig = plt.figure(figsize=(20, 10), dpi=300)
    gs = fig.add_gridspec(4, 4, hspace=0, wspace=0)

    # Scatter plot (main plot)
    ax_scatter = fig.add_subplot(gs[1:, :-1])
    ax_scatter.scatter(data_x, data_y, alpha=0.5)

    if axis_title_x:
        ax_scatter.set_xlabel(axis_title_x)
    if axis_title_y:
        ax_scatter.set_ylabel(axis_title_y)

    # Histogram on top (X distribution)
    ax_histx = fig.add_subplot(gs[0, :-1], sharex=ax_scatter)
    hist_x, bin_edges_x = np.histogram(data_x, bins=x_bins)

    for i in range(len(bin_edges_x) - 1):
        ax_histx.bar(
            bin_edges_x[i],
            hist_x[i],
            width=bin_edges_x[1] - bin_edges_x[0],
            color=get_color(bin_edges_x[i], colors_x, edges_x),
            alpha=0.7,
            align="edge",
            edgecolor="black",
        )

    ax_histx.set_ylabel("Count")
    ax_histx.xaxis.set_tick_params(labelbottom=False)  # Hide x labels

    # Histogram on right (Y distribution)
    ax_histy = fig.add_subplot(gs[1:, -1], sharey=ax_scatter)
    hist_y, bin_edges_y = np.histogram(data_y, bins=y_bins)

    for i in range(len(bin_edges_y) - 1):
        ax_histy.barh(
            bin_edges_y[i],
            hist_y[i],
            height=bin_edges_y[1] - bin_edges_y[0],
            color=get_color(bin_edges_y[i], colors_y, edges_y),
            alpha=0.7,
            align="edge",
            edgecolor="black",
        )

    ax_histy.set_xlabel("Count")
    ax_histy.yaxis.set_tick_params(labelleft=False)  # Hide y labels

    if output_file is not None:
        fig.savefig(output_file, dpi=fig.dpi)

    plt.show()
