import pandas as pd
from copy import deepcopy
from os import path
import warnings
from cryocat.utils import ioutils
from pathlib import PureWindowsPath


class Mdoc:
    """Class for reading, writing, and manipulating Mdoc files."""

    def __init__(self, input_path=None, titles=None, project_info=None, imgs=None, section_id="ZValue"):
        """
        Parameters
        ----------
        input_path : str, optional
            Path to an existing ``.mdoc`` file to read.  When supplied and the
            file exists, all other arguments are ignored.
        titles : list of str, optional
            Header title lines (without the surrounding ``[`` ``]`` brackets).
        project_info : dict, optional
            Key–value pairs from the mdoc header (e.g. ``PixelSpacing``,
            ``Voltage``).
        imgs : pandas.DataFrame, optional
            Per-image metadata; one row per tilt image.
        section_id : str, default='ZValue'
            Column name used as the section identifier (``'ZValue'`` or
            ``'FrameSet'``).
        """
        if input_path and path.isfile(input_path):
            self.input_path = input_path
            self.titles, self.project_info, self.imgs, self.section_id = self._read_mdoc(input_path)
        else:
            self.titles = titles
            self.project_info = project_info
            self.imgs = imgs
            self.section_id = section_id

    def write(self, out_path=None, overwrite=False, removed=False):
        """Write the Mdoc data to a file.

        Parameters
        ----------
        out_path : str, optional
            Output file path.  Defaults to ``self.file_path`` when ``None``.
        overwrite : bool, default=False
            Allow overwriting an existing file.
        removed : bool, default=False
            When ``True``, images marked as removed are also written out.

        Raises
        ------
        FileExistsError
            If the output file already exists and ``overwrite`` is ``False``.
        """
        if not out_path:
            out_path = self.file_path
        if path.isfile(out_path) and not overwrite:
            raise FileExistsError("File {} already exists. Set overwrite=True to overwrite.".format(out_path))

        with open(out_path, "w") as f:
            # write header
            for key, value in self.project_info.items():
                f.write("{} = {}\n".format(key, value))
            f.write("\n")
            for title in self.titles:
                f.write("[{}]\n".format(title))
                f.write("\n")

            # write images
            for index, row in self.imgs.iterrows():
                if removed or (not removed and not row["Removed"]):
                    f.write("[{} = {}]\n".format(self.section_id, row[self.section_id]))
                    for column in self.imgs.columns:
                        if (column != self.section_id) and (column != "Removed"):
                            f.write("{} = {}\n".format(column, row[column]))
                    f.write("\n")

    def add_field(self, field_name, field_value):
        """Add or overwrite a column in the per-image DataFrame.

        Parameters
        ----------
        field_name : str
            Column name to add or update.
        field_value : scalar or array-like
            Value(s) to assign; broadcast rules follow pandas conventions.
        """
        self.imgs[field_name] = field_value

    def sort_by_tilt(self, reset_z_value=False):
        """Sort images by tilt angle in ascending order.

        Parameters
        ----------
        reset_z_value : bool, default=False
            When ``True``, the ``ZValue`` column is reset to a sequential
            integer range after sorting.
        """
        self.imgs = self.imgs.sort_values(by="TiltAngle")
        if reset_z_value:
            self.imgs["ZValue"] = range(self.imgs.shape[0])

    def remove_image(self, index):
        """Mark a single image as removed.

        Parameters
        ----------
        index : int
            DataFrame index of the image to mark.
        """
        self.imgs.loc[index, "Removed"] = True

    def remove_images(self, indices, kept_only=True):
        """Mark multiple images as removed.

        Parameters
        ----------
        indices : array-like of int
            Positional indices (within the kept or full image list) to mark.
        kept_only : bool, default=True
            When ``True``, ``indices`` are interpreted relative to the
            currently kept (non-removed) images only.
        """
        if kept_only:
            kept_indices = self.kept_images().index
        else:
            kept_indices = self.imgs.index
        for index in indices:
            index = kept_indices[index]
            self.remove_image(index)

    def removed_images(self):
        """Return the subset of images that have been marked as removed.

        Returns
        -------
        pandas.DataFrame
            Rows from ``self.imgs`` where ``Removed`` is ``True``.
        """
        return self.imgs[self.imgs["Removed"] == True]

    def kept_images(self):
        """Return the subset of images that have not been removed.

        Returns
        -------
        pandas.DataFrame
            Rows from ``self.imgs`` where ``Removed`` is ``False``.
        """
        return self.imgs[self.imgs["Removed"] == False]

    def keep_images(self, indices):
        """Mark multiple images as kept (un-remove them).

        Parameters
        ----------
        indices : array-like of int
            DataFrame indices of images to un-remove.
        """
        self.imgs.loc[indices, "Removed"] = False

    def reset_images(self):
        """Mark all images as kept by clearing the ``Removed`` flag."""
        self.imgs["Removed"] = False

    def keep_image(self, index):
        """Mark a single image as kept (un-remove it).

        Parameters
        ----------
        index : int
            DataFrame index of the image to un-remove.
        """
        self.keep_images([index])

    def get_image(self, index):
        """Return a single image row by its DataFrame index.

        Parameters
        ----------
        index : int
            DataFrame index of the image.

        Returns
        -------
        pandas.Series
            The image row.
        """
        return self.imgs.loc[index]

    def get_images(self, indices):
        """Return multiple image rows by their DataFrame indices.

        Parameters
        ----------
        indices : array-like of int
            DataFrame indices of the images to retrieve.

        Returns
        -------
        pandas.DataFrame
            The selected rows.
        """
        return self.imgs.loc[indices]

    def get_image_by_zvalue(self, zvalue):
        """Return the image row(s) with a specific ZValue.

        Parameters
        ----------
        zvalue : int
            The ZValue to look up.

        Returns
        -------
        pandas.DataFrame
            Matching image row(s).
        """
        return self.imgs[self.imgs["ZValue"] == zvalue]

    def get_images_by_zvalues(self, zvalues):
        """Return image rows matching any of the given ZValues.

        Parameters
        ----------
        zvalues : array-like of int
            ZValues to include.

        Returns
        -------
        pandas.DataFrame
            Matching image rows.
        """
        return self.imgs[self.imgs["ZValue"].isin(zvalues)]

    def get_image_by_zvalue_range(self, zvalue_min, zvalue_max):
        """Return image rows whose ZValue falls within ``[zvalue_min, zvalue_max]``.

        Parameters
        ----------
        zvalue_min : int
            Lower bound (inclusive).
        zvalue_max : int
            Upper bound (inclusive).

        Returns
        -------
        pandas.DataFrame
            Matching image rows.
        """
        return self.imgs[(self.imgs["ZValue"] >= zvalue_min) & (self.imgs["ZValue"] <= zvalue_max)]

    def get_images_by_zvalue_ranges(self, zvalue_ranges):
        """Return image rows matching any of the given ZValue ranges.

        Parameters
        ----------
        zvalue_ranges : list of tuple of int
            Each element is ``(zvalue_min, zvalue_max)`` (both inclusive).

        Returns
        -------
        pandas.DataFrame
            Concatenated rows from all matching ranges (index reset).
        """
        imgs = pd.DataFrame(columns=self.imgs.columns)
        for zvalue_min, zvalue_max in zvalue_ranges:
            imgs = pd.concat([imgs, self.get_image_by_zvalue_range(zvalue_min, zvalue_max)], ignore_index=True)
        return imgs

    def get_image_feature(self, column_name):
        """Return a single column from the image DataFrame.

        Parameters
        ----------
        column_name : str
            Column name.

        Returns
        -------
        pandas.Series
            The requested column.
        """
        return self.imgs[column_name]

    def get_image_features(self, features):
        """Return multiple columns from the image DataFrame.

        Parameters
        ----------
        features : list of str
            Column names to return.

        Returns
        -------
        pandas.DataFrame
            The selected columns.
        """
        return self.imgs[features]

    def reorder_images(self, indices):
        """Reorder the image DataFrame to the given index order.

        Parameters
        ----------
        indices : array-like of int
            New row order specified as DataFrame index values.
        """
        self.imgs = self.imgs.reindex(indices)
        self.imgs.reset_index(drop=True, inplace=True)

    def change_frame_path(self, new_path=None):
        """Update the ``SubFramePath`` column to use a new directory.

        Strips the existing directory from each path (keeping only the
        filename) and optionally prepends ``new_path``.

        Parameters
        ----------
        new_path : str, optional
            New directory to prepend to each frame filename.  When ``None``
            or empty, paths are reduced to filenames only.
        """
        def add_new_path(input_path):
            return path.join(new_path, input_path)

        self.imgs["SubFramePath"] = self.imgs["SubFramePath"].apply(path.basename)

        if new_path is not None and new_path != "":
            self.imgs["SubFramePath"] = self.imgs["SubFramePath"].apply(add_new_path)

    def update_pixel_size(self, new_pixel_size):
        """Update the pixel size in both the project info and per-image data.

        Parameters
        ----------
        new_pixel_size : float
            New pixel spacing value to write into ``PixelSpacing``.
        """
        self.project_info["PixelSpacing"] = new_pixel_size
        self.imgs["PixelSpacing"] = new_pixel_size

    def convert_section_type(self, new_section_id="FrameSet"):
        """Convert the mdoc section type between ``ZValue`` and ``FrameSet``.

        Renames the section-ID column, rebuilds the project-info header, and
        updates ``self.section_id`` and ``self.titles`` accordingly.

        Parameters
        ----------
        new_section_id : str, default='FrameSet'
            Target section type; must be ``'ZValue'`` or ``'FrameSet'``.

        Raises
        ------
        ValueError
            If ``new_section_id`` is neither ``'ZValue'`` nor ``'FrameSet'``.
        """
        if self.section_id == new_section_id:
            warnings.warn("The new section id is the same as the existing one - no changes were made.")
            return

        self.imgs.rename(columns={self.section_id: new_section_id}, inplace=True)

        if new_section_id == "ZValue":

            _, img_path = path.split(self.imgs["SubFramePath"].values[0])
            img_file = img_path.split("_")[0] + "_" + img_path.split("_")[1] + ".mrc"

            new_project_info = {
                "PixelSpacing": self.imgs["PixelSpacing"].values[0],
                "Voltage": self.project_info["Voltage"],
                # "Version": project_info["T"],
                "ImageFile": img_file,
                "ImageSize": self.imgs["UncroppedSize"].values[0].replace("-", ""),
                "DataMode": 1,
            }
            new_titles = ["T = " + self.project_info["T"]]
            second_title = (
                "T = Tilt axis angle = "
                + str(self.imgs["RotationAngle"].values[0] - 90)
                + ", binning = "
                + str(self.imgs["Binning"].values[0])
                + " spot = "
                + str(self.imgs["SpotSize"].values[0])
                + " camera = "
                + str(self.imgs["CameraIndex"].values[0])
            )
            new_titles.append(second_title)
        elif new_section_id == "FrameSet":
            t_entry = self.titles[0].split(" ")
            new_project_info = {
                "T": " ".join(t_entry[2:]),
                "Voltage": self.project_info["Voltage"],
            }
            new_titles = []
        else:
            raise ValueError("Currently onlyt conversion between ZValue and FrameSet is supported.")

        self.section_id = new_section_id
        self.titles = new_titles
        self.project_info = new_project_info

    @staticmethod
    def _read_mdoc(input_path):
        with open(input_path, "r") as f:
            lines = f.readlines()

            # separate first part of lines until a first occurrence of a line starting with "[ZValue"
            header = []  # list of header lines
            section_id = None #init
            for line in lines:
                if line.startswith("[ZValue"):
                    section_id = "ZValue"
                    break
                elif line.startswith("[FrameSet"):
                    section_id = "FrameSet"
                    break
                # append only non-empty lines
                if line.strip():
                    header.append(line.strip())

            if section_id is None:
                section_id = "ZValue"

            titles, project_info = Mdoc._parse_header(header)

            # continue after header
            data = lines[lines.index(line) :]
            imgs = Mdoc._parse_images(data, section_id)

            return titles, project_info, imgs, section_id

    @staticmethod
    def _parse_header(header):
        titles = []
        project_info = {}
        for line in header:
            if line.startswith("["):
                title = line.strip("[").strip("]").strip()
                titles.append(title)
                # TODO parse the title ?
            else:
                key, value = line.split("=")
                project_info[key.strip()] = Mdoc._format_value(value)

        return titles, project_info

    @staticmethod
    def _parse_images(data, section_id):
        # split the lines into sections, each starting with line starting with "[ZValue"
        sections = []
        section = []
        for line in data:
            if line.startswith("[" + section_id) and section:
                sections.append(section)
                section = []
            if line.strip():
                section.append(line)
        sections.append(section)

        # determine dataframe columns from the first section
        columns = [section_id]
        columns.extend([line.split("=")[0].strip() for line in sections[0][1:]])

        imgs = pd.DataFrame(columns=columns)
        for section in sections:
            # parse section
            img = {}
            for line in section:
                if line.startswith("["):
                    img[section_id] = line.split("=")[1].strip().strip("]").strip()
                else:
                    key, value = line.split("=")
                    img[key.strip()] = Mdoc._format_value(value)
            imgs = pd.concat([imgs, pd.DataFrame(img, index=[0])], ignore_index=True)

        # prepare flag for removed images
        imgs["Removed"] = False

        # convert ZValues to int
        try:
            imgs[section_id] = pd.to_numeric(imgs[section_id], errors='coerce') #convert to numeric, coercing to naN
            imgs[section_id] = imgs[section_id].astype('Int64')
        except ValueError:
            pass


        # convert TiltAngle to float
        if "TiltAngle" in imgs.columns:
            imgs["TiltAngle"] = imgs["TiltAngle"].astype(float)

        return imgs

    @staticmethod
    def _format_value(value):
        if value.strip().isdigit():
            formatted = int(value.strip())
        elif value.strip().replace(".", "", 1).isdigit():
            formatted = float(value.strip())
        else:
            formatted = value.strip()
        return formatted


def remove_images(input_mdoc, idx_to_remove, numbered_from_1=True, output_path=None):
    """Remove images from an mdoc file by index.

    Parameters
    ----------
    input_mdoc : str
        Path to the input ``.mdoc`` file.
    idx_to_remove : str or array-like of int
        Indices of images to remove, or a path to a file containing them.
    numbered_from_1 : bool, default=True
        When ``True``, indices are treated as 1-based.
    output_path : str, optional
        If given, the modified mdoc is written to this path.

    Returns
    -------
    Mdoc
        The modified :class:`Mdoc` object.
    """
    mdoc = Mdoc(input_mdoc)
    idx_to_remove_final = ioutils.indices_load(idx_to_remove, numbered_from_1=numbered_from_1)
    if idx_to_remove_final is not None:
        mdoc.remove_images(idx_to_remove_final)

        if output_path:
            mdoc.write(output_path, overwrite=True)

    return mdoc


def get_tilt_angles(input_mdoc, output_path=None):
    """Extract tilt angles from an mdoc file.

    Parameters
    ----------
    input_mdoc : str
        Path to the ``.mdoc`` file.
    output_path : str, optional
        If given, tilt angles are written as a single-column CSV to this path.

    Returns
    -------
    numpy.ndarray
        1-D array of tilt angles in degrees.
    """
    mdoc = Mdoc(input_mdoc)

    if output_path:
        mdoc.imgs["TiltAngle"].to_csv(output_path, index=False, header=False)

    return mdoc.imgs["TiltAngle"].values


def sort_mdoc_by_tilt_angles(input_mdoc, reset_z_value=False, output_path=None):
    """Sort an mdoc file by tilt angle.

    Parameters
    ----------
    input_mdoc : str
        Path to the ``.mdoc`` file.
    reset_z_value : bool, default=False
        When ``True``, the ``ZValue`` column is reset to a sequential range
        after sorting.
    output_path : str, optional
        If given, the sorted mdoc is written to this path.

    Returns
    -------
    Mdoc
        The sorted :class:`Mdoc` object.
    """
    mdoc = Mdoc(input_mdoc)
    mdoc.sort_by_tilt(reset_z_value=reset_z_value)

    if output_path:
        mdoc.write(output_path, overwrite=True)

    return mdoc


def split_mdoc_file(input_mdoc, new_id=None, output_folder=None):
    """Split an mdoc file into one :class:`Mdoc` object per image.

    Parameters
    ----------
    input_mdoc : str or Mdoc
        Path to the ``.mdoc`` file or an already-loaded :class:`Mdoc`.
    new_id : str, optional
        When given, the section type is converted to ``new_id`` before
        splitting (e.g. ``'FrameSet'``).
    output_folder : str, optional
        If given, each per-image mdoc is written to this directory, named
        after the ``SubFramePath`` basename with a ``.mdoc`` extension.

    Returns
    -------
    list of Mdoc
        One :class:`Mdoc` per image, in the original order.

    Raises
    ------
    ValueError
        If ``input_mdoc`` is neither a ``str`` nor a :class:`Mdoc`.
    """
    if isinstance(input_mdoc, str):
        full_mdoc = Mdoc(input_mdoc)
    elif isinstance(input_mdoc, Mdoc):
        full_mdoc = deepcopy(input_mdoc)
    else:
        raise ValueError("The specified input mdoc has to be of type str or Mdoc")

    if new_id is not None:
        full_mdoc.convert_section_type()
        if new_id == "FrameSet":
            full_mdoc.imgs["FrameSet"] = 0

    mdocs_all = []
    for e in range(full_mdoc.imgs.shape[0]):
        new_mdoc = Mdoc(
            titles=full_mdoc.titles,
            project_info=full_mdoc.project_info,
            imgs=pd.DataFrame(full_mdoc.imgs.iloc[e : e + 1]),
            section_id=full_mdoc.section_id,
        )
        if output_folder is not None:
            frames_file = PureWindowsPath(new_mdoc.imgs["SubFramePath"].values[0]).name # raw data typically saved in a Windows machine
            new_mdoc.write(output_folder + frames_file + ".mdoc", overwrite=True)

        mdocs_all.append(new_mdoc)

    return mdocs_all


def merge_mdoc_files(mdoc_path, new_id=None, reorder=True, stripFramePath=False, output_path=None):
    """Merge multiple mdoc files matching a path prefix into a single :class:`Mdoc`.

    Discovers all ``*.mdoc`` files in the same directory that share the
    given prefix, reads them in order, and concatenates their image data.

    Parameters
    ----------
    mdoc_path : str
        Prefix path used to discover sibling ``.mdoc`` files (directory +
        filename prefix).
    new_id : str, optional
        When given, the merged mdoc's section type is converted to this ID.
    reorder : bool, default=True
        When ``True``, images are sorted by tilt angle and ZValues are reset.
    stripFramePath : bool, default=False
        When ``True``, directory information is stripped from ``SubFramePath``.
    output_path : str, optional
        If given, the merged mdoc is written to this path.

    Returns
    -------
    Mdoc
        The merged :class:`Mdoc` object.
    """
    dir_only, prefix = path.split(mdoc_path)
    list_of_mdocs = ioutils.get_files_prefix_suffix(dir_path=dir_only, prefix=prefix, suffix=".mdoc")

    all_imgs = pd.DataFrame()
    for i, m in enumerate(list_of_mdocs):
        if i == 0:
            titles, project_info, imgs, section_id = Mdoc._read_mdoc(dir_only + "/" + m)
        else:
            _, _, imgs, _ = Mdoc._read_mdoc(dir_only + "/" + m)
        imgs[section_id] = i
        all_imgs = pd.concat([all_imgs, imgs])

    merged_mdoc = Mdoc(titles=titles, project_info=project_info, imgs=all_imgs, section_id=section_id)

    if new_id is not None:
        merged_mdoc.convert_section_type(new_section_id=new_id)

    if stripFramePath:
        merged_mdoc.change_frame_path()

    if reorder:
        merged_mdoc.sort_by_tilt(reset_z_value=True)

    if output_path is not None:
        merged_mdoc.write(out_path=output_path, overwrite=True)

    return merged_mdoc


def update_mdoc_features(mdoc_path, features_dict, output_path=None):
    """Update a specific entry in the mdoc file.

    Parameters
    ----------
    mdoc_path : str
        Path to the mdoc file to be updated.
    features_dict : dict
        Dictionary containing the features to be updated as keys and their new value.
    output_path : str, optional, default=None
        Path to save the updated mdoc file. If None, no updated file is saved.

    Returns
    -------
    Mdoc
        The updated Mdoc object.

    Raises
    ------
    KeyError
        If the specified feature is not found in the project info or image data.
    """
    mdoc = Mdoc(mdoc_path)

    for feature, value in features_dict.items():
        if feature in mdoc.project_info.keys():
            mdoc.project_info[feature] = value
            if feature in mdoc.imgs.columns:
                mdoc.imgs[feature] = value
        elif feature in mdoc.imgs.columns:
            mdoc.imgs[feature] = value
        else:
            raise KeyError(f"Entry '{feature}' not found in project info or image data.")

    if output_path:
        mdoc.write(output_path, overwrite=True)

    return mdoc
