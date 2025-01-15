import pandas as pd
from copy import deepcopy
from os import path
import warnings
from cryocat import ioutils


class Mdoc:
    """Class for reading, writing, and manipulating Mdoc files."""

    def __init__(self, file_path=None, titles=None, project_info=None, imgs=None, section_id="ZValue"):
        if file_path and path.isfile(file_path):
            self.file_path = file_path
            self.titles, self.project_info, self.imgs, self.section_id = self._read_mdoc(file_path)
        else:
            self.titles = titles
            self.project_info = project_info
            self.imgs = imgs
            self.section_id = section_id

    def write(self, out_path=None, overwrite=False, removed=False):
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
        self.imgs[field_name] = field_value

    def sort_by_tilt(self, reset_z_value=False):
        self.imgs = self.imgs.sort_values(by="TiltAngle")
        if reset_z_value:
            self.imgs["ZValue"] = range(self.imgs.shape[0])

    def remove_image(self, index, kept_only=True):
        if kept_only:
            kept_indices = self.kept_images().index
            index = kept_indices[index]
        self.imgs.loc[index, "Removed"] = True

    def remove_images(self, indices, kept_only=True):
        for index in indices:
            self.remove_image(index, kept_only)

    def removed_images(self):
        return self.imgs[self.imgs["Removed"] == True]

    def kept_images(self):
        return self.imgs[self.imgs["Removed"] == False]

    def keep_images(self, indices):
        self.imgs.loc[indices, "Removed"] = False

    def reset_images(self):
        self.imgs["Removed"] = False

    def keep_image(self, index):
        self.keep_images([index])

    def get_image(self, index):
        return self.imgs.loc[index]

    def get_images(self, indices):
        return self.imgs.loc[indices]

    def get_image_by_zvalue(self, zvalue):
        return self.imgs[self.imgs["ZValue"] == zvalue]

    def get_images_by_zvalues(self, zvalues):
        return self.imgs[self.imgs["ZValue"].isin(zvalues)]

    def get_image_by_zvalue_range(self, zvalue_min, zvalue_max):
        return self.imgs[(self.imgs["ZValue"] >= zvalue_min) & (self.imgs["ZValue"] <= zvalue_max)]

    def get_images_by_zvalue_ranges(self, zvalue_ranges):
        imgs = pd.DataFrame(columns=self.imgs.columns)
        for zvalue_min, zvalue_max in zvalue_ranges:
            imgs = pd.concat([imgs, self.get_image_by_zvalue_range(zvalue_min, zvalue_max)], ignore_index=True)
        return imgs

    def get_image_feature(self, feature):
        return self.imgs[feature]

    def get_image_features(self, features):
        return self.imgs[features]

    def reorder_images(self, indices):
        self.imgs = self.imgs.reindex(indices)
        self.imgs.reset_index(drop=True, inplace=True)

    def change_frame_path(self, new_path=None):

        def add_new_path(file_name):
            return path.join(new_path, file_name)

        self.imgs["SubFramePath"] = self.imgs["SubFramePath"].apply(path.basename)

        if new_path is not None and new_path != "":
            self.imgs["SubFramePath"] = self.imgs["SubFramePath"].apply(add_new_path)

    def update_pixel_size(self, new_pixel_size):
        self.project_info["PixelSpacing"] = new_pixel_size
        self.imgs["PixelSpacing"] = new_pixel_size

    def convert_section_type(self, new_section_id="FrameSet"):

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
    def _read_mdoc(file_path):
        with open(file_path, "r") as f:
            lines = f.readlines()

            # separate first part of lines until a first occurrence of a line starting with "[ZValue"
            header = []  # list of header lines
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
        temp_column = imgs.astype({section_id: int})
        imgs[section_id] = temp_column[section_id]

        # convert TiltAngle to float
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


def remove_images(input_mdoc, idx_to_remove, numbered_from_1=True, output_file=None):

    mdoc = Mdoc(input_mdoc)
    idx_to_remove_final = ioutils.indices_load(idx_to_remove, numbered_from_1=numbered_from_1)
    mdoc.remove_images(idx_to_remove_final)

    if output_file:
        mdoc.write(output_file, overwrite=True)

    return mdoc


def get_tilt_angles(input_mdoc, output_file=None):

    mdoc = Mdoc(input_mdoc)

    if output_file:
        mdoc.imgs["TiltAngle"].to_csv(output_file, index=False, header=False)

    return mdoc.imgs["TiltAngle"].values


def sort_mdoc_by_tilt_angles(input_mdoc, reset_z_value=False, output_file=None):

    mdoc = Mdoc(input_mdoc)
    mdoc.sort_by_tilt(reset_z_value=reset_z_value)

    if output_file:
        mdoc.write(output_file, overwrite=True)

    return mdoc


def split_mdoc_file(input_mdoc, new_id=None, output_folder=None):

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
            frames_file = path.split(new_mdoc.imgs["SubFramePath"].values[0])[-1]
            new_mdoc.write(output_folder + frames_file + ".mdoc", overwrite=True)

        mdocs_all.append(new_mdoc)

    return mdocs_all


def merge_mdoc_files(mdoc_path, new_id=None, reorder=True, stripFramePath=False, output_file=None):

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

    if output_file is not None:
        merged_mdoc.write(out_path=output_file, overwrite=True)

    return merged_mdoc
