import pandas as pd
import os


class Mdoc:
    """
    Class for reading, writing, and manipulating Mdoc files.

    Attributes
    ----------
    file_path : str
        Path to the Mdoc file.
    titles : list
        List of titles in the Mdoc file.
    project_info : dict
        Dictionary of general project information.
    imgs : pandas.DataFrame
        Dataframe of images, one row = one image section in the mdoc file.

    Examples
    --------
    mdoc = Mdoc('path/to/mdoc/file')
    mdoc.remove_images([0, 1, 2, 3])
    mdoc.keep_images([1, 2])
    mdoc.reset_images()
    mdoc.get_images([0, 1, 2])
    mdoc.get_image_features(['ZValue', 'SubFramePath'])
    mdoc.write('path/to/output/file')
    """

    def __init__(self, file_path=None):
        self.file_path = file_path
        self.titles = []
        self.project_info = {}
        self.imgs = None

        if os.path.isfile(file_path):
            self.titles, self.project_info, self.imgs = self._read_mdoc(file_path)

    def write(self, out_path=None, overwrite=False, removed=False):
        if not out_path:
            out_path = self.file_path
        if os.path.isfile(out_path) and not overwrite:
            raise FileExistsError('File {} already exists. Set overwrite=True to overwrite.'.format(out_path))

        with open(out_path, 'w') as f:
            # write header
            for key, value in self.project_info.items():
                f.write('{} = {}\n'.format(key, value))
            f.write('\n')
            for title in self.titles:
                f.write('[{}]\n'.format(title))
                f.write('\n')

            # write images
            for index, row in self.imgs.iterrows():
                if removed or (not removed and not row['removed']):
                    f.write('[ZValue = {}]\n'.format(row['ZValue']))
                    for column in self.imgs.columns:
                        if (column != 'ZValue') and (column != 'removed'):
                            f.write('{} = {}\n'.format(column, row[column]))
                    f.write('\n')

    def remove_image(self, index, kept_only=True):
        if kept_only:
            kept_indices = self.kept_images().index
            index = kept_indices[index]
        self.imgs.loc[index, 'removed'] = True

    def remove_images(self, indices, kept_only=True):
        for index in indices:
            self.remove_image(index, kept_only)

    def removed_images(self):
        return self.imgs[self.imgs['removed'] == True]

    def kept_images(self):
        return self.imgs[self.imgs['removed'] == False]

    def keep_images(self, indices):
        self.imgs.loc[indices, 'removed'] = False

    def reset_images(self):
        self.imgs['removed'] = False

    def keep_image(self, index):
        self.keep_images([index])

    def get_image(self, index):
        return self.imgs.loc[index]

    def get_images(self, indices):
        return self.imgs.loc[indices]

    def get_image_by_zvalue(self, zvalue):
        return self.imgs[self.imgs['ZValue'] == zvalue]

    def get_images_by_zvalues(self, zvalues):
        return self.imgs[self.imgs['ZValue'].isin(zvalues)]

    def get_image_by_zvalue_range(self, zvalue_min, zvalue_max):
        return self.imgs[(self.imgs['ZValue'] >= zvalue_min) & (self.imgs['ZValue'] <= zvalue_max)]

    def get_images_by_zvalue_ranges(self, zvalue_ranges):
        imgs = pd.DataFrame(columns=self.imgs.columns)
        for zvalue_min, zvalue_max in zvalue_ranges:
            imgs = pd.concat([imgs, self.get_image_by_zvalue_range(zvalue_min, zvalue_max)], ignore_index=True)
        return imgs

    def get_image_feature(self, feature):
        return self.imgs[feature]

    def get_image_features(self, features):
        return self.imgs[features]

    @staticmethod
    def _read_mdoc(file_path):
        with open(file_path, 'r') as f:
            lines = f.readlines()

            # separate first part of lines until a first occurrence of a line starting with "[ZValue"
            header = []  # list of header lines
            for line in lines:
                if line.startswith('[ZValue'):
                    break
                # append only non-empty lines
                if line.strip():
                    header.append(line.strip())

            titles, project_info = Mdoc._parse_header(header)

            # continue after header
            data = lines[lines.index(line):]
            imgs = Mdoc._parse_images(data)

            return titles, project_info, imgs

    @staticmethod
    def _parse_header(header):
        titles = []
        project_info = {}
        for line in header:
            if line.startswith('['):
                title = line.strip('[').strip(']').strip()
                titles.append(title)
                # TODO parse the title ?
            else:
                key, value = line.split('=')
                project_info[key.strip()] = Mdoc._format_value(value)

        return titles, project_info

    @staticmethod
    def _parse_images(data):
        # split the lines into sections, each starting with line starting with "[ZValue"
        sections = []
        section = []
        for line in data:
            if line.startswith('[ZValue') and section:
                sections.append(section)
                section = []
            if line.strip():
                section.append(line)
        sections.append(section)

        # determine dataframe columns from the first section
        columns = ['ZValue']
        columns.extend([line.split('=')[0].strip() for line in sections[0][1:]])

        imgs = pd.DataFrame(columns=columns)
        for section in sections:
            # parse section
            img = {}
            for line in section:
                if line.startswith('['):
                    img['ZValue'] = line.split('=')[1].strip().strip(']').strip()
                else:
                    key, value = line.split('=')
                    img[key.strip()] = Mdoc._format_value(value)
            imgs = pd.concat([imgs, pd.DataFrame(img, index=[0])], ignore_index=True)

        # prepare flag for removed images
        imgs['removed'] = False

        return imgs

    @staticmethod
    def _format_value(value):
        if value.strip().isdigit():
            formatted = int(value.strip())
        elif value.strip().replace('.', '', 1).isdigit():
            formatted = float(value.strip())
        else:
            formatted = value.strip()
        return formatted
