import struct
import os
import re
import pandas as pd
from dataclasses import dataclass
from cryocat import ioutils


@dataclass
class ImodHeader:
    # based on information here https://bio3d.colorado.edu/imod/betaDoc/binspec.html
    # there is discrepancy of 4 bytes somewhere in that description though
    name: str = "IMODv1.2"
    header_format = ""
    encoding: str = "utf-8"  # Default encoding

    @staticmethod
    def check_sequence(file_handler, control_sequence, encoding):
        control_bytes = bytes(control_sequence.encode(encoding))
        buffer_size = 4
        buffer = file_handler.read(buffer_size)

        while buffer:
            if buffer == control_bytes:
                file_handler.seek(-buffer_size, 1)  # Seek back by 4 bytes
                return file_handler.tell()
            else:
                next_bytes = file_handler.read(buffer_size)
                if not next_bytes:
                    break  # End of file reached
                buffer = next_bytes

        # If the sequence is not found, return -1 or handle accordingly
        return -1

    @classmethod
    def read_from_file(cls, file_handler, encoding):
        header = cls(encoding=encoding)
        file_position = ImodHeader.check_sequence(file_handler, header.name[0:4], encoding=encoding)

        if file_position == -1:
            pass
        else:
            header_size = struct.calcsize(header.header_format)
            header_data = file_handler.read(header_size)
            header = header.from_bytes(header_data, encoding)
            return header


@dataclass
class ContourHeader(ImodHeader):
    name: str = "CONT"
    psize: int = 0
    flags: int = 0
    time: int = 0
    surf: int = 0

    header_format = ">4s iiii"

    def to_bytes(self):
        values = (
            self.name.encode(self.encoding),
            self.psize,
            self.flags,
            self.time,
            self.surf,
        )
        return struct.pack(self.header_format, *values)

    @classmethod
    def from_bytes(cls, data, encoding):
        values = struct.unpack(cls.header_format, data)
        return cls(
            name=values[0].decode(encoding).rstrip("\0"),
            psize=values[1],
            flags=values[2],
            time=values[3],
            surf=values[4],
        )


@dataclass
class ObjectHeader(ImodHeader):
    name: str = "OBJT"
    extra_data: str = ""
    contsize: int = 0
    flags: int = 402653184
    axis: int = 0
    drawmode: int = 1
    red: float = 0.0
    green: float = 1.0
    blue: float = 0.0
    pdrawsize: int = 3
    symbol: int = 0  # 0 = circle, 1 = none, 2 = square, 3 = triangle, 4 = star
    symsize: int = 3
    linewidth2: int = 1
    linewidth: int = 1
    linesty: int = 0
    symflags: int = 0
    sympad: int = 0
    trans: int = 0
    meshsize: int = 0
    surfsize: int = 8

    # for conversion to and from bytes
    header_format = ">64s 68s i I ii fff i BBBBBBBB ii"

    def to_bytes(self):
        values = (
            self.name.encode(self.encoding),
            self.extra_data.encode(self.encoding),
            self.contsize,
            self.flags,
            self.axis,
            self.drawmode,
            self.red,
            self.green,
            self.blue,
            self.pdrawsize,
            self.symbol,
            self.symsize,
            self.linewidth2,
            self.linewidth,
            self.linesty,
            self.symflags,
            self.sympad,
            self.trans,
            self.meshsize,
            self.surfsize,
        )
        return struct.pack(self.header_format, *values)

    @classmethod
    def from_bytes(cls, data, encoding):
        values = struct.unpack(cls.header_format, data)
        return cls(
            name=values[0].decode(encoding).rstrip("\0"),
            contsize=values[2],
            flags=values[3],
            axis=values[4],
            drawmode=values[5],
            red=values[6],
            green=values[7],
            blue=values[8],
            pdrawsize=values[9],
            symbol=values[10],
            symsize=values[11],
            linewidth2=values[12],
            linewidth=values[13],
            linesty=values[14],
            symflags=values[15],
            sympad=values[16],
            trans=values[17],
            meshsize=values[18],
            surfsize=values[19],
        )


@dataclass
class ModelHeader(ImodHeader):
    name: str = "IMOD-NewModel"
    xmax: int = 4096
    ymax: int = 4096
    zmax: int = 2000
    objsize: int = 1
    flags: int = 62976
    drawmode: int = 1
    mousemode: int = 1
    blacklevel: int = 0
    whitelevel: int = 255
    xoffset: float = 0.0
    yoffset: float = 0.0
    zoffset: float = 0.0
    xscale: float = 1.0
    yscale: float = 1.0
    zscale: float = 1.0
    object: int = 1
    contour: int = 0
    point: int = 0
    res: int = 3
    thresh: int = 128
    pixsize: float = 1.0
    units: int = -9  # 0 = pixels, 3 = km, 1 = m, -2 = cm, -3 = mm, -6 = microns, -9 = nm, -10 = Angstroms, -12 = pm
    csum: int = 100000000  # Checksum storage. Used for autosave only.
    alpha: float = 0.0
    beta: float = 0.0
    gamma: float = 0.0

    # for conversion to bytes
    header_format = ">128s iiii I iiii fff fff iiiii f ii fff"

    def to_bytes(self):
        values = (
            self.name.encode(self.encoding),
            self.xmax,
            self.ymax,
            self.zmax,
            self.objsize,
            self.flags,
            self.drawmode,
            self.mousemode,
            self.blacklevel,
            self.whitelevel,
            self.xoffset,
            self.yoffset,
            self.zoffset,
            self.xscale,
            self.yscale,
            self.zscale,
            self.object,
            self.contour,
            self.point,
            self.res,
            self.thresh,
            self.pixsize,
            self.units,
            self.csum,
            self.alpha,
            self.beta,
            self.gamma,
        )
        return struct.pack(self.header_format, *values)

    @classmethod
    def from_bytes(cls, data, encoding):
        values = struct.unpack(cls.header_format, data)
        return cls(
            name=values[0].decode(encoding).rstrip("\0"),
            xmax=values[1],
            ymax=values[2],
            zmax=values[3],
            objsize=values[4],
            flags=values[5],
            drawmode=values[6],
            mousemode=values[7],
            blacklevel=values[8],
            whitelevel=values[9],
            xoffset=values[10],
            yoffset=values[11],
            zoffset=values[12],
            xscale=values[13],
            yscale=values[14],
            zscale=values[15],
            object=values[16],
            contour=values[17],
            point=values[18],
            res=values[19],
            thresh=values[20],
            pixsize=values[21],
            units=values[22],
            csum=values[23],
            alpha=values[24],
            beta=values[25],
            gamma=values[26],
        )


def read_mod_files(input_path, file_prefix="", file_suffix=".mod"):

    if os.path.isfile(input_path):
        m_df = read_mod_file(input_path)
        mod_filename_no_ext = os.path.splitext(os.path.basename(input_path))[0] #get filename without the extension
        pattern_id = r'\d+' #regex to get the tomo number from filename
        mod_id = re.search(pattern_id, mod_filename_no_ext).group(0) #get first occurrence of the tomo number in file name
        m_df["mod_id"] = mod_id #append mod_id column to dataframe object
        
        return m_df

    else:
        mod_files = ioutils.get_files_prefix_suffix(input_path, prefix=file_prefix, suffix=file_suffix)
        mod_df = pd.DataFrame()
        object_max = 0
        for m in mod_files:
            mod_id = m[len(file_prefix) : -len(file_suffix)] if file_suffix else m[len(file_prefix) :]
            m_df = read_mod_file(input_path + m)
            m_df["mod_id"] = mod_id
            m_df["object_id"] += object_max
            object_max = m_df["object_id"].max()
            mod_df = pd.concat([mod_df, m_df], ignore_index=True)

        return mod_df


def read_mod_file(file_path):

    encoding = ioutils.get_file_encoding(file_path)

    with open(file_path, "rb") as f:
        # Read 4 bytes at a time
        while True:
            segment = f.read(4)
            if not segment:
                break  # End of file

            # Check if the segment is not equal to "OBJT"
            if (
                segment == b"OBJT"
                or segment == b"CONT"
                or segment == b"VIEW"
                or segment == b"SIZE"
                or segment == b"IMAT"
                or segment == b"MESH"
                or segment == b"IMOD"
                or segment == b"MINX"
            ):
                segment = segment.decode(encoding)
                # print(f"Found segment: {segment}")

    with open(file_path, "rb") as f:
        # Read the 8-byte ID
        id_data = f.read(8)
        # if id_data != b'\x4F\x4A\x42\x54\x43\x4F\x4E\x54':  # Check for the expected ID
        #    print("Invalid ID. This may not be a valid binary model file.")
        #    return None

        mh = ModelHeader.read_from_file(f, encoding=encoding)
        objsize = mh.objsize
        points = []
        for o in range(objsize):
            oh = ObjectHeader.read_from_file(f, encoding=encoding)
            num_contours = oh.contsize
            object_radius = oh.pdrawsize

            for c in range(num_contours):
                ch = ContourHeader.read_from_file(f, encoding=encoding)
                num_points = ch.psize
                for _ in range(num_points):
                    point_format = ">fff"
                    pt_header_size = struct.calcsize(point_format)
                    pt_header_data = f.read(pt_header_size)
                    pt_header = struct.unpack(point_format, pt_header_data)
                    points.append((o + 1, c + 1, pt_header[0], pt_header[1], pt_header[2], object_radius))

        # Create a DataFrame
        df = pd.DataFrame(points, columns=["object_id", "contour_id", "x", "y", "z", "object_radius"])

    return df


def write_model_binary(df, filename):
    # Count the number of object IDs and contours
    num_objects = df["object_id"].nunique()

    # Open the file in binary write mode
    with open(filename, "wb") as file:
        # Write the magic bytes and version id
        file.write(b"IMODV1.2")

        # Create and write the header
        header = ModelHeader(objsize=num_objects)
        file.write(header.to_bytes())

        # Loop over unique object IDs
        for obj_id in df["object_id"].unique():
            object_df = df[df["object_id"] == obj_id]

            # Create and write the ObjectHeader
            obj_header = ObjectHeader(contsize=object_df["contour_id"].nunique())
            file.write(obj_header.to_bytes())

            # Loop over contours of the current object
            for contour_id, contour_data in object_df.groupby("contour_id"):
                # Create and write the ContourHeader
                contour_header = ContourHeader(psize=len(contour_data))
                file.write(contour_header.to_bytes())

                # Write x, y, z coordinates for each contour
                coordinates = contour_data[["x", "y", "z"]].values.astype(">f4")
                file.write(coordinates.tobytes())

        # Write the end of data marker
        file.write(b"IEOF")
