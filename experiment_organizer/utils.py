from struct import pack, unpack

import os
import numpy as np
import pandas as pd
import re


def get_sub_paths(parent_path, dir_only=False):
    """
    Returns a list of sub-paths to sub-directories and files.

    :param parent_path: the path to search for sub-paths
    :param dir_only: only return directory paths
    :returns: list of sub-paths
    """
    # ATTENTION: https://bugs.python.org/issue33105
    # os.path.isfile returns False on Windows when file path is longer
    # than 260 characters!
    # FIX: use extended-length path
    if os.name == 'nt' and re.search(r"^\\\\{2}[^\?]", parent_path):
        parent_path = os.path.abspath(parent_path).replace(
            "\\\\", "\\\\?\\UNC\\"
        )

    for child_name in os.listdir(parent_path):
        child_path = os.path.join(parent_path, child_name)
        if os.path.isfile(child_path):
            if not dir_only:
                yield child_path
        else:
            yield child_path  # + "/"


def init_empty_dataframe(columns, dtypes, index=None):
    df = pd.DataFrame(index=index)
    for c, d in zip(columns, dtypes):
        df[c] = pd.Series(dtype=d)
    return df


class DefaultDict(dict):
    def __init__(self, *args, **kwargs):
        self.update(*args, **kwargs)

    def get(self, key, try_default=True):
        if try_default:
            if key not in dict.keys(self) and 'default' in dict.keys(self):
                return dict.__getitem__(self, 'default')
        return dict.__getitem__(self, key)

    def __getitem__(self, key):
        return self.get_or_default(key)


class ATMatrix():
    # @see https://github.com/Dushistov/libspatialite/blob/
    #          156e902140c96dbcdc87409ac5824b1e6575ad84/src/gaiageo/gg_matrix.c
    MATRIX_MAGIC_START = 0x00
    MATRIX_MAGIC_DELIMITER = 0x3a
    MATRIX_MAGIC_END = 0xb3

    _matrix = None

    def _encode_matrix(self):
        (
            xx, xy, xz, xoff,
            yx, yy, yz, yoff,
            zx, zy, zz, zoff,
            w1, w2, w3, w4,
        ) = self._matrix.flatten()

        # we encode the byte object with a little-endian byte order
        # @see: '<' format character within struct's format string
        is_little_endian = True
        return pack(
            '<B?' + 'dB'*16,
            ATMatrix.MATRIX_MAGIC_START, is_little_endian,
            xx, ATMatrix.MATRIX_MAGIC_DELIMITER,
            xy, ATMatrix.MATRIX_MAGIC_DELIMITER,
            xz, ATMatrix.MATRIX_MAGIC_DELIMITER,
            xoff, ATMatrix.MATRIX_MAGIC_DELIMITER,
            yx, ATMatrix.MATRIX_MAGIC_DELIMITER,
            yy, ATMatrix.MATRIX_MAGIC_DELIMITER,
            yz, ATMatrix.MATRIX_MAGIC_DELIMITER,
            yoff, ATMatrix.MATRIX_MAGIC_DELIMITER,
            zx, ATMatrix.MATRIX_MAGIC_DELIMITER,
            zy, ATMatrix.MATRIX_MAGIC_DELIMITER,
            zz, ATMatrix.MATRIX_MAGIC_DELIMITER,
            zoff, ATMatrix.MATRIX_MAGIC_DELIMITER,
            w1, ATMatrix.MATRIX_MAGIC_DELIMITER,
            w2, ATMatrix.MATRIX_MAGIC_DELIMITER,
            w3, ATMatrix.MATRIX_MAGIC_DELIMITER,
            w4, ATMatrix.MATRIX_MAGIC_END
        )

    def _decode_binary(self, binary):
        if len(binary) != 146:
            raise ValueError("Unexpected binary data found!")

        (
            matrix_magic_start, is_little_endian,
            xx, _, xy, _, xz, _, xoff, _,
            yx, _, yy, _, yz, _, yoff, _,
            zx, _, zy, _, zz, _, zoff, _,
            w1, _, w2, _, w3, _, w4, matrix_magic_end
        ) = unpack('<B?' + 'dB'*16, binary)

        if (
            matrix_magic_start != ATMatrix.MATRIX_MAGIC_START or
            matrix_magic_end != ATMatrix.MATRIX_MAGIC_END
        ):
            raise ValueError("Unexpected binary data found!")

        if is_little_endian is not True:
            raise ValueError("Unexpected endianness found!")

        return np.array([
            [xx, xy, xz, xoff],
            [yx, yy, yz, yoff],
            [zx, zy, zz, zoff],
            [w1, w2, w3, w4]
        ], dtype=np.double)

    def __init__(self, data=None):
        if data is None:
            self._matrix = np.array([  # identity
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ], dtype=np.double)
        elif isinstance(data, bytes):
            self._matrix = self._decode_binary(data)
        elif isinstance(data, ATMatrix):
            self._matrix = ATMatrix.as_ndarray().copy()
        elif isinstance(data, (np.ndarray, np.generic)):
            self._matrix = data.copy()
        elif isinstance(data, (tuple, list)):
            self._matrix = np.array([
                [data[0], data[1], data[2], data[3]],
                [data[4], data[5], data[6], data[7]],
                [data[8], data[9], data[10], data[11]],
                [data[12], data[13], data[14], data[15]]
            ], dtype=np.double)
        else:
            raise RuntimeError(
                (
                    "The data '%s' cannot be parsed as ATMatrix! "
                    "Unknown data type."
                ) % (data,)
            )

    def as_2darray(self):
        (
            xx, xy, xz, xoff,
            yx, yy, yz, yoff,
            zx, zy, zz, zoff,
            w1, w2, w3, w4,
        ) = self._matrix.flatten()
        return np.array([
            [xx, xy, xoff],
            [yx, yy, yoff],
            [w1, w2, w4]
        ], dtype=np.double)

    def as_3darray(self):
        return self._matrix

    def as_binary(self):
        return self._encode_matrix()
