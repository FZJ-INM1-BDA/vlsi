from typing import ContextManager, Callable
import gzip
from io import BufferedReader
import struct
import json

import numpy as np
import nibabel as nib

from .base import SpatialIndex, V0SpatialIndex
from .util import ALIAS_SUFFIX


class ReadableSpatialIndex(SpatialIndex):

    readable = True

    def __init__(
        self,
        filepath,
        *,
        reader_read: Callable[[str], ContextManager[BufferedReader]] = lambda p: open(
            p, mode="rb"
        ),
    ):
        super().__init__(filepath, mode="r")

        self.reader_read = reader_read

        with self.reader_read(filepath) as fp:
            meta = fp.read().decode()

        if meta.startswith(V0SpatialIndex.HEADER):
            self.VOXEL_SUFFIX = V0SpatialIndex.VOXEL_SUFFIX
            self.ATTR_SUFFIX = V0SpatialIndex.ATTR_SUFFIX

        voxel_filepath = str(self.filepath) + self.VOXEL_SUFFIX
        with self.reader_read(voxel_filepath) as fp:
            voxel_file_bytes = fp.read(-1)
            try:
                voxel_file_bytes = gzip.decompress(voxel_file_bytes)
            except gzip.BadGzipFile:
                ...
            self._nii = nib.Nifti1Image.from_bytes(voxel_file_bytes)

        self.dataobj = np.array(self._nii.dataobj)
        assert (
            self._nii.get_data_dtype() == np.uint64
        ), f"Expected to be of type uint64, but was {self._nii.get_data_dtype()}"

    @property
    def affine(self):
        return self._nii.affine

    def read(self, pos):
        phys2vox = np.linalg.inv(self.affine)

        coordinate_count = len(pos)
        hom = np.c_[np.array(pos), np.ones(coordinate_count)]
        coords = np.dot(hom, phys2vox.T)[:, :3]

        pos = self.validate_pos(coords)
        x, y, z = pos.T

        result = []

        with self.reader_read(str(self.filepath) + self.ATTR_SUFFIX) as reader:
            for val in self.dataobj[x, y, z].tolist():
                offset = val >> 32
                bytes_to_read = val & self.UINT32_MAX
                reader.seek(int(offset))
                decoded = reader.read(int(bytes_to_read))
                if len(decoded) == 0:
                    continue
                result.append(decoded)
            return result


class ReadableSparseIndex(ReadableSpatialIndex):
    def __init__(self, filepath, *, reader_read=lambda p: open(p, mode="rb")):
        super().__init__(filepath, reader_read=reader_read)
        with reader_read(f"{filepath}{ALIAS_SUFFIX}") as fp:
            self.alias_label = json.loads(fp.read())

    def read_and_parse(self, pos):
        result: list[dict[str, float]] = []
        for buf in self.read(pos):
            assert (
                len(buf) % 10 == 0
            ), f"Expected buf to be multiple of 10, but was {len(buf)}"
            counter = 0
            res: dict[str, float] = {}
            result.append(res)
            while counter < len(buf):
                _b = buf[counter : counter + 10]

                mapv = _b[-4:]
                map_value = struct.unpack("<f", mapv)[0]
                regionalias = _b[:-4].decode()
                rname = self.alias_label.get(regionalias, regionalias)
                res[rname] = map_value
                counter += 10

        return result
