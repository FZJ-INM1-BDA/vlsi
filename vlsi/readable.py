from typing import ContextManager, Callable
import gzip
from io import BufferedReader

import numpy as np
import nibabel as nib

from .base import SpatialIndex, V0SpatialIndex


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
            nii = nib.Nifti1Image.from_bytes(voxel_file_bytes)

        self.dataobj = np.array(nii.dataobj)
        assert (
            nii.get_data_dtype() == np.uint64
        ), f"Expected to be of type uint64, but was {nii.get_data_dtype()}"

    def read(self, pos):
        pos = self.validate_pos(pos)
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
