from pathlib import Path
from collections import defaultdict
from collections.abc import Iterable

import numpy as np
import nibabel as nib

from .base import SpatialIndex
from .util import logger


class SpatialIndexWriteExc(Exception):
    pass


class WritableSpatialIndex(SpatialIndex):

    writable = True

    _buffer: dict[tuple[int, int, int], bytes] = defaultdict(bytes)

    def __init__(self, filepath):
        super().__init__(filepath, mode="w")
        self.closed = False

    def write(self, pos, data):
        if self.closed:
            raise SpatialIndexWriteExc("WritableIndex already closed.")
        pos = self.validate_pos(pos)
        assert isinstance(
            data, Iterable
        ), f"Expecting data to be instance of Iterable, but was not: data={data}"
        for p, datum in zip(pos.tolist(), data):
            assert isinstance(
                datum, bytes
            ), f"Expecting datum to be of type bytes, but was not: datum={datum}"
            self._buffer[tuple(p)] += datum

    def save(self, affine=np.eye(4), shape: tuple[int, int, int] = None):
        if self.closed:
            raise SpatialIndexWriteExc("WritableIndex already closed.")

        voxel_file = str(self.filepath) + self.VOXEL_SUFFIX
        attr_file = str(self.filepath) + self.ATTR_SUFFIX
        manifestfile_file = str(self.filepath)

        if Path(voxel_file).exists():
            raise SpatialIndexWriteExc(
                f"File {voxel_file} already exist. Flushing incomplete files not yet supported"
            )

        if Path(attr_file).exists():
            raise SpatialIndexWriteExc(
                f"File {attr_file} already exist. Flushing incomplete files not yet supported"
            )

        total_bytes = 0
        for buf in self._buffer.values():
            total_bytes += len(buf)

        if total_bytes > self.UINT32_MAX:
            raise SpatialIndexWriteExc(
                "Spatial index data > uint32max. Currently not supported"
            )

        X, Y, Z = np.array(list(self._buffer.keys())).T
        minshape = np.array([(np.max(X) + 1, np.max(Y) + 1, np.max(Z) + 1)])

        if shape:
            shape = np.array(shape)
            assert shape.shape == (
                3,
            ), f"Expected shape have of shape (3,), but got {shape.shape}"
            assert np.all(
                shape >= minshape
            ), f"Expecting provided shape {shape} >= minshape {minshape}"
        else:
            shape = minshape

        # shape needs to be 1 + max index
        arr = np.zeros(shape, dtype=np.uint64)
        _offset_counter = 0

        total_count = len(self._buffer)
        logger.debug(f"Total buffer items: {total_count}")
        with open(attr_file, "wb") as fp:

            for idx, (key, _towrite) in enumerate(self._buffer.items()):
                if idx % 10 == 0:
                    logger.debug(
                        f"Progress: {idx} / {total_count} (updated every 10 items)"
                    )

                # as amazing it appears, bitshift has lower priority than addition
                # e.g. python -c "print(1 << 1 + 1)" prints 4, rather than 3
                arr[key] = np.uint64((_offset_counter << 32) + len(_towrite))

                _offset_counter += len(_towrite)
                fp.write(_towrite)

        # needs to explicitly specify uint64 dtype
        img = nib.Nifti1Image(arr, affine=affine, dtype=np.uint64)
        nib.save(img, voxel_file)

        with open(manifestfile_file, "wb") as fp:
            fp.write(self.HEADER.encode("utf-8"))
        self.closed = True
