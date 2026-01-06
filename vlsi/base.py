from pathlib import Path
from typing import Union, Literal, List, Dict, Tuple, ContextManager, Callable
from collections import defaultdict
from collections.abc import Iterable
import gzip
from contextlib import contextmanager
from functools import partial
from io import BufferedReader

import numpy as np
import nibabel as nib


@contextmanager
def open_context(filepath: str | Path, mode="rb"):
    try:
        with open(filepath, mode) as fp:
            yield fp
    finally:
        ...


class SpatialIndexWriteExc(Exception):
    pass


class SpatialIndex:

    UINT32_MAX = 4_294_967_295

    readable = False
    writable = False

    HEADER = """SPATIALINDEX-UTF8-V1"""
    VOXEL_SUFFIX = ".spatialindex.voxel.nii.gz"
    ATTR_SUFFIX = ".spatialindex.attr.bin"

    def __init__(self, filepath: Union[str, Path], mode: Literal["r", "w"] = "r"):
        self.filepath = filepath
        self.mode = mode

    @staticmethod
    def validate_pos(pos):
        pos = np.array(pos, dtype=np.uint32)
        assert (
            len(pos.shape) == 2
        ), f"pos needs to have len(pos.shape) to be 2, but found to be {len(pos.shape)}"
        assert (
            pos.shape[1] == 3
        ), f"expecting Nx3 array, but found to be Nx{pos.shape[1]}"
        return pos

    def read(self, pos: Union[List[List[int]], np.ndarray]) -> List[bytes]:
        raise NotImplementedError

    def write(self, pos: Union[List[List[int]], np.ndarray], data: List[bytes]):
        raise NotImplementedError

    def save(self):
        raise NotImplementedError


class WritableSpatialIndex(SpatialIndex):

    _buffer: Dict[Tuple[int, int, int], List[bytes]] = defaultdict(list)

    def write(self, pos, data):
        pos = self.validate_pos(pos)
        assert isinstance(
            data, Iterable
        ), f"Expecting data to be instance of Iterable, but was not: data={data}"
        for p, datum in zip(pos.tolist(), data):
            assert isinstance(
                datum, bytes
            ), f"Expecting datum to be of type bytes, but was not: datum={datum}"
            self._buffer[tuple(p)].append(datum)

    def flush(self):
        voxel_file = self.filepath.with_suffix(self.VOXEL_SUFFIX)
        attr_file = self.filepath.with_suffix(self.ATTR_SUFFIX)

        if voxel_file.exists():
            raise SpatialIndexWriteExc(
                f"File {voxel_file} already exist. Flushing incomplete files not yet supported"
            )

        if attr_file.exists():
            raise SpatialIndexWriteExc(
                f"File {attr_file} already exist. Flushing incomplete files not yet supported"
            )

        total_bytes = 0
        for arrbytes in self._buffer.values():
            for b in arrbytes:
                total_bytes += len(b)

        if total_bytes > self.UINT32_MAX:
            raise SpatialIndexWriteExc(
                f"Spatial index data > uint32max. Currently not supported"
            )

        X, Y, Z = np.array(list(self._buffer.keys())).T

        arr = np.zeros((np.max(X) - 1, np.max(Y) - 1, np.max(Z) - 1), dtype=np.uint64)

        _offset_counter = 0
        _towritearr = []

        for key, value in self._buffer.items():
            _towrite = b"".join(value)
            arr[key] = np.uint64(_offset_counter << 32 + len(_towrite))
            _offset_counter += len(_towrite)
            _towritearr.append(_towrite)

        img = nib.Nifti1Image(arr, affine=np.eye(4))
        nib.save(img, voxel_file)

        with open(attr_file, "wb") as fp:
            fp.write(b"".join(_towritearr))


class ReadableSpatialIndex(SpatialIndex):

    def __init__(
        self,
        filepath,
        mode="r",
        *,
        reader_read: Callable[[str], ContextManager[BufferedReader]] = lambda p: open(
            p, mode="rb"
        ),
    ):
        super().__init__(filepath, mode)

        self.reader_read = reader_read

        voxel_filepath = self.filepath + self.VOXEL_SUFFIX
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

        with self.reader_read(self.filepath + self.ATTR_SUFFIX) as reader:
            for val in self.dataobj[x, y, z].tolist():
                offset = val >> 32
                bytes_to_read = val & self.UINT32_MAX
                reader.seek(int(offset))
                decoded = reader.read(int(bytes_to_read))
                if len(decoded) == 0:
                    continue
                result.append(decoded)
            return result
