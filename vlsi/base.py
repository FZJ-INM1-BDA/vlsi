from pathlib import Path
from typing import Union, Literal, ContextManager, Callable
from collections import defaultdict
from collections.abc import Iterable
import gzip
from io import BufferedReader

import numpy as np
import nibabel as nib


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
        """Validate that a position array matches expected geometry.

        The function verifies that the input array has:
          1. Shape with 2 dimensions.
          2. Second dimension size of length 3 (representing 3D coordinates).

        Parameters
        ----------
        pos : array_like
            Input array expected to represent N coordinates in 3D space. Must be
            convertible to uint32 dtype, with shape (N, 3).

        Returns
        -------
        numpy.ndarray of dtype uint32
            A copy of the array with the validated shape, cast to uint32 dtype.

        Raises
        ------
        AssertionError
            If the input does not have:
              1. Shape of length 2 (N-dimensional coordinates).
              2. Second dimension size of 3 (3D coordinates).

        Examples
        --------
        >>> positions = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
        >>> val_pos = validate_pos(positions)  # Will run assertions and return uint32 array
        """
        pos = np.array(pos, dtype=np.uint32)
        assert (
            len(pos.shape) == 2
        ), f"pos needs to have len(pos.shape) to be 2, but found to be {len(pos.shape)}"
        assert (
            pos.shape[1] == 3
        ), f"expecting Nx3 array, but found to be Nx{pos.shape[1]}"
        return pos

    def read(self, pos: Union[list[list[int]], np.ndarray]) -> list[bytes]:
        raise NotImplementedError

    def write(self, pos: Union[list[list[int]], np.ndarray], data: list[bytes]):
        """Write data into memory buffer at specified positions.

        Writes a sequence of byte objects (`data`) into the buffer at the
        corresponding voxel positions specified by `pos`. Each position in
        `pos` must correspond exactly to the position of the corresponding data
        in `data`.

        Parameters
        ----------
        pos : sequence of int or tuple of int
            The memory positions where the data will be written.
            Must be iterable and of the same length as `data`.
            Each position can be either a single integer (for 1D) or a tuple
            of integers (for multi-dimensional addressing).
        data : sequence of bytes
            The data to be written into the memory buffer.
            Must be an iterable collecting of bytes objects.

        Raises
        ------
        AssertionError
            If any of the validation checks fail, an assertion error will be raised
            with a descriptive message explaining the failure.
        NotImplementedError
            If the SpatialIndex is not writable
        """
        raise NotImplementedError

    def save(self, affine=np.eye(4)):
        """Save the in memory spatial index data to on disk files.

        Parameters
        ----------
        affine : array_like, shape (4, 4)
            The affine transformation matrix to apply to the voxel data.
            Default: `np.eye(4)` (identity matrix).

        Raises
        ------
        SpatialIndexWriteExc
            If either the voxel file or attribute file already exists,
            or if the total byte size of all attribute data exceeds `uint32`.
        NotImplementedError
            If the SpatialIndex is not writable
        """
        raise NotImplementedError


class WritableSpatialIndex(SpatialIndex):

    writable = True

    _buffer: dict[tuple[int, int, int], list[bytes]] = defaultdict(list)

    def __init__(self, filepath, mode="w"):
        super().__init__(filepath, mode)

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

    def save(self, affine=np.eye(4)):
        voxel_file = str(self.filepath) + self.VOXEL_SUFFIX
        attr_file = str(self.filepath) + self.ATTR_SUFFIX

        if Path(voxel_file).exists():
            raise SpatialIndexWriteExc(
                f"File {voxel_file} already exist. Flushing incomplete files not yet supported"
            )

        if Path(attr_file).exists():
            raise SpatialIndexWriteExc(
                f"File {attr_file} already exist. Flushing incomplete files not yet supported"
            )

        total_bytes = 0
        for arrbytes in self._buffer.values():
            for b in arrbytes:
                total_bytes += len(b)

        if total_bytes > self.UINT32_MAX:
            raise SpatialIndexWriteExc(
                "Spatial index data > uint32max. Currently not supported"
            )

        X, Y, Z = np.array(list(self._buffer.keys())).T

        # shape needs to be 1 + max index
        arr = np.zeros((np.max(X) + 1, np.max(Y) + 1, np.max(Z) + 1), dtype=np.uint64)
        _offset_counter = 0
        _towritearr = []

        for key, value in self._buffer.items():
            _towrite = b"".join(value)

            # as amazing it appears, bitshift has lower priority than addition
            # e.g. python -c "print(1 << 1 + 1)" prints 4, rather than 3
            arr[key] = np.uint64((_offset_counter << 32) + len(_towrite))

            _offset_counter += len(_towrite)
            _towritearr.append(_towrite)

        # needs to explicitly specify uint64 dtype
        img = nib.Nifti1Image(arr, affine=affine, dtype=np.uint64)
        nib.save(img, voxel_file)

        with open(attr_file, "wb") as fp:
            fp.write(b"".join(_towritearr))


class ReadableSpatialIndex(SpatialIndex):

    readable = True

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
