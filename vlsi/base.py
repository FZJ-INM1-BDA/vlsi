from pathlib import Path
from typing import Union, Literal
import numpy as np


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

    def save(self, affine=np.eye(4), shape: tuple[int, int, int] = None):
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
            or if the index is already closed (saved)
        NotImplementedError
            If the SpatialIndex is not writable
        """
        raise NotImplementedError
