# vlsi

Variable-Length-Spatial-Index (or velvet-silk) is an implementation of variable length datatype for 3-dimensional array. It is designed with write-once-read-many paradigm, and is thus optimized for fast and memory efficient read operations. 

## Background

see [background.md](./background.md)

## Specification

see [specifications.md](./specifications.md)

## Usages

A trivial example is included below. For more realistic usages, please refer to e2e tests.

```python
import numpy as np
from vlsi import WritableSpatialIndex, ReadableSpatialIndex, SpatialIndexWriteExc

def main():

    windex = WritableSpatialIndex("testfile")
    windex.write(
        [
            [0, 0, 0],
            [42, 42, 42],
        ],
        [b"origin of life", b"meaning of life"],
    )
    windex.write([[0, 0, 0]], [b"another write"])
    windex.save(np.eye(4), shape=(100, 200, 100))

    try:
        windex.write([[0, 0, 0]], [b"should fail"])
    except SpatialIndexWriteExc:
        ...

    try:
        windex.save(np.eye(4), shape=(100, 200, 100))
    except SpatialIndexWriteExc:
        ...

        
    rindex = ReadableSpatialIndex("testfile")
    assert rindex.read([[0, 0, 0]]) == [b"origin of lifeanother write"]
    assert rindex.read([[42, 42, 42]]) == [b"meaning of life"]
    assert rindex.read([[1, 1, 1]]) == []

if __name__ == "__main__":
    main()

```

## Similar projects

[sparseindex in siibra-python](https://github.com/FZJ-INM1-BDA/siibra-python/blob/66c13cd/siibra/volumes/sparsemap.py#L30) (original inspiration of this project)

[v2 sparsedex in siibra-python](https://github.com/FZJ-INM1-BDA/siibra-python/blob/2997206/siibra/atlases/sparsemap.py) (first implementation of `vlsi`)

[vlen codec in numcodecs](https://github.com/zarr-developers/numcodecs/blob/e0ddee6b6d01bfd35a91085ec0a20dae6bf50e13/numcodecs/vlen.pyx#L48)

[numpy.save](https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html#module-numpy.lib.format)

## LICENSE

apache 2.0
