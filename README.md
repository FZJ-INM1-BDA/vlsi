# vlsi

Variable-Length-Spatial-Index (or velvet-silk) is an implementation of variable length datatype for N-dimensional array. It is designed with write-once-read-many paradigm, and thus aims to be fast and memory efficient read operations. 

## Background

see [background.md](./background.md)

## Specification

see [specifications.md](./specifications.md)

## Similar projects

[sparseindex in siibra-python](https://github.com/FZJ-INM1-BDA/siibra-python/blob/66c13cd/siibra/volumes/sparsemap.py#L30) (original inspiration of this project)

[v2 sparsedex in siibra-python](https://github.com/FZJ-INM1-BDA/siibra-python/blob/2997206/siibra/atlases/sparsemap.py) (first implementation of `vlsi`)

[vlen codec in numcodecs](https://github.com/zarr-developers/numcodecs/blob/e0ddee6b6d01bfd35a91085ec0a20dae6bf50e13/numcodecs/vlen.pyx#L48)


## LICENSE

TBD
