# Background

This project was originally implemented in [siibra-python](https://github.com/FZJ-INM1-BDA/siibra-python).

## Initial motivation

Each of the many statistical maps catalogued by siibra often contains hundreds of individual NifTI images. This pose a challenge to probabilistic assignments, where the operation would need to either

1/ read and cache all NifTI images into memory

2/ iterate over all NifTI images and only read them as needed with no caching

The former boasts significant improvement to speed to subsequent assignments, at the cost of significant memory usage[^memoryprofile]. Despite this, the performance of both approaches suffer from a cold start, as hundreds of NiFTi files need to be opened and potentially decompressed. (This performance penalty punishes the latter more severely, as each subsequent probabilistic assignment would incur performance cost over and over again.)

## First implementation in siibra-python v1

The first implementation of spatial index (hereafter referred to as "old spatial index") contains several inefficiencies:

- spatial index encodes line number 

The old spatial index, each of the voxel encodes the line number in the corresponding assignment table. As a result, in order to decode an entry in the spatial dictionary, the "assignment table" file needs to be read from the beginning of the file.

- assignment table contains full region name

The old spatial index contains region names in the assignment table. This bloats the size of the assignment table. To alleviate the bloat, the assignment table is gzipped.

- the assignment table is stored in memory

The combination of both of the above means that the assignment table is often downloaded (or read) and decompressed in memory. 

- lack of version info

This makes upgrading/backwards compatibility difficult.

## Comparison to siibra-python v1

### Advantages

- memory efficiency

    The only files that need to be stored in memory are `{filename}.spatialindex.nii.gz`. See [^vlsimemory] and [^prevmemory] for a memory usage comparison.

- better cold start performance

    seek-read can be done without reading the entire file. If `{filename}.spatialindex.attr.bin` is available remotely, `vlsi` allows the probs value to be retrieved without downloading the entire file.

### Disadvantages

- easily invalidated

    As `vlsi` uses offset and byte ranges as pointers, new writes to the index will likely invalidate the entire index, and new index must be created.

- mild performance penalty

    As `{filename}.spatialindex.attr.bin` will be on filesystem rather than in memory, repeated assignment can incur performance penalty. An escape hatch may be, allow the file content to be cached in memory, if a flag is set. 

- `{filename}.spatialindex.attr.bin` cannot be compressed (per file basis)
    
    As `vlsi` uses byte range, the file itself cannot be compressed. 


## References

[^memoryprofile]: conservative estimate of memory usage of storing Julich Brain 3.0.3 high granularity statistical map (175 regions per hemisphere) in ICBM 152 nonlinear asymmetric space.

    ```
    350 (number of niftis) * 193 * 229 * 193 (NifTI shape) * 4 (float32 datatype)

    = 11942029400 bytes = 11389 mb
    ```

[^vlsimemory]: conservative estimate of memory usage of `vlsi` of a map in ICBM 152 nonlinear asymetric space

    ```
    193 * 229 * 193 (NifTI shape) * 8 (uint64 datatype)

    = 68240168 bytes = 65 mb
    ```
    
[^prevmemory]: conservative estimate of memory usage of old spatial index of a map in ICBM 152 nonlinear asymetric space

    ```
    193 * 229 * 193 (NifTI shape) * 4 (uint32 datatype)

    = 34120084 bytes = 32 mb

    ~1000 mb for `{filename}.sparseindex.probs.txt`

    = ~1030 mb
    ```
