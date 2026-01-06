# Specification

`vlsi` contains three files, two of which are needed to decode the sparse index.

## 0/ Metadata file `{filename}`

utf-8 encoeded text file. *Must* start with `SPATIALINDEX-UTF8-V1`. Not needed to decode sparse index.

## 2/ Spatial dictionary file `{filename}.spatialindex.voxel.nii.gz`

uint64 NiFTi-1 Image file. Each voxel encodes two 32 bit values, offset and bytes.

2.1/ Higher 32 bit (i.e. `voxel_value >> 32`) encodes offset. 

2.2/ Lower 32 bit (i.e. `voxel_value & 0xffffffff`) encodes bytes.

2.3/ For every voxel, offset (2.1) + bytes (2.2) *must* be less or equal to the file size of attr file (3.)

## 3/ Data file `{filename}.spatialindex.attr.bin`

3.1/ For all voxel in (2.), byte range from offset (2.1) to 
offset + bytes (2.2) (i.e. `seek(offset); read(bytes)`) represent must be retrievable.

> n.b. The interpretation of the bytes returned is deliberately not defined. Client using `vlsi` is responsible for decoding the returned bytes.


<!-- ## Usage

Below demonstrates a step-by-step walkthrough on `vlsi`'s read and write implmentations. They have already been implemented in python, but is nevertheless useful for:

1/ translating it into other languages

2/ code auditing

### reading voxel index

To access the probability assignment at `[x, y, z]` voxel position

0/ read `{filename}.sparseindex.alias.json`, parse as JSON object.

1/ read the `uint64` value at the voxel position `[x, y, z]` from `{filename}.sparseindex.voxel.nii.gz`

2/ decode the `offset` by right shift 32 bits; `bytes` by using the bit mask `0xffffffff`

3/ read the attr file `{filename}.spatialindex.attr.bin`, seek `offset` and read `bytes`.  -->


<!-- ## Examples

### Writing

The below example writes Julich Brain 2.9 statistical map to base filename `icbm152_julich2_9` in directory `mesi`.

```python
import siibra
from tqdm import tqdm
from siibra.atlases.sparsemap import SparseIndex

mp = siibra.get_map("2.9", "icbm 152", "statistical")
spi = SparseIndex("icbm152_julich2_9", mode="w")


progress = tqdm(total=len(mp.regions), leave=True)
for regionname in mp.regions:
    volumes = mp.find_volumes(regionname)
    assert len(volumes) == 1
    volume = volumes[0]
    spi.add_img(volume.fetch(), regionname)
    progress.update(1)
progress.close()
spi.save()
```

### Reading

The below example reads the MESI saved above.

```python

import siibra
import numpy as np
from siibra.attributes.locations import Point
from siibra.atlases.sparsemap import SparseIndex


spi = siibra.atlases.sparsemap.SparseIndex("icbm152_julich2_9", mode="r")

pt_phys = [-4.077, -79.717, 11.356]
space = siibra.get_space("icbm 152")
pt = Point(coordinate=pt_phys, space_id=space.ID)
affine = np.linalg.inv(spi.affine)
pt_voxel = pt.transform(affine)
voxelcoord = np.array(pt_voxel.coordinate).astype("int")

val = spi.read([voxelcoord])
print(val) # prints [{'Area hOc2 (V2, 18) - left hemisphere': 0.33959856629371643, 'Area hOc1 (V1, 17, CalcS) - left hemisphere': 0.6118946075439453}]

```

If the spatial index is available over HTTP:

```python
import siibra

remote_spi = siibra.atlases.sparsemap.SparseIndex("https://data-proxy.ebrains.eu/api/v1/buckets/test-sept-22/icbm152_julich2_9.mesi", mode="r")

print(remote_spi.read([voxelcoord])) # prints [{'Area hOc2 (V2, 18) - left hemisphere': 0.33959856629371643, 'Area hOc1 (V1, 17, CalcS) - left hemisphere': 0.6118946075439453}]
``` -->



## Potential future developments

- Binary Spec

Binary probability file/metadata file to improve performance and further reduce memory usage.

## References


