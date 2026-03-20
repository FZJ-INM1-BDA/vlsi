import pytest
import json
from tempfile import TemporaryDirectory
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from typing import Union

import requests

from vlsi import ReadableSpatialIndex, ReadableSparseIndex, V0_ALIAS_SUFFIX
from vlsi.base import V0SpatialIndex

url = "https://data-proxy.ebrains.eu/api/v1/buckets/reference-atlas-data/sparse-indices/colin27-jba30-hg"


@pytest.fixture(scope="session")
def local_si_name():
    name = "foo"

    def download(url: str, dest: Union[str, Path]):
        resp = requests.get(url)
        resp.raise_for_status()
        with open(dest, "wb") as fp:
            fp.write(resp.content)

    with TemporaryDirectory() as _dir:

        fname = Path(_dir) / name

        with ThreadPoolExecutor() as ex:
            list(
                ex.map(
                    download,
                    [
                        url,
                        url + V0SpatialIndex.VOXEL_SUFFIX,
                        url + V0SpatialIndex.ATTR_SUFFIX,
                        url + V0_ALIAS_SUFFIX,
                    ],
                    [
                        fname,
                        fname.with_suffix(V0SpatialIndex.VOXEL_SUFFIX),
                        fname.with_suffix(V0SpatialIndex.ATTR_SUFFIX),
                        fname.with_suffix(V0_ALIAS_SUFFIX),
                    ],
                )
            )

        yield fname


@pytest.fixture(scope="session")
def local_spatial_index(local_si_name):
    yield ReadableSpatialIndex(str(local_si_name))


@pytest.fixture(scope="session")
def local_sparse_index(local_si_name):
    yield ReadableSparseIndex(str(local_si_name))


@pytest.fixture(scope="session")
def remote_spatial_index():

    class RemoteReader:
        def __init__(self, path: str):
            self.marker = 0
            self.session = requests.Session()
            self.url = path

        def read(self, bytes=-1):

            if bytes == -1:
                resp = self.session.get(self.url)
                resp.raise_for_status()
                return resp.content

            assert bytes > 0

            end = bytes + self.marker - 1
            headers = {"Range": f"bytes={self.marker}-{end}"}
            resp = self.session.get(self.url, headers=headers)
            resp.raise_for_status()
            return resp.content

        def seek(self, offset):
            self.marker = offset

    @contextmanager
    def reader_read(url: str):
        yield RemoteReader(url)

    yield ReadableSpatialIndex(url, reader_read=reader_read)


jba30_colin_args = [
    (
        [-56.0, -6.0, 9],
        json.dumps(
            {
                "d35d2d": 0.0003129999968223274,
                "c4d406": 0.3191109895706177,
                "6652b2": 0.28016701340675354,
                "943ce3": 9.999999974752427e-07,
                "23e136": 0.0006760000251233578,
                "749a1e": 0.3054789900779724,
                "ef6127": 0.0942310020327568,
                "1f3c97": 2.099999983329326e-05,
            }
        ).encode("utf-8")
        + b"\n",
    ),
    (
        [8.0, 68.0, 1.0],
        json.dumps(
            {"a72203": 0.3991990089416504, "ab1fe3": 0.08381400257349014}
        ).encode("utf-8")
        + b"\n",
    ),
]


@pytest.mark.parametrize(
    "pos, expected",
    [
        *[
            pytest.param([p], [ex], id=f"single-pos-idx-{idx}")
            for idx, (p, ex) in enumerate(jba30_colin_args)
        ],
        pytest.param(
            [arg[0] for arg in jba30_colin_args],
            [arg[1] for arg in jba30_colin_args],
            id="merged-array",
        ),
    ],
)
def test_local_legacy_spatial_indices(
    pos, expected, local_spatial_index: ReadableSpatialIndex
):
    assert local_spatial_index.read(pos) == expected


jba30_colin_fullanme_args = [
    (
        [-56.0, -6.0, 9],
        {
            "Area OP3 (POperc) - left hemisphere": 0.0003129999968223274,
            "Area OP4 (POperc) - left hemisphere": 0.3191109895706177,
            "Area Op5 (Frontal Operculum) - left hemisphere": 0.28016701340675354,
            "Area Op6 (Frontal Operculum) - left hemisphere": 9.999999974752427e-07,
            "Area TE 1.0 (HESCHL) - left hemisphere": 0.0006760000251233578,
            "Area TE 1.2 (HESCHL) - left hemisphere": 0.3054789900779724,
            "Area TE 2.1 (STG) - left hemisphere": 0.0942310020327568,
            "Area TE 3 (STG) - left hemisphere": 2.099999983329326e-05,
        },
    ),
    (
        [8.0, 68.0, 1.0],
        {
            "Area Fp1 (FPole) - right hemisphere": 0.3991990089416504,
            "Area Fp2 (FPole) - right hemisphere": 0.08381400257349014,
        },
    ),
]


@pytest.mark.parametrize(
    "pos, expected",
    [
        *[
            pytest.param([p], [ex], id=f"single-pos-idx-{idx}")
            for idx, (p, ex) in enumerate(jba30_colin_fullanme_args)
        ],
        pytest.param(
            [arg[0] for arg in jba30_colin_fullanme_args],
            [arg[1] for arg in jba30_colin_fullanme_args],
            id="merged-array",
        ),
    ],
)
def test_local_legacy_spatial_indices_fullname(pos, expected, local_sparse_index):
    assert local_sparse_index.read_and_parse(pos) == expected


@pytest.mark.parametrize(
    "pos, expected",
    [
        *[
            pytest.param([p], [ex], id=f"single-pos-idx-{idx}")
            for idx, (p, ex) in enumerate(jba30_colin_args)
        ],
        pytest.param(
            [arg[0] for arg in jba30_colin_args],
            [arg[1] for arg in jba30_colin_args],
            id="merged-array",
        ),
    ],
)
def test_remote_legacy_spatial_indices(
    pos, expected, remote_spatial_index: ReadableSpatialIndex
):
    assert remote_spatial_index.read(pos) == expected
