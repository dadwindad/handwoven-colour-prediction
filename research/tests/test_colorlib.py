import json
from pathlib import Path

import numpy as np

from weavecolor.colorlib import delta_e_2000, lab_to_xyz, xyz_to_lab
from weavecolor import data as D

TESTDATA = json.loads((Path(__file__).parent / "sharma2005_ciede2000_testdata.json").read_text())


def test_ciede2000_matches_sharma_2005():
    lab1 = np.array([r["lab1"] for r in TESTDATA])
    lab2 = np.array([r["lab2"] for r in TESTDATA])
    expected = np.array([r["de00"] for r in TESTDATA])
    assert len(TESTDATA) == 34
    np.testing.assert_allclose(delta_e_2000(lab1, lab2), expected, atol=1e-4)
    # CIEDE2000 is symmetric for these pairs
    np.testing.assert_allclose(delta_e_2000(lab2, lab1), expected, atol=1e-4)


def test_lab_xyz_round_trip():
    lab = np.array([[88.52, -2.92, 11.78], [18.93, -0.67, -0.76], [5.0, 3.0, -4.0]])
    for white in ("D65_10", "D65_2"):
        np.testing.assert_allclose(xyz_to_lab(lab_to_xyz(lab, white), white), lab, atol=1e-9)


def test_dataset_is_complete():
    ds = D.load()
    assert len(ds.codes) == 25
    assert len(ds.lab) == 625
    assert len({(w, f) for w, f in zip(ds.warp, ds.weft)}) == 625
    assert int(ds.flag.sum()) == 29  # 28 asymmetric-pair rows + fabric 38+38
    assert ds.yarn_flag["38"]
