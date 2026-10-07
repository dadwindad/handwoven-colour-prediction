import numpy as np
import pytest

from weavecolor import data as D
from weavecolor.colorlib import delta_e_2000
from weavecolor.models import PHYSICS, ReflectanceMixModel


@pytest.fixture(scope="module")
def ds():
    return D.load()


@pytest.mark.parametrize("cls", [c for c in PHYSICS if issubclass(c, ReflectanceMixModel)])
def test_mixing_a_yarn_with_itself_returns_the_yarn(ds, cls):
    m = cls()
    extra = np.array(cls.extra0, float)
    R = m._R(ds.warp_lab)
    back = m._lab(m.mix(R, R, 0.3, extra))
    assert delta_e_2000(back, ds.warp_lab).max() < 1e-6


@pytest.mark.parametrize("cls", PHYSICS)
def test_physics_models_fit_and_beat_nothing_crazy(ds, cls):
    idx = np.flatnonzero(~ds.flag)
    m = cls().fit(ds, idx)
    err = delta_e_2000(m.predict(ds, idx), ds.lab[idx]).mean()
    assert 0 < m.w < 1
    assert err < 12  # every law should at least land in the right region of colour space
