from srhf.options import Options
import pytest

def test_default_options():
    opts = Options()
    assert opts.diis is True
    assert opts.second_order is False
    assert opts.scf_max_iter == 50
    assert opts.e_convergence == pytest.approx(1e-7)

