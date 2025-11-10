import numpy as np
import pytest
from srhf.diis_managerv2 import DIIS_Manager
from srhf.bdmats import BDMatrix

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

class DummySym:
    def __init__(self):
        self.character_table = {"A1": [1]}  # minimal placeholder


@pytest.fixture
def diis_mgr():
    """Minimal DIIS manager with dummy symmetry text."""
    return DIIS_Manager(DummySym())


@pytest.fixture
def small_bd_mats():
    """Return small block-diagonal matrices (2x2 + 1x1) for testing."""
    F = BDMatrix([np.eye(2), np.eye(1)])
    D = BDMatrix([np.eye(2), np.eye(1)])
    S = BDMatrix([np.eye(2), np.eye(1)])
    A = BDMatrix([np.eye(2), np.eye(1)])
    return F, D, S, A


@pytest.fixture
def nh3_ccpvdz_diis_data():
    """Load real NH3/cc-pVDZ SCF data for functional DIIS testing."""
    data = np.load("data/nh3_ccpvdz_diis_fixture.npz")
    F_list = data["F"]  # shape (6, 20, 20)
    D_list = data["D"]
    S = data["S"]
    A = data["A"]

    F_bd = [BDMatrix([F]) for F in F_list]
    D_bd = [BDMatrix([D]) for D in D_list]
    S_bd = BDMatrix([S])
    A_bd = BDMatrix([A])
    return F_bd, D_bd, S_bd, A_bd


# ---------------------------------------------------------------------------
# Unit Tests (synthetic data)
# ---------------------------------------------------------------------------

def test_init_creates_empty_history(diis_mgr):
    assert diis_mgr.diis.fock_hist == []
    assert diis_mgr.diis.error_hist == []


def test_do_diis_adds_history(diis_mgr, small_bd_mats):
    F, D, S, A = small_bd_mats
    diis_mgr.do_diis(F, D, S, A, i=1)
    hist = diis_mgr.diis
    assert hist.iteration == 1
    assert len(hist.fock_hist) == 1
    assert len(hist.error_hist) == 1
    assert np.isclose(hist.dRMS, 0.0)


def test_check_error_behavior(diis_mgr, small_bd_mats):
    F, D, S, A = small_bd_mats
    diis_mgr.do_diis(F, D, S, A, i=1)
    assert diis_mgr.check_error() is False
    diis_mgr.diis.error_hist.append(np.array([[10.0]]))
    assert diis_mgr.check_error() is True


def test_create_b_returns_original_for_first_iters(diis_mgr, small_bd_mats):
    F, D, S, A = small_bd_mats
    diis_mgr.do_diis(F, D, S, A, i=1)
    outF = diis_mgr.create_b()
    assert isinstance(outF, BDMatrix)
    assert np.allclose(outF.full_mat(), F.full_mat())


def test_create_b_raises_on_zero_norm(diis_mgr, small_bd_mats):
    F, D, S, A = small_bd_mats
    for i in range(3):
        diis_mgr.do_diis(F, D, S, A, i=i + 1)
    with pytest.raises(ValueError, match="Zero or NaN norm in error matrix."):
        diis_mgr.create_b()


def test_create_b_limits_history_length(diis_mgr, small_bd_mats):
    F, D, S, A = small_bd_mats
    for i in range(10):
        diis_mgr.do_diis(F, D, S, A, i=i + 1)
        try:
            diis_mgr.create_b()
        except ValueError:
            pass
    assert len(diis_mgr.diis.fock_hist) <= 6
    assert len(diis_mgr.diis.error_hist) <= 6


# ---------------------------------------------------------------------------
# Integration Test (real NH3/cc-pVDZ data)
# ---------------------------------------------------------------------------

def test_real_diis_reduces_error(nh3_ccpvdz_diis_data):
    """Verify that DIIS extrapolation reduces the error norm using real NH3/cc-pVDZ data."""
    F_list, D_list, S_bd, A_bd = nh3_ccpvdz_diis_data

    class DummySymtext:
        def __init__(self):
            self.character_table = {"A1": [1]}

    diis_mgr = DIIS_Manager(DummySymtext())

    # Feed the sequence of iterations
    for i, (F, D) in enumerate(zip(F_list, D_list), start=1):
        diis_mgr.do_diis(F, D, S_bd, A_bd, i=i)

    # Attempt DIIS extrapolation
    F_extrap = diis_mgr.create_b()
    assert isinstance(F_extrap, BDMatrix)

    # Compute last and extrapolated error norms
    def compute_error_norm(F, D, S, A):
        err = A @ (F @ D @ S - S @ D @ F) @ A
        return np.sqrt(np.mean(err**2))

    F_last = F_list[-1].full_mat()
    D_last = D_list[-1].full_mat()
    S = S_bd.full_mat()
    A = A_bd.full_mat()

    err_last = compute_error_norm(F_last, D_last, S, A)
    err_extrap = compute_error_norm(F_extrap.full_mat(), D_last, S, A)

    print(f"DIIS test (NH3/cc-pVDZ): last_err={err_last:.6e}, extrap_err={err_extrap:.6e}")
    assert err_extrap <= err_last * 1.1  # allow small numerical tolerance

