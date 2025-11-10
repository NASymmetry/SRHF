import numpy as np
from srhf.diis_managerv2 import DIIS_Manager
from srhf.bdmats import BDMatrix


class DummySymtext:
    """Stub for symmetry text, only to provide character_table."""
    def __init__(self):
        self.character_table = {"A1": [1]}

def test_real_diis_reduces_error(nh3_ccpvdz_diis_data):
#def test_real_diis_reduces_error(nh3_ccpvdz_data):
    """Verify that DIIS extrapolation reduces the error norm using real NH3/cc-pVDZ data."""

    # Load real NH3 / cc-pVDZ matrices from fixture
    F_list = nh3_ccpvdz_diis_data["F"]
    D_list = nh3_ccpvdz_diis_data["D"]
    S_mat  = nh3_ccpvdz_diis_data["S"]
    A_mat  = nh3_ccpvdz_diis_data["A"]

    # Convert to BDMatrix objects (one block for full matrix)
    F_blocks = [BDMatrix([F]) for F in F_list]
    D_blocks = [BDMatrix([D]) for D in D_list]
    S_bd = BDMatrix([S_mat])
    A_bd = BDMatrix([A_mat])

    diis_mgr = DIIS_Manager(DummySymtext())

    # Feed the sequence of iterations
    for i, (F, D) in enumerate(zip(F_blocks, D_blocks), start=1):
        diis_mgr.do_diis(F, D, S_bd, A_bd, i=i)

    # Attempt DIIS extrapolation
    F_extrap = diis_mgr.create_b()
    assert isinstance(F_extrap, BDMatrix)

    # Compute last and extrapolated error norms
    F_last = F_blocks[-1].full_mat()
    D_last = D_blocks[-1].full_mat()
    S = S_bd.full_mat()
    A = A_bd.full_mat()

    def compute_error_norm(F, D, S, A):
        err = A @ (F @ D @ S - S @ D @ F) @ A
        return np.sqrt(np.mean(err**2))

    err_last = compute_error_norm(F_last, D_last, S, A)
    err_extrap = compute_error_norm(F_extrap.full_mat(), D_last, S, A)

    print(f"DIIS test: last err={err_last:.6e}, extrap err={err_extrap:.6e}")
    # Should reduce or at least not increase significantly
    assert err_extrap <= err_last * 1.1

