import numpy as np
import pytest
from srhf.srhf_helper import SOrbitals
from srhf.bdmats import BDMatrix

def test_sorbitals_init(nh3_ccpvdz_data):
    """Verify that SOrbitals initializes correctly with NH3/cc-pVDZ data."""
    data = nh3_ccpvdz_data

    symtext = data["symtext"]
    salcs = data["salcs"]
    ndocc = data["ndocc"]
    options = data["options"]
    nbfxns = data["nbfxns"]
    fxn_list = data["fxn_list"]
    basis = data["basis"]
    molecule = data["molecule"]
    basis_input = data["basis_input"]
    bset = data["bset"]

    so = SOrbitals(symtext, salcs, ndocc, options, nbfxns, fxn_list,
                   basis, molecule, basis_input, bset)

    # core checks
    assert hasattr(so, "symtext")
    assert hasattr(so, "salcs")
    assert hasattr(so, "basis")
    assert hasattr(so, "ndocc")
    assert so.nbfxns == nbfxns
    assert so.symtext is symtext
    assert isinstance(so.salcs.salc_sets, list)

def test_sad_guess_reproduces_saved(nh3_ccpvdz_data, nh3_ccpvdz_so_integrals):
    """Check that SOrbitals.sad_guess() reproduces the saved NH3 SAD Density Matrix."""
    data = nh3_ccpvdz_data
    saved = np.load("data/nh3_sad_data.npz", allow_pickle=True)
    D_i = BDMatrix(list(saved["D_blocks"]))

    so = SOrbitals(
        data["symtext"], data["salcs"], data["ndocc"], data["options"],
        data["nbfxns"], data["fxn_list"], data["basis"],
        data["molecule"], data["basis_input"], data["bset"]
    )
    so_ints = nh3_ccpvdz_so_integrals
    print(so_ints) 
    S = so_ints["S_blocks"]
    T = so_ints["T_blocks"]
    V = so_ints["V_blocks"]
    #D_new, docc_new = so.sad_guess_v2(S, T, V)
    D_new = so.sad_guess_v2(S, T, V)

    assert np.allclose(D_new.full_mat(), D_i.full_mat(), atol=1e-10)
