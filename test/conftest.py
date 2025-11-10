import pytest
import numpy as np
#from srhf import rhf
from srhf.rhf import SRHF
from srhf.options import Options
import molsym
from molsym.molecule import Molecule
from molsym.salcs.spherical_harmonics import SphericalHarmonics
from molsym.salcs.projection_op import ProjectionOp

@pytest.fixture
def h2_sto3g():
    """Simple H2 molecule with default basis"""
    return Molecule("H 0 0 0; H 0 0 0.74", basis="sto-3g")

@pytest.fixture
def default_opts():
    return Options()

@pytest.fixture
def tol():
    return 1e-8


import pytest
import numpy as np
from srhf.bdmats import BDMatrix

# -------------------------------
# BDMatrix fixtures
# -------------------------------

@pytest.fixture
def bd_eye_2x1():
    """Two-block identity BDMatrix: 2x2 and 1x1."""
    return BDMatrix([np.eye(2), np.eye(1)])

@pytest.fixture
def bd_random_3x2():
    """Two blocks of random square matrices for generic tests."""
    np.random.seed(42)
    return BDMatrix([np.random.rand(3, 3), np.random.rand(2, 2)])

@pytest.fixture
def bd_mixed_shapes():
    """Block with empty + non-empty parts for edge case testing."""
    return BDMatrix([np.array([]), np.eye(2)])

@pytest.fixture
def bd_scalar_blocks():
    """Simple scalar blocks, e.g. [[2]], [[3]]."""
    return BDMatrix([np.array([[2.0]]), np.array([[3.0]])])

@pytest.fixture
def nh3_ccpvdz_data():
    data = np.load("data/nh3_ccpvdz_diis_fixture.npz")
    F_list = data["F"]  # shape (6, 20, 20)
    D_list = data["D"]  # shape (6, 20, 20)
    S = data["S"]
    A = data["A"]

    # Wrap as BDMatrix with a single block for convenience
    F_bd = [BDMatrix([F]) for F in F_list]
    D_bd = [BDMatrix([D]) for D in D_list]
    S_bd = BDMatrix([S])
    A_bd = BDMatrix([A])

    return F_bd, D_bd, S_bd, A_bd

import pytest
import numpy as np
import psi4

# -----------------------------------------------------------------------------
# Global defaults
# -----------------------------------------------------------------------------
@pytest.fixture(scope="session")
def tol():
    """Numerical tolerance for matrix equality checks."""
    return 1e-10


@pytest.fixture(scope="session")
def default_opts():
    """Minimal options namespace for testing."""
    class Opts:
        guess = "core"
        exploit_degen = True
        sad_cycles = 3
        docc = None
    return Opts()

@pytest.fixture(scope="session")
def nh3_ccpvdz_diis_data():
    """Load DIIS-space (SO basis) matrices from saved .npz file."""
    data = np.load("data/nh3_ccpvdz_diis_fixture.npz", allow_pickle=True)
    return data

# -----------------------------------------------------------------------------
# NH3 / cc-pVDZ real dataset
# -----------------------------------------------------------------------------
@pytest.fixture(scope="session")
def nh3_ccpvdz_data():
    """
    Load precomputed NH3/cc-pVDZ Fock, density, overlap, and A matrices,
    plus on-demand AO integrals (S, T, V, I) from Psi4.
    """

    # Build molecule & basis with Psi4
    mol = """
    noreorient
    0 1
    units bohr
    N       0.00000000     0.00000000     0.13125886     
    H      -0.88122565    -1.52632759    -0.60791885     
    H      -0.88122565     1.52632759    -0.60791885     
    H       1.76245129    -0.00000000    -0.60791885     
    """
    basis_input = "cc-pvdz"
    
    options_kwargs = {
            "subgroup" : False,
            "exploit_degen" : True,
            "guess" : "sad",
            "benchmark" : False,
            "intsdpd": False,
            "e_convergence": 1e-10,
            "d_convergence": 1e-10,
            "compare_psi" : False,
            "diis": True,
            "scf_max_iter" : 50,
            "mp2" : False,
            "sparse_transform" : False,
            }
    options_obj = Options(**options_kwargs)
    job = SRHF(mol, basis_input, options_obj)
    molecule = psi4.geometry(mol)
    molecule.update_geometry()
    ndocc = job.process_input() //2
    schema = job.qc()
    qcmol = Molecule.from_schema(schema)
    symtext = molsym.Symtext.from_molecule(qcmol)
    mol = symtext.mol
    molecule.set_geometry(psi4.core.Matrix.from_array(mol.coords))
    job.basis = psi4.core.BasisSet.build(molecule, 'BASIS', basis_input, puream = True)
    ints = psi4.core.MintsHelper(job.basis)

    bset, nbas_vec = job.get_basis()

    coords = SphericalHarmonics(symtext, bset)
    salcs = ProjectionOp(symtext, coords)
    nbfxns = psi4.core.BasisSet.nbf(job.basis)
     
    salcs.sort_to('blocks')
    salcs.salc_sets = []
    fxn_list = []
    for ir, irrep in enumerate(symtext.irreps):
        if len(salcs.salcs_by_irrep[ir]) == 0:
            salcs.salc_sets.append(np.zeros((0, nbfxns)))
        else:
            salcs.salc_sets.append(np.vstack([salcs[i].coeffs for i in salcs.salcs_by_irrep[ir]]))

            #salcs.salc_sets.append(np.row_stack([salcs[i].coeffs for i in salcs.salcs_by_irrep[ir]]))
    print("The schema")     
    print(schema)     
    #basis = psi4.core.BasisSet.build(mol, "BASIS", "cc-pVDZ")
    #mints = psi4.core.MintsHelper(basis)
    #nh3_data = {
    #    "mol": mol,
    #    "basis": basis,
    #    "S": mints.ao_overlap().np,
    #    "T": mints.ao_kinetic().np,
    #    "V": mints.ao_potential().np,
    #    "I": mints.ao_eri().np
    #}
    nh3_data = {
    "symtext" : symtext,
    "salcs" : salcs,
    "ndocc" : ndocc,
    "options" : options_obj,
    "nbfxns" : nbfxns,
    "fxn_list" : fxn_list,
    "basis" : job.basis,
    "molecule" : molecule,
    "basis_input" : basis_input,
    "bset" : bset,  
    }
    return nh3_data

@pytest.fixture(scope="session")
def nh3_ccpvdz_integrals(nh3_ccpvdz_data):
    """Compute AO integrals using the basis from nh3_ccpvdz_data."""
    import psi4

    basis = nh3_ccpvdz_data["basis"]
    mints = psi4.core.MintsHelper(basis)

    ao_ints = {
        "S": mints.ao_overlap().np,
        "T": mints.ao_kinetic().np,
        "V": mints.ao_potential().np,
        "ERI": mints.ao_eri().np,
    }
    return ao_ints

@pytest.fixture(scope="session")
def nh3_ccpvdz_so_integrals(nh3_ccpvdz_data, nh3_ccpvdz_integrals):
    """Transform AO integrals (S, T, V) to the symmetry-adapted (SO) basis."""
    import numpy as np
    

    so_ints = {}
    path = "data/nh3_ccpvdz_so_integrals.npz"
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files} 

