"""
Level 2 (block construction correctness) and Level 3 (Fock contraction
correctness) checks for srhf.degen_tensor.DegenIntegralFactory.

Every comparison here is against a reference built independently of
DegenIntegralFactory/DPD's own machinery: a plain dense, uncompressed
AO->SO ERI transform (mirroring SRHF.aotoso_2, reimplemented locally) sliced
or summed by hand -- not a copy of the code under test.

Two systems are used deliberately:
  - NH3/cc-pVDZ (C3v): has exactly one degenerate irrep (E), so it can only
    ever produce "at most one side degenerate" and "same-irrep-on-both-
    sides" (E x E) blocks -- never a block where bra and ket are two
    DIFFERENT degenerate irreps.
  - CH4/cc-pVDZ (Td): has THREE degenerate irreps (E, T1, T2) simultaneously
    populated, producing genuine cross-irrep degenerate blocks (E x T1,
    E x T2, T1 x T2, ...). This is what exposed a real bug: an earlier
    version of DegenIntegralFactory special-cased only bra_irrep==ket_irrep
    (e.g. T2 x T2), which happened to be correct for every case NH3/STO-3G-
    methane/ammonia could exercise, but was wrong for cross-irrep
    degenerate-both blocks -- the correct condition is "both bra and ket
    are SOME degenerate irrep", matching DPD.lookup_degen()'s braket==3
    condition, not irrep equality. Kept as a permanent regression test.
"""
import numpy as np
import psi4
import pytest
import molsym
from molsym.molecule import Molecule
from molsym.salcs.spherical_harmonics import SphericalHarmonics
from molsym.salcs.projection_op import ProjectionOp

from srhf.rhf import SRHF
from srhf.options import Options
from srhf.srhf_helper import SOrbitals, DPD
from srhf.degen_tensor import DegenTensor, DegenIntegralFactory
from srhf.bdmats import BDMatrix


def _dense_so_eri(ERI_ao, salc_sets):
    """Full/uncompressed AO->SO ERI transform, all partners, all irreps --
    mirrors SRHF.aotoso_2 exactly but reimplemented here independently."""
    parts = [salc.T for salc in salc_sets if salc.shape[0] != 0]
    s = np.concatenate(parts, axis=1)
    t1 = np.einsum("PQRS,Pp->pQRS", ERI_ao, s, optimize="optimal")
    t2 = np.einsum("pQRS,Qq->pqRS", t1, s, optimize="optimal")
    t3 = np.einsum("pqRS,Rr->pqrS", t2, s, optimize="optimal")
    return np.einsum("pqrS,Ss->pqrs", t3, s, optimize="optimal")


def _full_offsets(salc_sets):
    offsets, o = [], 0
    for salc in salc_sets:
        offsets.append(o)
        o += salc.shape[0]
    return offsets


def _build_setup(mol_str, basis_input):
    """Mirrors the setup steps of SRHF.run() up through building
    so_orbitals/ERI, independently of the nh3_ccpvdz_* conftest fixtures so
    this works for arbitrary molecules (e.g. methane/cc-pVDZ)."""
    opts = Options(subgroup=False, exploit_degen=True, guess="core", sparse_transform=False)
    job = SRHF(mol_str, basis_input, opts)
    job.ndocc = job.process_input() // 2
    schema = job.qc()
    qcmol = Molecule.from_schema(schema)
    job.symtext = molsym.Symtext.from_molecule(qcmol)
    mol = job.symtext.mol
    job.molecule.set_geometry(psi4.core.Matrix.from_array(mol.coords))
    job.basis = psi4.core.BasisSet.build(job.molecule, 'BASIS', job.basis_input, puream=True)
    ints = psi4.core.MintsHelper(job.basis)
    bset, nbas_vec = job.get_basis()
    coords = SphericalHarmonics(job.symtext, bset)
    job.salcs = ProjectionOp(job.symtext, coords)
    job.nbfxns = psi4.core.BasisSet.nbf(job.basis)
    job.salcs.sort_to('blocks')
    job.salcs.salc_sets = []
    fxn_list = []
    for ir, irrep in enumerate(job.symtext.irreps):
        if len(job.salcs.salcs_by_irrep[ir]) == 0:
            job.salcs.salc_sets.append(np.zeros((0, job.nbfxns)))
        else:
            job.salcs.salc_sets.append(
                np.row_stack([job.salcs[i].coeffs for i in job.salcs.salcs_by_irrep[ir]])
            )

    so_orbitals = SOrbitals(
        job.symtext, job.salcs, job.ndocc, job.options, job.nbfxns, fxn_list,
        job.basis, job.molecule, job.basis_input, bset,
    )
    ERI_ao = ints.ao_eri().np
    dpd = DPD(job.salcs.salcs_by_irrep, job.symtext, job.salcs, so_orbitals, None, job.options)
    nonzero_blocks = dpd.nonzero_tiles()
    factory = DegenIntegralFactory(job.salcs, job.symtext, so_orbitals, job.options)
    E_dense = _dense_so_eri(ERI_ao, job.salcs.salc_sets)
    return {
        "symtext": job.symtext,
        "salcs": job.salcs,
        "so_orbitals": so_orbitals,
        "ERI_ao": ERI_ao,
        "nonzero_blocks": nonzero_blocks,
        "factory": factory,
        "E_dense": E_dense,
        "full_offsets": _full_offsets(job.salcs.salc_sets),
    }


@pytest.fixture(scope="module")
def nh3_setup(nh3_ccpvdz_data, nh3_ccpvdz_integrals):
    d = nh3_ccpvdz_data
    so_orbitals = SOrbitals(
        d["symtext"], d["salcs"], d["ndocc"], d["options"], d["nbfxns"],
        d["fxn_list"], d["basis"], d["molecule"], d["basis_input"], d["bset"],
    )
    ERI_ao = nh3_ccpvdz_integrals["ERI"]
    dpd = DPD(d["salcs"].salcs_by_irrep, d["symtext"], d["salcs"], so_orbitals, None, d["options"])
    nonzero_blocks = dpd.nonzero_tiles()
    factory = DegenIntegralFactory(d["salcs"], d["symtext"], so_orbitals, d["options"])
    E_dense = _dense_so_eri(ERI_ao, d["salcs"].salc_sets)
    full_offsets = _full_offsets(d["salcs"].salc_sets)
    return {
        "symtext": d["symtext"],
        "salcs": d["salcs"],
        "so_orbitals": so_orbitals,
        "ERI_ao": ERI_ao,
        "nonzero_blocks": nonzero_blocks,
        "factory": factory,
        "E_dense": E_dense,
        "full_offsets": full_offsets,
    }


@pytest.fixture(scope="module")
def methane_ccpvdz_setup():
    mol_str = """
noreorient
0 1
units bohr
C       0.00000000     0.00000000     0.00000000
H       1.18813758    -1.18813758     1.18813758
H      -1.18813758     1.18813758     1.18813758
H       1.18813758     1.18813758    -1.18813758
H      -1.18813758    -1.18813758    -1.18813758
"""
    return _build_setup(mol_str, "cc-pvdz")


# ---------------------------------------------------------------------------
# Shared checks, parametrized over the setup fixtures above
# ---------------------------------------------------------------------------

@pytest.fixture(params=["nh3", "methane_ccpvdz"])
def setup(request, nh3_setup, methane_ccpvdz_setup):
    return {"nh3": nh3_setup, "methane_ccpvdz": methane_ccpvdz_setup}[request.param]


def test_has_a_genuinely_degenerate_irrep(setup):
    # Sanity check that the fixture actually exercises degeneracy at all
    # (unlike water/C2v -- see test/smoke_degen.py's docstring).
    degens = [irrep.d for irrep in setup["symtext"].irreps]
    assert any(d > 1 for d in degens)


def test_nondegenerate_blocks_match_dense_reference_exactly(setup):
    symtext = setup["symtext"]
    checked_any = False
    for block in setup["nonzero_blocks"]:
        degens = [symtext.irreps[ir].d for ir in block]
        if any(d > 1 for d in degens):
            continue
        checked_any = True
        idx = [range(setup["full_offsets"][ir], setup["full_offsets"][ir] + setup["so_orbitals"].irreplength[ir])
               for ir in block]
        ref = setup["E_dense"][np.ix_(*idx)]
        [got] = setup["factory"].degen_ERI_transform(setup["ERI_ao"], [block], swap=False)
        assert np.allclose(got.array, ref, atol=1e-10)
    assert checked_any, "expected at least one fully nondegenerate block"


def test_single_degenerate_side_blocks_match_partner0_slice(setup):
    """bra XOR ket degenerate (not both): compressed block should equal the
    dense reference sliced at partner 0 on every axis -- pure selection,
    no summing."""
    symtext = setup["symtext"]
    so_orbitals = setup["so_orbitals"]
    checked_any = False
    for block in setup["nonzero_blocks"]:
        degens = [symtext.irreps[ir].d for ir in block]
        bra_degen, ket_degen = degens[0], degens[2]
        if not ((bra_degen > 1) ^ (ket_degen > 1)):
            continue
        checked_any = True
        idx = [range(setup["full_offsets"][ir], setup["full_offsets"][ir] + so_orbitals.irreplength[ir])
               for ir in block]
        ref = setup["E_dense"][np.ix_(*idx)]
        [got] = setup["factory"].degen_ERI_transform(setup["ERI_ao"], [block], swap=False)
        assert np.allclose(got.array, ref, atol=1e-10), f"block {block} mismatch"
    assert checked_any, "expected at least one bra-xor-ket-degenerate block"


def test_both_degenerate_blocks_match_summed_reference(setup):
    """Both bra and ket are (possibly different) degenerate irreps:
    compressed block should equal the dense reference with bra fixed at
    partner 0 and the ket SUMMED over all its degenerate partners -- not a
    plain slice, and NOT limited to bra_irrep==ket_irrep."""
    symtext = setup["symtext"]
    so_orbitals = setup["so_orbitals"]
    both_degen = [b for b in setup["nonzero_blocks"]
                  if symtext.irreps[b[0]].d > 1 and symtext.irreps[b[2]].d > 1]
    assert both_degen, "expected at least one both-sides-degenerate block"
    for block in both_degen:
        ir1, ir3 = block[0], block[2]
        ket_degen = symtext.irreps[ir3].d
        il_bra = so_orbitals.irreplength[ir1]
        il_ket = so_orbitals.irreplength[ir3]
        bra_idx = range(setup["full_offsets"][ir1], setup["full_offsets"][ir1] + il_bra)
        ref = None
        for k in range(ket_degen):
            ket_base = setup["full_offsets"][ir3] + k * il_ket
            ket_idx = range(ket_base, ket_base + il_ket)
            piece = setup["E_dense"][np.ix_(bra_idx, bra_idx, ket_idx, ket_idx)]
            ref = piece if ref is None else ref + piece
        [got] = setup["factory"].degen_ERI_transform(setup["ERI_ao"], [block], swap=False)
        assert np.allclose(got.array, ref, atol=1e-10), f"block {block} mismatch"


def test_methane_has_cross_irrep_degenerate_blocks(methane_ccpvdz_setup):
    # Guards the fixture itself: confirms cc-pVDZ methane actually produces
    # a block where bra and ket are two DIFFERENT degenerate irreps (E, T1,
    # T2 are all populated) -- the specific case NH3 structurally cannot
    # produce (only one degenerate irrep, E) and that exposed the bug.
    symtext = methane_ccpvdz_setup["symtext"]
    cross = [b for b in methane_ccpvdz_setup["nonzero_blocks"]
             if b[0] != b[2] and symtext.irreps[b[0]].d > 1 and symtext.irreps[b[2]].d > 1]
    assert cross, "expected at least one cross-irrep degenerate block for methane/cc-pVDZ"


# ---------------------------------------------------------------------------
# Level 3: Fock (J/K) contraction correctness against a brute-force sum over
# ALL raw AO-symmetry partner functions -- independent of DPD, DegenTensor,
# and DegenIntegralFactory's own internal bookkeeping.
# ---------------------------------------------------------------------------

def test_jk_contraction_matches_brute_force_full_partner_sum(setup):
    symtext = setup["symtext"]
    so_orbitals = setup["so_orbitals"]
    E_dense = setup["E_dense"]
    full_offsets = setup["full_offsets"]

    np.random.seed(0)
    D_blocks = []
    for h in range(len(symtext.irreps)):
        il = so_orbitals.irreplength[h]
        if il == 0:
            D_blocks.append(np.array([]))
        else:
            a = np.random.rand(il, il)
            D_blocks.append(a + a.T)
    D = BDMatrix(D_blocks)

    factory = setup["factory"]
    ERI_ao = setup["ERI_ao"]
    nonzero_blocks = setup["nonzero_blocks"]
    repacked = factory.degen_ERI_transform(ERI_ao, nonzero_blocks, swap=False)
    repacked_swapped = factory.degen_ERI_transform(ERI_ao, nonzero_blocks, swap=True)

    checked = 0
    for b, block in enumerate(nonzero_blocks):
        d_sym = block[3]
        bra_ir = block[0]
        il_bra = so_orbitals.irreplength[bra_ir]
        ket_degen = symtext.irreps[d_sym].d
        il_ket = so_orbitals.irreplength[d_sym]

        J = DegenTensor.einsum('pqrs,rs->pq', repacked[b], D.blocks[d_sym])
        K = DegenTensor.einsum('pqrs,rs->pq', repacked_swapped[b], D.blocks[d_sym])

        D_full_ket = np.zeros((il_ket * ket_degen, il_ket * ket_degen))
        for i in range(ket_degen):
            D_full_ket[i*il_ket:(i+1)*il_ket, i*il_ket:(i+1)*il_ket] = D.blocks[d_sym]

        p_idx = range(full_offsets[bra_ir], full_offsets[bra_ir] + il_bra)
        r_idx = range(full_offsets[d_sym], full_offsets[d_sym] + il_ket * ket_degen)

        ERI_slice = E_dense[np.ix_(p_idx, p_idx, r_idx, r_idx)]
        J_brute = np.einsum('pqrs,rs->pq', ERI_slice, D_full_ket)

        E_dense_swapped = np.swapaxes(E_dense, 1, 2)
        ERI_swap_slice = E_dense_swapped[np.ix_(p_idx, p_idx, r_idx, r_idx)]
        K_brute = np.einsum('pqrs,rs->pq', ERI_swap_slice, D_full_ket)

        assert np.allclose(J.array, J_brute, atol=1e-9), f"J mismatch for block {block}"
        assert np.allclose(K.array, K_brute, atol=1e-9), f"K mismatch for block {block}"
        checked += 1

    assert checked == len(nonzero_blocks)
