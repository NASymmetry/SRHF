"""
Tests for MP2.run_degen_tensor() (srhf/mp2.py), the exploit_degen=True-aware
MP2 correlation energy path.

Level 1 (this file, first section): a pure-math regression test for the
"quadratic reuse" bug found while first designing this feature --
DegenTensor.einsum is designed for LINEAR consumption (a tagged block's
degeneracy factor is charged once, correct if the result is used once, e.g.
added into a Fock matrix). MP2's energy formula is QUADRATIC in the same
integral (IJAB * (2*IJAB - swap(IJAB))): if DegenTensor.einsum already
multiplied the array by `degen` once, squaring it afterward gives `degen**2`,
not the physically correct `degen**1` (Sum_u X_u**2 == degen * X**2 when X_u
is constant across partners, NOT (degen*X)**2). This is why the shipped
MP2.run_degen_tensor() implementation does NOT use DegenTensor/
DegenIntegralFactory at all -- it reuses run_symm()'s combined-tensor formula
unchanged, after tiling so_orbitals.C/eps across each irrep's own degeneracy
count (see run_degen_tensor()'s docstring in srhf/mp2.py for the full
history, including why an earlier per-irrep-pair-block decomposition of this
method was found to be wrong even for water). This Level 1 test is kept as a
regression guard against reintroducing that mechanism into MP2.

Level 2 (this file, second section): end-to-end correctness against Psi4
conventional MP2, covering water (no degenerate irreps -- exploit_degen has
no effect), ammonia/STO-3G (self-paired E irrep), and methane/cc-pVDZ (E, T1,
T2 all populated -- cross-irrep degenerate blocks, the case that exposed the
original per-block design's bug). For each: srhf's correlation energy must
match Psi4, and must agree between exploit_degen=True and exploit_degen=False
on the same molecule/geometry.

Note: importing srhf.mp2 requires srhf/ itself on sys.path, because mp2.py's
own `from bdmats import BDMatrix` is a bare (non srhf.-package-qualified)
import -- see test/smoke_mp2.py's docstring for the full explanation.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "srhf"))

import numpy as np
import psi4
import pytest

from srhf.rhf import SRHF
from srhf.options import Options
from srhf.degen_tensor import AxisMeta, DegenTensor
from srhf.mp2 import MP2


# ---------------------------------------------------------------------------
# Level 1: degen**1 vs degen**2 mechanism regression test (no chemistry)
# ---------------------------------------------------------------------------

def test_degen_tensor_einsum_then_square_gives_wrong_exponent():
    """
    Demonstrates the exact mechanism bug: using DegenTensor.einsum to build
    a fully-contracted MO tensor and then squaring it (as MP2's Coulomb term
    naively would) double-charges the degeneracy factor.
    """
    np.random.seed(0)
    degen = 3
    ERI = np.random.rand(4, 3)  # axes: p (bra, degenerate), r (ket, trivial)
    C_mat = np.random.rand(4, 5)  # mimics occ_C: contracts p away into new label I
    d_vec = np.random.rand(3)     # mimics a ket-side contraction, contracts r away

    # Mimic a Case-B ERI block: bra axis p is the sole pending axis of its
    # coupled pair (matching DegenIntegralFactory's convention), ket trivial.
    tagged = DegenTensor(ERI, (AxisMeta(irrep=0, degen=degen, pending=True),
                                AxisMeta(irrep=1, degen=1, pending=False)))

    # MP2's IAJB pattern: EVERY original axis (p, r) gets contracted away,
    # replaced entirely by new MO labels (I) -- unlike the Fock build, where
    # the bra axis survives to become the Fock matrix's own index. p is the
    # sole pending, contracted-away label here, so DegenTensor.einsum should
    # charge `degen` once: this equals degen * raw -- correct for a SUM
    # Sum_u X_u, which IS what a single (non-squared) use of this needs.
    linear_result = DegenTensor.einsum("pr,pI,r->I", tagged, C_mat, d_vec)
    raw = np.einsum("pr,pI,r->I", ERI, C_mat, d_vec)
    assert np.allclose(linear_result.array, degen * raw)

    # "Quadratic" (mis)use, mirroring MP2's IJAB * (2*IJAB - swap(IJAB)):
    # squaring the auto-weighted linear result gives degen**2 * raw**2.
    buggy_squared = linear_result.array ** 2

    # The physically correct quantity for a sum of squares over `degen`
    # constant-valued partners is Sum_u X_u**2 == degen * X**2 (X_u == raw
    # for every u when the partner-independence property holds -- proven
    # for Case B in the implementation plan; verified against real
    # chemistry in test_mp2_block_correctness.py). NOT (degen*raw)**2.
    correct_sum_of_squares = degen * raw ** 2

    assert not np.allclose(buggy_squared, correct_sum_of_squares), (
        "expected the naive DegenTensor.einsum-then-square approach to "
        "disagree with the correct sum-of-squares -- if this assertion "
        "fails, either degen==1 (test is vacuous) or the two coincidentally "
        "matched; re-check the test setup"
    )
    assert np.allclose(buggy_squared, degen ** 2 * raw ** 2)

    # The correct implementation: use the UNWEIGHTED (.array-only, plain
    # np.einsum, never DegenTensor.einsum) value, square it, THEN apply one
    # factor of degen manually -- matching MP2.run_degen_tensor()'s design
    # and SRHF.degen_rhf_energy's existing "weight applied outside/after
    # the contraction, once" pattern.
    manual_result = degen * raw ** 2
    assert np.allclose(manual_result, correct_sum_of_squares)


# ---------------------------------------------------------------------------
# Level 2: end-to-end correctness vs Psi4 conventional MP2
# ---------------------------------------------------------------------------

CASES = {
    "water": ("cc-pvdz", """
noreorient
0 1
units bohr
O       0.00000000     0.00000000    -0.14322735
H       0.00000000     1.43844196     0.98630424
H       0.00000000    -1.43844196     0.98630424
"""),
    "ammonia": ("sto-3g", """
noreorient
0 1
units bohr
N       0.00000000     0.00000000     0.13125886
H      -0.88122565    -1.52632759    -0.60791885
H      -0.88122565     1.52632759    -0.60791885
H       1.76245129    -0.00000000    -0.60791885
"""),
    "methane": ("cc-pvdz", """
noreorient
0 1
units bohr
C       0.00000000     0.00000000     0.00000000
H       1.18813758    -1.18813758     1.18813758
H      -1.18813758     1.18813758     1.18813758
H       1.18813758     1.18813758    -1.18813758
H      -1.18813758    -1.18813758    -1.18813758
"""),
}


def _run_srhf_mp2_corr(basis, mol_str, exploit_degen):
    opts = Options(subgroup=False, exploit_degen=exploit_degen, guess="sad",
                    scf_max_iter=100, e_convergence=1e-10, d_convergence=1e-10,
                    diis=True, sparse_transform=False, degen_tensor=False, mp2=False)
    job = SRHF(mol_str, basis, opts)
    job.run()
    mp2 = MP2(job.molecule, job.options, job.so_orbitals, job.ERI, job.repacked_bigERI)
    return mp2.run_degen_tensor()


def _psi4_mp2_corr(basis, mol_str):
    psi4.core.clean()
    psi4.core.clean_options()
    psi4.set_options({"basis": basis, "scf_type": "pk", "mp2_type": "conv",
                       "e_convergence": 1e-10, "d_convergence": 1e-10})
    psi4.geometry(mol_str)
    return psi4.energy("mp2") - psi4.energy("scf")


@pytest.mark.parametrize("name", list(CASES.keys()))
def test_run_degen_tensor_matches_psi4(name):
    basis, mol_str = CASES[name]
    ref_corr = _psi4_mp2_corr(basis, mol_str)
    corr_true = _run_srhf_mp2_corr(basis, mol_str, exploit_degen=True)
    corr_false = _run_srhf_mp2_corr(basis, mol_str, exploit_degen=False)

    assert abs(corr_true - ref_corr) < 1e-9
    assert abs(corr_false - ref_corr) < 1e-9
    assert abs(corr_true - corr_false) < 1e-9
