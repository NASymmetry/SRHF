"""
Unit tests for SO_RHF._build_soscf_hessian() (srhf/sorhf.py), the dense,
cross-irrep-coupled orbital-rotation Hessian that replaced the original
per-irrep-block-diagonal BDMatrix construction. See soscf_newton_step's
docstring in sorhf.py for the full design rationale: the original Hessian
structurally excluded coupling between same-irrep occupied-virtual rotation
pairs belonging to DIFFERENT irreps (h != h'), which is generically nonzero
by the standard two-electron selection rule (h⊗h always contains the
totally symmetric irrep, for ANY irrep h -- true even for abelian point
groups with no degenerate irreps at all). The old Hessian still converged
to the correct energy but needed far more Newton iterations for real
point-group symmetry than for C1.

Uses exploit_degen=False throughout -- deliberately keeps this test's scope
to the cross-irrep coupling fix only, not the separately-deferred degen_h
chain-rule question noted in soscf_newton_step's docstring.

Parametrized over water/STO-3G (C2v, no degenerate irreps -- isolates
"multiple irreps" from "degenerate irreps" as separate variables) and
methane/cc-pVDZ (Td, E+T1+T2 all populated). Methane/cc-pVDZ specifically
exercises an edge case STO-3G bases can't: an irrep can be populated with
real AO functions while having ZERO occupied orbitals (cc-pVDZ methane's E
and T1 irreps have virtuals only). That shape -- occ_C.blocks[h] of shape
(irreplength[h], 0) -- broke an earlier version of _build_soscf_hessian
(block_diag's occ_C_combined filter used the block's own .size, which is 0
for a (n,0)-shaped block even though the irrep has n real AOs), caught only
by running the full smoke_so_rhf.py end-to-end -- included here too so a
fast, pytest-collected run catches it immediately during development
instead of needing a full Psi4-comparison SCF run.

Fast/deterministic enough to be pytest-collected, unlike test/smoke_so_rhf.py
(which does full Psi4-comparison SCF runs and iteration-count assertions).
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "srhf"))

import numpy as np
import psi4
import pytest
from scipy.linalg import expm, block_diag

from srhf.sorhf import SO_RHF
from srhf.options import Options
from srhf.bdmats import BDMatrix
from srhf.degen_tensor import DegenIntegralFactory

WATER = """
noreorient
0 1
units bohr
O 0.000000000000 0.000000000000 -0.143225816552
H 0.000000000000 1.638036840407 1.136548822547
H 0.000000000000 -1.638036840407 1.136548822547
"""

METHANE_CCPVDZ = """
noreorient
0 1
units bohr
C       0.00000000     0.00000000     0.00000000
H       1.18813758    -1.18813758     1.18813758
H      -1.18813758     1.18813758     1.18813758
H       1.18813758     1.18813758    -1.18813758
H      -1.18813758    -1.18813758    -1.18813758
"""

CASES = {
    "water-sto3g": (WATER, "sto-3g"),
    "methane-ccpvdz": (METHANE_CCPVDZ, "cc-pvdz"),
}


@pytest.fixture(scope="module", params=list(CASES), ids=list(CASES))
def converged_job(request):
    mol_str, basis = CASES[request.param]
    opts = Options(subgroup=False, exploit_degen=False, guess="sad",
                    scf_max_iter=50, e_convergence=1e-11, d_convergence=1e-11,
                    diis=True, second_order=True)
    job = SO_RHF(mol_str, basis, opts)
    job.run()
    return job


def _hessian_pieces(job):
    so = job.so_orbitals
    occ_C = job.C.slicev2([":", ":ndocc_ir"], so.Orbs)
    moF = job.F.einsum('ui,vj,uv', job.C, job.C, job.F)
    return job._build_soscf_hessian(job.bigERI, occ_C, job.C, moF, so)


def _dense_energy(C_dense, occ_idx, H_dense, I_dense):
    """RHF electronic energy (nuclear repulsion omitted -- a constant,
    irrelevant to any derivative) for a given, possibly non-self-consistent,
    dense C, in the same 'one representative partner per irrep, combined'
    basis Biajb_dense/MO_dense already use.

    Matches this codebase's own D/E convention exactly (confirmed against
    SO_RHF.build_D/degen_rhf_energy): D = C_occ @ C_occ.T with NO factor of
    2 for double occupancy, F = H + 2J - K, E = sum(D*(H+F)) with NO 0.5 --
    NOT the textbook E = 0.5*sum(D*(H+F)) with a doubled D. Verified this
    reproduces job.wfn_energy - job.enuc exactly at the converged C, and
    that the finite-difference gradient vanishes there as expected."""
    C_occ = C_dense[:, occ_idx]
    D = C_occ @ C_occ.T
    J = np.einsum('pqrs,rs->pq', I_dense, D, optimize='optimal')
    K = np.einsum('prqs,rs->pq', I_dense, D, optimize='optimal')
    F = H_dense + 2 * J - K
    return np.sum(D * (H_dense + F))


def test_soscf_hessian_cross_irrep_matches_finite_difference(converged_job):
    """Gold-standard check: perturb two DIFFERENT-irrep same-irrep rotation
    pairs simultaneously and confirm the resulting energy curvature matches
    Biajb_dense's off-diagonal entry -- the only check that proves the
    cross-irrep coupling terms are quantitatively correct, independent of
    whether the outer SCF loop converges."""
    job = converged_job
    so = job.so_orbitals
    Biajb_dense, active_by_irrep, gn_flat, occ_num, comb_occ, comb_virt, MO_dense, I_dense = _hessian_pieces(job)
    assert Biajb_dense.size > 0

    pair_irrep = np.concatenate([np.full(n, h) for h, n in enumerate(active_by_irrep)])
    p = 0
    q = next(k for k in range(len(pair_irrep)) if pair_irrep[k] != pair_irrep[p])

    C_combined = block_diag(*[b for b in job.C.blocks if b.size])
    H_dense = block_diag(*[b for b in so.H.blocks if b.size])
    occ_idx = np.concatenate([
        np.arange(BDMatrix.irrep_offsets(so.irreplength)[h],
                   BDMatrix.irrep_offsets(so.irreplength)[h] + orb.ndocc_ir)
        for h, orb in enumerate(so.Orbs) if orb.ndocc_ir
    ])

    N = C_combined.shape[0]

    def kappa(pair_idx):
        k = np.zeros((N, N))
        i, a = comb_occ[pair_idx], comb_virt[pair_idx]
        k[i, a] = 1.0
        k[a, i] = -1.0
        return k

    kp, kq = kappa(p), kappa(q)

    def energy(tp, tq):
        C_rot = C_combined @ expm(tp * kp + tq * kq)
        return _dense_energy(C_rot, occ_idx, H_dense, I_dense)

    h = 2e-3
    fd = (energy(h, h) - energy(h, -h) - energy(-h, h) + energy(-h, -h)) / (4 * h * h)

    assert np.isclose(fd, Biajb_dense[p, q], atol=5e-4, rtol=1e-3), (
        f"finite-difference cross-irrep Hessian entry {fd} does not match "
        f"Biajb_dense[{p},{q}]={Biajb_dense[p, q]}"
    )


def test_soscf_hessian_diagonal_blocks_match_degen_integral_factory(converged_job):
    """Regression: I_dense's per-irrep-diagonal (same-irrep-on-all-4-axes)
    blocks must equal DegenIntegralFactory._transform(ERI_ao)'s
    corresponding blocks to machine precision -- an independently-derived
    'one representative partner per irrep, fully dense' ERI, built directly
    from the raw AO integrals rather than by slicing the already-SO-
    transformed bigERI. Confirms the two are provably equivalent (slicing
    commutes with a linear transform's output columns), not just
    coincidentally close."""
    job = converged_job
    so = job.so_orbitals
    _, _, _, _, _, _, _, I_dense = _hessian_pieces(job)

    factory = DegenIntegralFactory(so.salcs, so.symtext, so, job.options)
    ERI_ao = psi4.core.MintsHelper(so.basis).ao_eri().np
    I_oracle = factory._transform(ERI_ao)

    offsets = BDMatrix.irrep_offsets(so.irreplength)
    for h, il in enumerate(so.irreplength):
        if il == 0:
            continue
        o = offsets[h]
        block = I_dense[o:o + il, o:o + il, o:o + il, o:o + il]
        oracle_block = I_oracle[o:o + il, o:o + il, o:o + il, o:o + il]
        assert np.allclose(block, oracle_block, atol=1e-10), f"irrep {h} block mismatch"


def test_soscf_hessian_engages_cross_irrep_coupling(converged_job):
    """'Did it actually engage' check: off-diagonal (different-irrep pair)
    Hessian entries must NOT be uniformly near-zero -- guards against an
    index-bookkeeping bug that silently reproduces the old block-diagonal-
    by-irrep behavior while looking done."""
    job = converged_job
    Biajb_dense, active_by_irrep, *_ = _hessian_pieces(job)

    pair_irrep = np.concatenate([np.full(n, h) for h, n in enumerate(active_by_irrep)])
    cross_mask = pair_irrep[:, None] != pair_irrep[None, :]
    assert cross_mask.any(), "test molecule doesn't have 2+ active irreps -- can't test cross-irrep coupling"
    assert np.max(np.abs(Biajb_dense[cross_mask])) > 1e-6


# ---------------------------------------------------------------------------
# Cross-partner coupling (exploit_degen=True, the degen_h gap fix)
# ---------------------------------------------------------------------------
#
# The tests above deliberately use exploit_degen=False throughout, isolating
# the cross-irrep coupling fix from the separate cross-PARTNER coupling fix
# tested here. exploit_degen=True's rotation parameter for a degenerate
# irrep represents a rotation applied IDENTICALLY to all degen_h partners at
# once (that's what "exploiting degeneracy" means) -- _build_soscf_hessian
# tiles every irrep degen_h times and pools the resulting Hessian/gradient
# back down to one entry per representative pair (see its docstring in
# sorhf.py for the full derivation, including why a naive "multiply by
# degen_h" fix would have been quantitatively wrong).
#
# AMMONIA_STO3G (C3v, degen=2 E irrep) and METHANE_STO3G (Td, degen=3 T2
# irrep -- the only irrep populated, a clean single-degenerate-irrep stress
# test for d_h=3) are used here rather than the CASES molecules above,
# since water has no degenerate irreps at all and methane/cc-pVDZ's
# cross-irrep-AND-cross-partner combination (while a great end-to-end smoke
# test, see test/smoke_so_rhf.py) is harder to isolate cleanly at the
# single-entry unit-test level.

AMMONIA_STO3G = """
noreorient
0 1
units bohr
N       0.00000000     0.00000000     0.13125886
H      -0.88122565    -1.52632759    -0.60791885
H      -0.88122565     1.52632759    -0.60791885
H       1.76245129    -0.00000000    -0.60791885
"""

METHANE_STO3G = """
noreorient
0 1
units bohr
C       0.00000000     0.00000000     0.00000000
H       1.18813758    -1.18813758     1.18813758
H      -1.18813758     1.18813758     1.18813758
H       1.18813758     1.18813758    -1.18813758
H      -1.18813758    -1.18813758    -1.18813758
"""

DEGEN_CASES = {
    "ammonia-sto3g-E-degen2": (AMMONIA_STO3G, "sto-3g", 2),  # (mol_str, basis, irrep index of the degenerate irrep)
    "methane-sto3g-T2-degen3": (METHANE_STO3G, "sto-3g", 4),
}


@pytest.fixture(scope="module", params=list(DEGEN_CASES), ids=list(DEGEN_CASES))
def converged_degen_job(request):
    mol_str, basis, h_degen = DEGEN_CASES[request.param]
    opts = Options(subgroup=False, exploit_degen=True, guess="sad",
                    scf_max_iter=50, e_convergence=1e-11, d_convergence=1e-11,
                    diis=True, second_order=True)
    job = SO_RHF(mol_str, basis, opts)
    job.run()
    return job, h_degen


def _tiled_reference(job):
    """Build the fully-tiled (all degen_h partners of every irrep explicit)
    combined system directly from the CONVERGED, COMPRESSED job -- i.e. by
    tiling the representative partner's own C/H/ERI data, exactly mirroring
    _build_soscf_hessian's own internal tiling (and MP2.run_degen_tensor()'s
    recipe) -- rather than independently re-diagonalizing an uncompressed
    system. That distinction matters: an independently-diagonalized
    reference has an arbitrary relative eigenvector phase between partners
    (confirmed to produce a confusing, inconsistent-looking result during
    this fix's own derivation), whereas tiling the single diagonalized
    representative partner has no such ambiguity -- there is only ever one
    set of eigenvectors to begin with."""
    so = job.so_orbitals
    irreplength = so.irreplength
    full_sizes = [salc.shape[0] for salc in job.salcs.salc_sets]
    full_offsets = BDMatrix.irrep_offsets(full_sizes)
    populated = [h for h in range(len(irreplength)) if irreplength[h] > 0]

    idx_tiled, C_blocks, H_blocks, occ_idx, tile_start = [], [], [], [], {}
    o, oo = 0, 0
    for h in populated:
        degen_h = so.symtext.irreps[h].d
        il_h = irreplength[h]
        ndocc_h = so.Orbs[h].ndocc_ir
        tile_start[h] = o
        for mu in range(degen_h):
            idx_tiled.append(np.arange(full_offsets[h] + mu * il_h, full_offsets[h] + (mu + 1) * il_h))
            C_blocks.append(job.C.blocks[h])
            H_blocks.append(so.H.blocks[h])
            occ_idx += list(range(oo, oo + ndocc_h))
            o += il_h
            oo += il_h

    idx_tiled = np.concatenate(idx_tiled)
    I_tiled = job.bigERI[np.ix_(idx_tiled, idx_tiled, idx_tiled, idx_tiled)]
    C_tiled = block_diag(*C_blocks)
    H_tiled = block_diag(*H_blocks)
    occ_idx = np.array(occ_idx)
    return C_tiled, H_tiled, I_tiled, occ_idx, tile_start


def test_soscf_hessian_cross_partner_matches_finite_difference(converged_degen_job):
    """Gold-standard check for the degen_h fix: perturb representative
    partner 0's and partner 1's tiled copies of the SAME representative
    (i,a) pair simultaneously, and confirm the resulting energy curvature
    matches the new (post cross-partner fix) Biajb_dense[p,p] diagonal
    entry -- proves the pooled cross-partner term is quantitatively
    correct, the same standard this fix's own numerical derivation used."""
    job, h_degen = converged_degen_job
    so = job.so_orbitals
    Biajb_dense, active_by_irrep, gn_flat, *_ = _hessian_pieces(job)
    assert Biajb_dense.size > 0

    pair_irrep = np.concatenate([np.full(n, h) for h, n in enumerate(active_by_irrep)])
    p = np.where(pair_irrep == h_degen)[0][0]

    C_tiled, H_tiled, I_tiled, occ_idx, tile_start = _tiled_reference(job)
    N = C_tiled.shape[0]

    il_h = so.irreplength[h_degen]
    ndocc_h = so.Orbs[h_degen].ndocc_ir
    degen_h = so.symtext.irreps[h_degen].d

    def kappa(i, a):
        k = np.zeros((N, N))
        k[i, a] = 1.0
        k[a, i] = -1.0
        return k

    # Shared rotation: identical kappa applied to ALL degen_h partners at
    # once, not just the first two -- methane's T2 (degen=3) specifically
    # exercises this (a first version of this test only summed 2 partners
    # unconditionally and silently passed for degen=2 while being wrong for
    # degen=3, since it was missing the third partner's contribution).
    kappa_shared = sum(
        kappa(tile_start[h_degen] + mu * il_h, tile_start[h_degen] + mu * il_h + ndocc_h)
        for mu in range(degen_h)
    )

    def energy(C_dense):
        return _dense_energy(C_dense, occ_idx, H_tiled, I_tiled)

    h = 2e-3
    E0 = energy(C_tiled)
    Hpp_fd = (energy(C_tiled @ expm(h * kappa_shared)) - 2 * E0 + energy(C_tiled @ expm(-h * kappa_shared))) / h**2

    assert np.isclose(Hpp_fd, Biajb_dense[p, p], atol=5e-3, rtol=2e-3), (
        f"finite-difference cross-partner Hessian diagonal {Hpp_fd} does not "
        f"match Biajb_dense[{p},{p}]={Biajb_dense[p, p]}"
    )


def test_soscf_hessian_engages_cross_partner_coupling(converged_degen_job):
    """'Did it actually engage' check: with cross-partner coupling
    included, Biajb_dense's diagonal entry for a degenerate irrep's pair
    must differ meaningfully from what _build_soscf_hessian would compute
    treating that irrep as if it were NOT degenerate (i.e. from the same
    converged state, but with exploit_degen temporarily forced off) --
    guards against the tiling/pooling silently reducing to a no-op."""
    job, h_degen = converged_degen_job
    so = job.so_orbitals
    occ_C = job.C.slicev2([":", ":ndocc_ir"], so.Orbs)
    moF = job.F.einsum('ui,vj,uv', job.C, job.C, job.F)

    Biajb_dense, active_by_irrep, *_ = job._build_soscf_hessian(job.bigERI, occ_C, job.C, moF, so)
    pair_irrep = np.concatenate([np.full(n, h) for h, n in enumerate(active_by_irrep)])
    p = np.where(pair_irrep == h_degen)[0][0]

    original = job.options.exploit_degen
    try:
        job.options.exploit_degen = False
        Biajb_dense_no_tiling, *_ = job._build_soscf_hessian(job.bigERI, occ_C, job.C, moF, so)
    finally:
        job.options.exploit_degen = original

    assert not np.isclose(Biajb_dense[p, p], Biajb_dense_no_tiling[p, p], rtol=1e-3), (
        f"cross-partner coupling doesn't appear to be engaging: "
        f"{Biajb_dense[p, p]} vs {Biajb_dense_no_tiling[p, p]} (no tiling)"
    )
