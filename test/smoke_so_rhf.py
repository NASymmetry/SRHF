"""
End-to-end smoke test for SO_RHF (srhf/sorhf.py), the second-order
(Newton-Raphson/SOSCF) SCF class, cross-checked against Psi4's own RHF SCF
energy.

SO_RHF had zero prior test coverage or validated-convergence history for ANY
molecule or point group (confirmed by searching git history/test/ before
writing this file) -- fixing it surfaced a chain of pre-existing bugs, all
rooted in the same cause: the Newton-Raphson (SOSCF) step was apparently
never exercised for real (non-C1) point-group symmetry before:

  - BDMatrix.full_to_bd (srhf/bdmats.py) only sliced the first two axes of
    self, silently leaving axes 2-3 of a 4-index ERI tensor at the full,
    un-sliced AO dimension -- masked for C1 (single irrep spanning
    everything) since slicing "the whole thing" trivially matches the
    unsliced axes anyway.
  - sorhf.py's Newton step used the raw, untransformed AO ERI instead of the
    already-computed SO-transformed bigERI.
  - eye_diag_occ/eye_diag_virt (used to build the orbital-rotation Hessian)
    were built as a single block from so_orbitals.Orbs[0] only, instead of
    one identity block per irrep -- again a C1-only assumption (irrep 0 is
    the only irrep that exists in C1).
  - BDMatrix.reshape(self, *args) expects one shape spec per irrep as
    separate positional arguments (see test/test_bdmats.py's calling
    convention); sorhf.py called it with the whole per-irrep shape list as
    ONE argument, which "worked" only for C1 because ndarray.reshape also
    accepts a single list-of-ints argument -- silently wrong once there's
    more than one irrep.
  - sorhf.py's guess="sad" branch never called so_orbitals.ndocc_irrep(C,
    eps), leaving Orbs[h].ndocc_ir as None.
  - sorhf.py imported BDMatrix/DIIS_Manager/SOrbitals/DPD bare instead of
    package-qualified (from srhf.bdmats import BDMatrix, matching rhf.py),
    creating two distinct BDMatrix classes and breaking isinstance() checks.
  - BDMatrix.__mul__'s scalar check (`type(n) is int or float`, always
    truthy due to operator precedence) wrongly took the scalar-multiply
    branch for BDMatrix * BDMatrix.

None of these are specific to any one point group -- they're all "this loop/
slice/reshape only ever saw one irrep" bugs, so this file deliberately
exercises multiple molecules spanning abelian-multi-irrep (water, C2v),
non-abelian (methane, Td; ammonia, C3v), and a genuinely degenerate irrep
(ammonia's E) under both exploit_degen settings.

A first fix (block-diagonal-by-irrep Hessian -- BDMatrix.einsum contracts
every operand at the same irrep h) got convergence correct but slow: C1
converged in ~5 Newton iterations, real point-group symmetry took ~13, and
exploit_degen=True took ~24 (methane/STO-3G), because the Hessian
structurally excluded coupling between same-irrep occupied-virtual rotation
pairs of DIFFERENT irreps (h != h') -- coupling that's generically nonzero
by the standard two-electron selection rule (h⊗h always contains the
totally symmetric irrep for ANY irrep h, true even for abelian point groups
with no degenerate irreps at all). SO_RHF._build_soscf_hessian() now builds
a single DENSE Hessian spanning every irrep's active rotation pairs at once
(see its docstring in sorhf.py for the full design), restoring true 2nd-
order convergence for exploit_degen=False. exploit_degen=True initially
still converged faster than before but not to C1 parity -- a naive
"multiply by degen_h" chain-rule fix would have been WRONG (verified
numerically: the true Hessian has a genuine cross-PARTNER coupling term
within a degenerate irrep, not just a scalar factor -- see sorhf.py's
_build_soscf_hessian docstring for the full derivation). That's now fixed
too: _build_soscf_hessian tiles every irrep degen_h times (mirroring
MP2.run_degen_tensor()'s recipe) and pools the tiled Hessian/gradient back
down to one entry per representative pair, closing the gap -- confirmed
here to restore near-C1 convergence for exploit_degen=True as well,
including methane/cc-pVDZ (E+T1+T2 all simultaneously degenerate and
populated), which went from 28 iterations to matching/beating its own C1
baseline.

The dense-Hessian rewrite itself had one bug, caught by methane/cc-pVDZ
specifically: _build_soscf_hessian filtered occ_C.blocks/C.blocks/
moF.blocks by each block's own .size to decide which irreps were
"populated" -- correct for C.blocks/moF.blocks (always square, irreplength
x irreplength, so .size==0 iff the irrep truly has zero AOs), but wrong for
occ_C.blocks specifically: an irrep can be populated with real AO functions
while having ZERO occupied orbitals (e.g. cc-pVDZ methane's E/T1 irreps),
giving an (irreplength, 0)-shaped occ_C block whose .size is 0 even though
the irrep isn't empty. Filtering occ_C_combined by that condition silently
dropped those irreps' rows, desyncing occ_C_combined's dimension from
I_dense/C_combined's shared one -- fixed by filtering all three by
irreplength[h] > 0 (matching I_dense's own indexing) instead.
This file's iteration-count assertions are calibrated to that: tight for
exploit_degen=False (close to each molecule's own C1 baseline), loose for
exploit_degen=True (only asserts real improvement, not parity).

See test/test_soscf_hessian.py for the unit-level validation of the dense
Hessian itself (finite-difference gold standard, diagonal-block regression
against DegenIntegralFactory, and an "off-diagonal isn't trivially zero"
engagement check) -- this file is the end-to-end/Psi4-comparison layer.

Run with:
    conda activate p4dev && module load psi4/nightly
    python test/smoke_so_rhf.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "srhf"))

import psi4
from sorhf import SO_RHF
from options import Options

molecules = {
    "methane (Td, STO-3G)": ("sto-3g", False, """
noreorient
0 1
units bohr
C       0.00000000     0.00000000     0.00000000
H       1.18813758    -1.18813758     1.18813758
H      -1.18813758     1.18813758     1.18813758
H       1.18813758     1.18813758    -1.18813758
H      -1.18813758    -1.18813758    -1.18813758
"""),
    "methane (Td, cc-pVDZ, some irreps populated with zero occupied orbitals)": ("cc-pvdz", False, """
noreorient
0 1
units bohr
C       0.00000000     0.00000000     0.00000000
H       1.18813758    -1.18813758     1.18813758
H      -1.18813758     1.18813758     1.18813758
H       1.18813758     1.18813758    -1.18813758
H      -1.18813758    -1.18813758    -1.18813758
"""),
    "water (C2v, STO-3G)": ("sto-3g", False, """
noreorient
0 1
units bohr
O 0.000000000000 0.000000000000 -0.143225816552
H 0.000000000000 1.638036840407 1.136548822547
H 0.000000000000 -1.638036840407 1.136548822547
"""),
    "ammonia (C3v, STO-3G)": ("sto-3g", False, """
noreorient
0 1
units bohr
N       0.00000000     0.00000000     0.13125886
H      -0.88122565    -1.52632759    -0.60791885
H      -0.88122565     1.52632759    -0.60791885
H       1.76245129    -0.00000000    -0.60791885
"""),
}

psi4.core.be_quiet()

for name, (basis, subgroup, mol_str) in molecules.items():
    print(f"\n{'='*70}\n{name}\n{'='*70}")

    psi4.core.clean()
    psi4.core.clean_options()
    psi4.set_options({"basis": basis, "scf_type": "pk",
                       "e_convergence": 1e-10, "d_convergence": 1e-10})
    psi4.geometry(mol_str)
    ref_scf = psi4.energy("scf")
    print(f"Psi4 SCF energy: {ref_scf:.10f}")

    def run_job(subgroup_, exploit_degen):
        opts = Options(
            subgroup=subgroup_,
            exploit_degen=exploit_degen,
            guess="sad",
            scf_max_iter=50,
            e_convergence=1e-10,
            d_convergence=1e-10,
            diis=True,
            second_order=True,
        )
        job = SO_RHF(mol_str, basis, opts)
        job.run()
        return job

    # C1 baseline: only one irrep, so the dense Hessian is exactly the full
    # Hessian (no coupling is possible to miss) -- true 2nd-order Newton
    # convergence. Computed per-molecule rather than hardcoded so this
    # doesn't silently go stale if convergence criteria change.
    c1_job = run_job("C1", exploit_degen=False)
    c1_iters = c1_job.n_iterations
    print(f"C1 baseline iterations: {c1_iters}")

    for exploit_degen in (False, True):
        job = run_job(subgroup, exploit_degen)

        diff = job.wfn_energy - ref_scf
        print(f"SO_RHF SCF energy (exploit_degen={exploit_degen}): {job.wfn_energy:.10f}  "
              f"(diff vs psi4: {diff:.2e}, {job.n_iterations} iterations)")
        assert abs(diff) < 1e-8, f"MISMATCH for {name} (exploit_degen={exploit_degen}): {diff}"

        if not exploit_degen:
            # Tight: the dense cross-irrep Hessian should make real point-group
            # symmetry converge at essentially the same rate as C1.
            assert job.n_iterations <= c1_iters + 2, (
                f"{name} (exploit_degen=False) took {job.n_iterations} iterations, "
                f"expected close to the C1 baseline of {c1_iters} -- the dense "
                f"cross-irrep Hessian may not be engaging correctly"
            )
        else:
            # Tight, same as exploit_degen=False: _build_soscf_hessian now
            # also sums cross-PARTNER coupling within degenerate irreps
            # (tiling each irrep degen_h times and pooling back to one
            # entry per representative pair -- see its docstring in
            # sorhf.py), closing the previously-deferred degen_h gap.
            # Verified this restores near-C1 convergence even for the
            # hardest case here (methane/cc-pVDZ, E+T1+T2 all
            # simultaneously degenerate and populated): was 28 iterations
            # before this fix, now matches/beats the C1 baseline.
            assert job.n_iterations <= c1_iters + 2, (
                f"{name} (exploit_degen=True) took {job.n_iterations} iterations, "
                f"expected close to the C1 baseline of {c1_iters} -- the "
                f"cross-partner Hessian coupling may not be engaging correctly"
            )

print("\nALL SO_RHF CHECKS PASSED")
