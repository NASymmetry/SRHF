"""
End-to-end MP2 smoke test for MP2.run_degen_tensor(), the exploit_degen=True-
aware correlation energy path, cross-checked against Psi4's own conventional
(non-density-fitted) MP2 correlation energy. mp2_type must be "conv" here --
Psi4 defaults to density-fitted MP2 regardless of scf_type, which alone
introduces ~1e-5 Eh disagreement unrelated to SRHF correctness.

Mirrors test/smoke_mp2.py's conventions (bare-import style, since mp2.py's
own `from bdmats import BDMatrix` is a bare import -- mixing that with
`srhf.bdmats.BDMatrix` would create two distinct BDMatrix classes and break
isinstance() checks inside BDMatrix.einsum()). Unlike smoke_mp2.py, this file
runs run_degen_tensor() (not run_symm()) under BOTH exploit_degen=True and
exploit_degen=False, and deliberately includes methane/cc-pVDZ, whose E, T1,
and T2 irreps are all populated -- the case that exposed the original,
per-irrep-pair-block-decomposed design as wrong (see run_degen_tensor()'s
docstring in srhf/mp2.py for the full history).

guess="sad" is used rather than "core": methane/cc-pVDZ's core guess was
found to converge to a state ~0.016 Eh above the true SCF minimum (a
pre-existing SRHF SCF-guess issue, unrelated to this feature) -- sad
converges to the correct minimum for all molecules tested here.

Not collected by pytest (filename doesn't match test_*.py) since it runs real
Psi4 SCF/MP2 calculations and is meant for manual verification, not the fast
unit test suite.

Run with:
    conda activate p4dev && module load psi4/nightly
    python test/smoke_mp2_degen_tensor.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "srhf"))

import psi4
from rhf import SRHF
from mp2 import MP2
from options import Options

molecules = {
    "water (C2v, cc-pVDZ, no degenerate irreps)": ("cc-pvdz", """
noreorient
0 1
units bohr
O 0.000000000000 0.000000000000 -0.143225816552
H 0.000000000000 1.638036840407 1.136548822547
H 0.000000000000 -1.638036840407 1.136548822547
"""),
    "methane (Td, STO-3G, only T2 populated)": ("sto-3g", """
noreorient
0 1
units bohr
C       0.00000000     0.00000000     0.00000000
H       1.18813758    -1.18813758     1.18813758
H      -1.18813758     1.18813758     1.18813758
H       1.18813758     1.18813758    -1.18813758
H      -1.18813758    -1.18813758    -1.18813758
"""),
    "methane (Td, cc-pVDZ, E+T1+T2 populated, cross-irrep blocks)": ("cc-pvdz", """
noreorient
0 1
units bohr
C       0.00000000     0.00000000     0.00000000
H       1.18813758    -1.18813758     1.18813758
H      -1.18813758     1.18813758     1.18813758
H       1.18813758     1.18813758    -1.18813758
H      -1.18813758    -1.18813758    -1.18813758
"""),
    "ammonia (C3v, STO-3G, self-paired E)": ("sto-3g", """
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

for name, (basis, mol_str) in molecules.items():
    print(f"\n{'='*70}\n{name}\n{'='*70}")

    psi4.core.clean()
    psi4.core.clean_options()
    psi4.set_options({"basis": basis, "scf_type": "pk", "mp2_type": "conv",
                       "e_convergence": 1e-10, "d_convergence": 1e-10})
    psi4.geometry(mol_str)
    ref_scf = psi4.energy("scf")
    ref_mp2_total = psi4.energy("mp2")
    ref_corr = ref_mp2_total - ref_scf
    print(f"Psi4 SCF energy:  {ref_scf:.10f}")
    print(f"Psi4 MP2 corr:    {ref_corr:.10f}")

    results = {}
    for exploit_degen in (False, True):
        opts = Options(
            subgroup=False,
            exploit_degen=exploit_degen,
            guess="sad",
            scf_max_iter=100,
            e_convergence=1e-10,
            d_convergence=1e-10,
            diis=True,
            sparse_transform=False,
            degen_tensor=False,
            mp2=True,
        )
        job = SRHF(mol_str, basis, opts)
        job.run()

        post_hf = MP2(job.molecule, job.options, job.so_orbitals, job.ERI, job.repacked_bigERI)
        srhf_corr = post_hf.run_degen_tensor()
        results[exploit_degen] = srhf_corr
        diff = srhf_corr - ref_corr
        print(f"SRHF MP2 corr (exploit_degen={exploit_degen}): {srhf_corr:.10f}  (diff vs psi4: {diff:.2e})")
        assert abs(diff) < 1e-9, f"MISMATCH for {name} (exploit_degen={exploit_degen}): {diff}"

    consistency_diff = results[True] - results[False]
    print(f"exploit_degen=True vs False diff: {consistency_diff:.2e}")
    assert abs(consistency_diff) < 1e-9, f"exploit_degen inconsistency for {name}: {consistency_diff}"

print("\nALL MP2 (run_degen_tensor) CHECKS PASSED")
