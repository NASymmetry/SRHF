"""
End-to-end MP2 smoke test, cross-checked against Psi4's own conventional
(non-density-fitted) MP2 correlation energy. mp2_type must be "conv" here --
Psi4 defaults to density-fitted MP2 regardless of scf_type, which alone
introduces ~1e-5 Eh disagreement unrelated to SRHF correctness.

Uses run.py's bare-import style (srhf/ itself on sys.path) rather than the
srhf.* package-qualified imports used elsewhere, because mp2.py's own
`from bdmats import BDMatrix` is a bare import -- mixing that with
`srhf.bdmats.BDMatrix` would create two distinct BDMatrix classes and break
isinstance() checks inside BDMatrix.einsum().

Runs with exploit_degen=False: run_symm()/run_symm_block() don't yet support
exploit_degen=True on a point group with a genuinely degenerate irrep (see
the KNOWN LIMITATION note at the top of mp2.py) -- that combination raises a
shape-mismatch ValueError, so it's not exercised by this regression check.

Not collected by pytest (filename doesn't match test_*.py) since it runs real
Psi4 SCF/MP2 calculations and is meant for manual verification, not the fast
unit test suite.

Run with:
    conda activate p4dev && module load psi4/nightly
    python test/smoke_mp2.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "srhf"))

import psi4
from rhf import SRHF
from mp2 import MP2
from options import Options

molecules = {
    "water (C2v)": """
noreorient
0 1
units bohr
O 0.000000000000 0.000000000000 -0.143225816552
H 0.000000000000 1.638036840407 1.136548822547
H 0.000000000000 -1.638036840407 1.136548822547
""",
    "methane (Td)": """
noreorient
0 1
units bohr
C       0.00000000     0.00000000     0.00000000
H       1.18813758    -1.18813758     1.18813758
H      -1.18813758     1.18813758     1.18813758
H       1.18813758     1.18813758    -1.18813758
H      -1.18813758    -1.18813758    -1.18813758
""",
    "ammonia (C3v)": """
noreorient
0 1
units bohr
N       0.00000000     0.00000000     0.13125886
H      -0.88122565    -1.52632759    -0.60791885
H      -0.88122565     1.52632759    -0.60791885
H       1.76245129    -0.00000000    -0.60791885
""",
}

basis = "sto-3g"
psi4.core.be_quiet()
psi4.set_options({"basis": basis, "scf_type": "pk", "mp2_type": "conv", "e_convergence": 1e-10, "d_convergence": 1e-10})

for name, mol_str in molecules.items():
    print(f"\n{'='*70}\n{name}\n{'='*70}")

    psi4.geometry(mol_str)
    ref_scf = psi4.energy("scf")
    ref_mp2_total = psi4.energy("mp2")
    ref_corr = ref_mp2_total - ref_scf
    print(f"Psi4 SCF energy:  {ref_scf:.10f}")
    print(f"Psi4 MP2 corr:    {ref_corr:.10f}")
    psi4.core.clean()

    opts = Options(
        subgroup=False,
        exploit_degen=False,
        guess="core",
        scf_max_iter=50,
        e_convergence=1e-10,
        d_convergence=1e-10,
        diis=True,
        sparse_transform=False,
        mp2=True,
    )
    job = SRHF(mol_str, basis, opts)
    job.run()

    post_hf = MP2(job.molecule, job.options, job.so_orbitals, job.ERI, job.repacked_bigERI)
    srhf_corr = post_hf.run_symm()
    diff = srhf_corr - ref_corr
    print(f"SRHF MP2 corr:    {srhf_corr:.10f}  (diff vs psi4: {diff:.2e})")
    assert abs(diff) < 1e-9, f"MISMATCH for {name}: {diff}"

print("\nALL MP2 CHECKS PASSED")
