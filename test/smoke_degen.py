"""
End-to-end SCF smoke test against molecules with degenerate irreps.

Not collected by pytest (filename doesn't match test_*.py) since it runs real
Psi4 SCF calculations and is meant for manual verification after touching
rhf.py / srhf_helper.py, not for the fast unit test suite.

Water (C2v) is deliberately NOT used here: C2v is abelian, so every irrep is
1-dimensional and code paths gated on degenerate irreps (degen_rhf_energy's
per-irrep weighting, DPD.lookup_degen() and its degen_bra/ket/braket
repacking) never actually execute. Methane (Td: E, T2 irreps) and ammonia
(C3v: E irrep) do exercise those paths.

Run with:
    conda activate p4dev && module load psi4/nightly
    python test/smoke_degen.py
"""
import psi4
from srhf.rhf import SRHF
from srhf.options import Options

molecules = {
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
psi4.set_options({"basis": basis, "scf_type": "pk", "e_convergence": 1e-10, "d_convergence": 1e-10})

for name, mol_str in molecules.items():
    print(f"\n{'='*70}\n{name}\n{'='*70}")

    # Psi4 reference
    psi4.geometry(mol_str)
    ref_energy = psi4.energy("scf")
    print(f"Psi4 reference RHF energy: {ref_energy:.10f}")

    for guess in ("core", "gwh", "sad"):
        opts = Options(
            subgroup=False,
            exploit_degen=True,
            guess=guess,
            scf_max_iter=50,
            e_convergence=1e-10,
            d_convergence=1e-10,
            diis=True,
            sparse_transform=False,
            mp2=False,
        )
        job = SRHF(mol_str, basis, opts)
        job.run()
        diff = job.wfn_energy - ref_energy
        print(f"SRHF guess={guess:5s} energy: {job.wfn_energy:.10f}  (diff vs psi4: {diff:.2e})")
        assert abs(diff) < 1e-7, f"MISMATCH for {name} guess={guess}: {diff}"

print("\nALL DEGENERATE-IRREP CHECKS PASSED")
