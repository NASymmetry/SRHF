"""
End-to-end SCF smoke test for the DegenTensor-based ERI compression + Fock
build (options.degen_tensor=True), cross-checked against both Psi4 and
today's default (sparse_transform=True) code path.

Not collected by pytest (filename doesn't match test_*.py) since it runs real
Psi4 SCF calculations and is meant for manual verification after touching
srhf/degen_tensor.py or srhf/rhf.py's degen_tensor dispatch, not for the fast
unit test suite.

Unlike test/smoke_degen.py, water (C2v) IS included here deliberately: it's
the first time this specific new code path's "no real degeneracy" branch
gets exercised. Methane/cc-pVDZ (Td: E, T1, T2 all simultaneously
populated) is the case that actually exposed a bug during development: an
earlier version of DegenIntegralFactory special-cased only bra_irrep==
ket_irrep (e.g. T2 x T2 self-paired blocks), which happened to be correct
for methane/STO-3G and ammonia (each only ever populates ONE degenerate
irrep, so they can't produce a cross-irrep degenerate block at all) but was
wrong for cross-irrep degenerate-both blocks like E x T2 -- the correct
condition is "both bra and ket are SOME degenerate irrep", not irrep
equality. See srhf/degen_tensor.py's DegenIntegralFactory._make_block for
the fix and test/test_degen_integral_factory.py for the block-level
regression tests. Methane/STO-3G and ammonia are kept here too since they're
cheap and still real regression coverage of the "single/self-paired
degenerate irrep" cases.

Run with:
    conda activate p4dev && module load psi4/nightly
    python test/smoke_degen_tensor.py
"""
import psi4
from srhf.rhf import SRHF
from srhf.options import Options

molecules = {
    "water (C2v)": ("sto-3g", """
noreorient
0 1
units bohr
O 0.000000000000 0.000000000000 -0.143225816552
H 0.000000000000 1.638036840407 1.136548822547
H 0.000000000000 -1.638036840407 1.136548822547
"""),
    "methane/STO-3G (Td, only T2 populated)": ("sto-3g", """
noreorient
0 1
units bohr
C       0.00000000     0.00000000     0.00000000
H       1.18813758    -1.18813758     1.18813758
H      -1.18813758     1.18813758     1.18813758
H       1.18813758     1.18813758    -1.18813758
H      -1.18813758    -1.18813758    -1.18813758
"""),
    "methane/cc-pVDZ (Td, E+T1+T2 populated, cross-irrep blocks)": ("cc-pvdz", """
noreorient
0 1
units bohr
C       0.00000000     0.00000000     0.00000000
H       1.18813758    -1.18813758     1.18813758
H      -1.18813758     1.18813758     1.18813758
H       1.18813758     1.18813758    -1.18813758
H      -1.18813758    -1.18813758    -1.18813758
"""),
    "ammonia (C3v)": ("sto-3g", """
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

    psi4.set_options({"basis": basis, "scf_type": "pk", "e_convergence": 1e-10, "d_convergence": 1e-10})
    psi4.geometry(mol_str)
    ref_energy = psi4.energy("scf")
    print(f"Psi4 reference RHF energy: {ref_energy:.10f}")
    psi4.core.clean()

    for guess in ("core", "gwh", "sad"):
        common = dict(
            subgroup=False,
            exploit_degen=True,
            guess=guess,
            scf_max_iter=50,
            e_convergence=1e-10,
            d_convergence=1e-10,
            diis=True,
            mp2=False,
        )

        job_default = SRHF(mol_str, basis, Options(sparse_transform=True, degen_tensor=False, **common))
        job_default.run()

        job_degen_tensor = SRHF(mol_str, basis, Options(sparse_transform=False, degen_tensor=True, **common))
        job_degen_tensor.run()

        diff_vs_psi4 = job_degen_tensor.wfn_energy - ref_energy
        diff_vs_default = job_degen_tensor.wfn_energy - job_default.wfn_energy
        print(
            f"guess={guess:5s} degen_tensor energy: {job_degen_tensor.wfn_energy:.10f}  "
            f"(diff vs psi4: {diff_vs_psi4:.2e}, diff vs default path: {diff_vs_default:.2e})"
        )
        assert abs(diff_vs_psi4) < 1e-7, f"MISMATCH vs psi4 for {name} guess={guess}: {diff_vs_psi4}"
        # Looser than the psi4 comparison deliberately: two structurally
        # different SCF+DIIS code paths converging to the same physics can
        # differ by more than float64 noise in the last few bits without
        # indicating a bug (observed ~1e-8 for ammonia/C3v specifically,
        # consistent across guess types and code paths all session).
        assert abs(diff_vs_default) < 1e-7, f"MISMATCH vs default path for {name} guess={guess}: {diff_vs_default}"

print("\nALL DEGEN_TENSOR CHECKS PASSED")
