"""
Tests for SO_RHF.rhf_stability_analysis() (srhf/sorhf.py) -- real singlet
RHF->RHF wavefunction stability analysis.

The formula (the standard 1(A+B) TDHF/RPA matrix) is the same one already
used for the Newton-Raphson step's Hessian in _build_soscf_hessian, just
evaluated over EVERY occupied x virtual orbital pair (not just same-irrep
"active" pairs) with every degenerate-irrep partner independent (no
pooling) -- see rhf_stability_analysis's docstring for the full rationale.
Validated by direct comparison against Psi4's own stability_analysis='check'
output: exact match to 5-6 decimal places for every molecule tested here.

Note: Psi4 only supports the 8 abelian point groups internally, so for
ammonia (true C3v) and methane (true Td) it silently runs in a lower
abelian subgroup and labels roots in THAT subgroup's irreps -- this
codebase's own true non-Abelian irrep labels won't match Psi4's, so this
file compares raw eigenvalues only, never labels.

Also confirmed (source-level, reading Psi4's libscf_solver/rhf.cc and
hf.cc): Psi4 diagonalizes per symmetry block of its own point group and
keeps only min(dim_h, 5) lowest eigenvalues PER BLOCK, unioning across
blocks -- not a global lowest-N. So this file does NOT do a naive
"sort both lists, compare the first K positionally" comparison (that would
be fragile / silently wrong under truncation); instead it (1) checks the
global minimum matches exactly (Psi4's per-block truncation always
includes each block's own lowest value, so the printed list's smallest
entry is always the true global minimum -- immune to the truncation
subtlety and the single most physically important number), and (2)
subset-matches every Psi4-printed value against the full computed
spectrum.
"""
import os
import re
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "srhf"))

import numpy as np
import psi4
import pytest

from srhf.sorhf import SO_RHF
from srhf.options import Options

WATER = """
noreorient
0 1
units bohr
O 0.000000000000 0.000000000000 -0.143225816552
H 0.000000000000 1.638036840407 1.136548822547
H 0.000000000000 -1.638036840407 1.136548822547
"""

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

CASES = {
    "water-sto3g": (WATER, "sto-3g"),
    "ammonia-sto3g": (AMMONIA_STO3G, "sto-3g"),
    "methane-sto3g": (METHANE_STO3G, "sto-3g"),
}


def _psi4_stability_eigenvalues(mol_str, basis):
    """Run Psi4's own stability_analysis='check' and parse the printed
    'Lowest singlet (RHF->RHF) stability eigenvalues:' block live from its
    output file -- not hardcoded reference numbers, matching how every
    other test in this codebase compares against Psi4."""
    fd, out_path = tempfile.mkstemp(suffix=".out")
    os.close(fd)
    try:
        psi4.core.clean()
        psi4.core.clean_options()
        psi4.core.set_output_file(out_path, False)
        psi4.set_options({"basis": basis, "scf_type": "pk", "stability_analysis": "check",
                           "e_convergence": 1e-10, "d_convergence": 1e-10})
        psi4.geometry(mol_str)
        ref_scf = psi4.energy("scf")
        with open(out_path) as f:
            text = f.read()
        match = re.search(r"Lowest singlet \(RHF->RHF\) stability eigenvalues:\n(.*?)\n\s*\n", text, re.DOTALL)
        assert match, "could not find Psi4 stability eigenvalues block in output"
        eigenvalues = [float(x) for x in re.findall(r"[-+]?\d+\.\d+", match.group(1))]
        assert eigenvalues, "found the stability block but parsed no eigenvalues out of it"
        return ref_scf, eigenvalues
    finally:
        os.remove(out_path)


@pytest.fixture(scope="module", params=list(CASES), ids=list(CASES))
def psi4_reference(request):
    mol_str, basis = CASES[request.param]
    ref_scf, eigenvalues = _psi4_stability_eigenvalues(mol_str, basis)
    return request.param, mol_str, basis, ref_scf, eigenvalues


@pytest.mark.parametrize("exploit_degen", [False, True])
def test_rhf_stability_matches_psi4(psi4_reference, exploit_degen):
    name, mol_str, basis, ref_scf, psi4_eigenvalues = psi4_reference

    opts = Options(subgroup=False, exploit_degen=exploit_degen, guess="sad",
                    scf_max_iter=50, e_convergence=1e-11, d_convergence=1e-11,
                    diis=True, second_order=True)
    job = SO_RHF(mol_str, basis, opts)
    job.run()

    assert abs(job.wfn_energy - ref_scf) < 1e-8, (
        f"{name} (exploit_degen={exploit_degen}) SCF energy mismatch: "
        f"{job.wfn_energy} vs psi4 {ref_scf}"
    )

    result = job.rhf_stability_analysis()
    our_eigenvalues = sorted(result.eigenvalues.tolist())
    assert our_eigenvalues, f"{name}: no active occupied-virtual rotations to test"

    # Global minimum: the one comparison immune to Psi4's per-block
    # truncation, and the single most physically important number (is
    # the RHF solution actually a minimum).
    assert np.isclose(our_eigenvalues[0], min(psi4_eigenvalues), atol=1e-4), (
        f"{name} (exploit_degen={exploit_degen}): lowest eigenvalue mismatch -- "
        f"ours={our_eigenvalues[0]} psi4_min={min(psi4_eigenvalues)}"
    )

    # Subset-match every Psi4-printed value against our full spectrum
    # (degrades to a full one-to-one check when Psi4 didn't truncate).
    remaining = list(our_eigenvalues)
    for psi4_val in psi4_eigenvalues:
        closest_idx = min(range(len(remaining)), key=lambda i: abs(remaining[i] - psi4_val))
        assert abs(remaining[closest_idx] - psi4_val) < 1e-4, (
            f"{name} (exploit_degen={exploit_degen}): psi4 eigenvalue {psi4_val} "
            f"has no match in our spectrum {our_eigenvalues}"
        )
        remaining.pop(closest_idx)


def test_rhf_stability_requires_converged_run():
    opts = Options(subgroup=False, exploit_degen=False, guess="sad",
                    scf_max_iter=50, e_convergence=1e-11, d_convergence=1e-11,
                    diis=True, second_order=True)
    job = SO_RHF(WATER, "sto-3g", opts)
    with pytest.raises(RuntimeError):
        job.rhf_stability_analysis()


def test_report_rhf_stability_runs(capsys):
    opts = Options(subgroup=False, exploit_degen=False, guess="sad",
                    scf_max_iter=50, e_convergence=1e-11, d_convergence=1e-11,
                    diis=True, second_order=True)
    job = SO_RHF(WATER, "sto-3g", opts)
    job.run()
    result = job.rhf_stability_analysis()
    job.report_rhf_stability(result)
    out = capsys.readouterr().out
    assert "stability eigenvalues" in out
    assert "stable" in out.lower()


# ---------------------------------------------------------------------------
# Per-irrep label validation, using subgroup= to match Psi4's own (abelian)
# point group exactly
# ---------------------------------------------------------------------------
#
# Psi4 only supports the 8 abelian point groups internally, so it silently
# drops ammonia's true C3v down to Cs, and methane's true Td down to D2
# (confirmed by direct inspection of its output). Running our own SO_RHF
# forced into the SAME subgroup (via the subgroup= option, which any
# molecule here can be run in) makes both codes' irrep labels directly
# comparable -- not just the raw eigenvalue spectrum, as in the tests
# above, but the actual (label, eigenvalue) structure.
#
# This is only meaningful for an ABELIAN point group: every irrep is then
# 1-dimensional, so occ x virt always reduces to exactly ONE irrep (no
# Clebsch-Gordan multiplicity to worry about) -- exactly the case Psi4
# itself always computes in, since it never supports non-Abelian groups
# directly.

ABELIAN_SUBGROUP_CASES = {
    "ammonia-Cs": (AMMONIA_STO3G, "sto-3g", "Cs"),
    "methane-D2": (METHANE_STO3G, "sto-3g", "D2"),
}


def _abelian_product_irrep(symtext, occ_h, virt_h):
    """Direct product irrep of two 1D (abelian-group) irreps -- always a
    single irrep for an abelian group (no Clebsch-Gordan multiplicity),
    found by matching the elementwise character-table product against
    each irrep's own character row."""
    ctab = symtext.character_table
    product_chars = ctab[occ_h] * ctab[virt_h]
    matches = [c for c in range(len(symtext.irreps)) if np.allclose(ctab[c], product_chars)]
    assert len(matches) == 1, (
        f"direct product of irreps {occ_h},{virt_h} isn't a single irrep "
        f"({matches}) -- group isn't abelian?"
    )
    return matches[0]


def _labeled_eigenvalues(job, result):
    """(irrep_symbol, eigenvalue) pairs, one per unit of integer weight in
    each near-degenerate eigenvalue CLUSTER -- must be done at the cluster
    level, not per individual eigenvector: an eigenvector's own weight
    split within a group-REQUIRED degenerate eigenspace is an arbitrary
    np.linalg.eigh basis choice (confirmed empirically -- individual
    eigenvectors within e.g. methane/D2's triply-degenerate clusters come
    out as an arbitrary mix of B1/B2/B3), but the CLUSTER's total weight
    per irrep label lands on an exact integer and is basis-independent /
    physical (that integer is the true degeneracy Td's T2 forces onto D2's
    B1+B2+B3, each appearing with weight exactly 1)."""
    so = job.so_orbitals
    assert all(ir.d == 1 for ir in so.symtext.irreps), "test molecule/subgroup must be abelian"
    n_irreps = len(so.symtext.irreps)
    pair_label = np.array([
        _abelian_product_irrep(so.symtext, ho, hv)
        for ho, hv in zip(result.occ_irrep_of, result.virt_irrep_of)
    ])

    eigvals, eigvecs = result.eigenvalues, result.eigenvectors
    clusters = []
    start = 0
    for k in range(1, len(eigvals)):
        if eigvals[k] - eigvals[start] > 1e-6:
            clusters.append((start, k))
            start = k
    clusters.append((start, len(eigvals)))

    labeled = []
    for lo, hi in clusters:
        avg = float(np.mean(eigvals[lo:hi]))
        sq = np.sum(eigvecs[:, lo:hi] ** 2, axis=1)
        for lam in range(n_irreps):
            w = float(np.sum(sq[pair_label == lam]))
            n_round = round(w)
            if n_round >= 1:
                assert abs(w - n_round) < 1e-4, (
                    f"non-integer cluster weight {w} for irrep {so.symtext.irreps[lam].symbol} "
                    f"at eigenvalue {avg} -- clean per-irrep block decoupling assumption broken"
                )
                labeled += [(so.symtext.irreps[lam].symbol, avg)] * n_round
    labeled.sort(key=lambda x: x[1])
    return labeled


def _psi4_stability_labeled(mol_str, basis):
    fd, out_path = tempfile.mkstemp(suffix=".out")
    os.close(fd)
    try:
        psi4.core.clean()
        psi4.core.clean_options()
        psi4.core.set_output_file(out_path, False)
        psi4.set_options({"basis": basis, "scf_type": "pk", "stability_analysis": "check",
                           "e_convergence": 1e-10, "d_convergence": 1e-10})
        psi4.geometry(mol_str)
        ref_scf = psi4.energy("scf")
        with open(out_path) as f:
            text = f.read()
        match = re.search(r"Lowest singlet \(RHF->RHF\) stability eigenvalues:\n(.*?)\n\s*\n", text, re.DOTALL)
        assert match, "could not find Psi4 stability eigenvalues block in output"
        pairs = re.findall(r"([A-Za-z][\w']*)\s+([-+]?\d+\.\d+)", match.group(1))
        assert pairs
        return ref_scf, [(label, float(val)) for label, val in pairs]
    finally:
        os.remove(out_path)


@pytest.mark.parametrize("name", list(ABELIAN_SUBGROUP_CASES))
def test_rhf_stability_labels_match_psi4_degeneracy_structure(name):
    mol_str, basis, subgroup = ABELIAN_SUBGROUP_CASES[name]

    psi4_ref_scf, psi4_labeled = _psi4_stability_labeled(mol_str, basis)

    opts = Options(subgroup=subgroup, exploit_degen=False, guess="sad",
                    scf_max_iter=50, e_convergence=1e-11, d_convergence=1e-11,
                    diis=True, second_order=True)
    job = SO_RHF(mol_str, basis, opts)
    job.run()
    assert abs(job.wfn_energy - psi4_ref_scf) < 1e-8

    result = job.rhf_stability_analysis()
    our_labeled = _labeled_eigenvalues(job, result)

    # Group each side by eigenvalue cluster and compare MULTIPLICITY (how
    # many distinct-label roots share that eigenvalue), not exact label
    # strings -- Psi4 and molsym use different naming conventions for the
    # same abelian irreps (Psi4's "A"/"B1" vs molsym's "A_1"/"B_1", Psi4's
    # "Ap"/"App" vs molsym's "A'"/"A''"), but the DEGENERACY STRUCTURE is
    # convention-independent and is what actually tests the labeling
    # machinery here.
    def cluster_by_value(labeled_list):
        vals = sorted(v for _, v in labeled_list)
        groups, start = [], 0
        for k in range(1, len(vals)):
            if vals[k] - vals[start] > 1e-4:
                groups.append(vals[start:k])
                start = k
        groups.append(vals[start:])
        return groups

    psi4_groups = cluster_by_value(psi4_labeled)
    our_groups = cluster_by_value(our_labeled)

    # Every Psi4 cluster must be matched (nearest not-yet-consumed) by one
    # of ours with AT LEAST that multiplicity -- ">=" not "==" because
    # Psi4's per-block truncation (min(dim_h, 5) roots per symmetry block,
    # confirmed from its source, see this file's module docstring) can cut
    # off mid-degeneracy: ammonia/Cs's own 15-root spectrum gets truncated
    # to 10 printed roots, right in the middle of the eigenvalue-1.427541
    # pair (confirmed empirically -- Psi4 prints only 1 of that pair's 2
    # degenerate roots). "ours < psi4" would indicate a real bug (missing
    # degeneracy our own reconstruction should have found); "ours > psi4"
    # is expected and fine whenever Psi4 truncated a cluster.
    remaining = list(our_groups)
    for psi4_group in psi4_groups:
        idx = min(range(len(remaining)), key=lambda i: abs(np.mean(remaining[i]) - np.mean(psi4_group)))
        assert abs(np.mean(remaining[idx]) - np.mean(psi4_group)) < 1e-4, (
            f"{name}: no matching eigenvalue cluster for psi4 cluster {psi4_group}"
        )
        assert len(remaining[idx]) >= len(psi4_group), (
            f"{name}: degeneracy mismatch at eigenvalue ~{np.mean(psi4_group):.6f} -- "
            f"psi4 has {len(psi4_group)} distinct-label roots, ours only has {len(remaining[idx])}"
        )
        remaining.pop(idx)
