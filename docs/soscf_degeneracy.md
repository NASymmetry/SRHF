# Exploiting degeneracy in second-order SCF (`SO_RHF`)

This document explains how `srhf/sorhf.py`'s `SO_RHF` (the Newton-Raphson /
"second-order" SCF class) uses point-group symmetry, why the naive way of
compressing a degenerate irrep breaks the Newton step, and what it actually
takes to make `options.exploit_degen=True` converge as fast as not
exploiting degeneracy at all. It's a companion to
[`docs/degen_tensor.md`](degen_tensor.md) (the analogous story for the
Hartree-Fock Fock build and MP2) — the underlying symmetry facts are the
same, but SOSCF's *Hessian* runs into a subtlety neither of those needed to
solve.

## 1. What "second order" means here

Ordinary SCF (`SRHF`, `srhf/rhf.py`) finds the ground-state orbitals by
repeatedly diagonalizing the Fock matrix and mixing in DIIS extrapolation —
a *first-order* method: each step only uses the current gradient (how far
each occupied-virtual pair is from the Brillouin condition `F_ia = 0`), not
how the gradient itself is curving.

`SO_RHF` instead takes a genuine **Newton-Raphson step** in the space of
occupied-virtual orbital rotations: given the gradient `g` (how much the
energy changes to first order under a small rotation) and the Hessian `H`
(how the *gradient itself* changes — the energy's second derivative), it
solves `H·x = g` and rotates the orbitals by `x` directly toward the
stationary point. Done exactly, this converges quadratically — very few
iterations, independent of point-group symmetry. Everything in this
document is about what it takes to build `H` *correctly* once symmetry
compression is involved, since building `H` wrong doesn't break
correctness (the code still finds the right answer — see below) but does
throw away the whole point of using a second-order method in the first
place: speed.

## 2. What `exploit_degen=True` actually compresses

For a **degenerate** irrep (dimension `d_h > 1` — e.g. `E` in C3v, `T1`/`T2`
in Td), the point group doesn't just have one copy of that irrep's orbitals
— it has `d_h` symmetry-equivalent *partners*. `options.exploit_degen=True`
stores only **one representative partner** per degenerate irrep
(`so_orbitals.C.blocks[h]` has shape `(irreplength[h], irreplength[h])`,
not `(d_h·irreplength[h], d_h·irreplength[h])`), and recovers physical
quantities that need the full picture by multiplying by `degen_h` where
appropriate — e.g. the total electronic energy:

```python
# SO_RHF.degen_rhf_energy, srhf/sorhf.py
degen = SOrbs.Orbs[h].degen if self.options.exploit_degen else 1
E += degen * np.sum(d * HF.blocks[h])
```

This is safe for the energy because every partner's contribution is
*identical* (the Hamiltonian can't distinguish between symmetry-equivalent
partners), so "one value times `degen_h`" is exact. The question this whole
document is really about: **does that same "compute once, multiply by
`degen_h`" trick work for the Newton step's gradient and Hessian too?** The
answer turns out to be "yes for the gradient, no for the Hessian" — and
understanding *why* is the whole story.

## 3. The rotation parameters: always one shared value per representative pair

Regardless of `exploit_degen`, the only occupied-virtual pairs `(i,a)` that
can have a nonzero gradient are pairs where `i` and `a` belong to the
**same irrep** — a standard fact for a closed-shell singlet reference (an
occ-virt rotation mixing different irreps isn't a stationary-symmetry-
preserving direction, so its first-order energy change is exactly zero).
`gn`/`x`/`U` are all sized `ndocc_h × nvirt_h` per irrep — the *compressed*
occupation counts — never `d_h` times that, whether or not degeneracy is
exploited. Physically: a rotation parameter for a degenerate irrep doesn't
rotate "the representative partner" in isolation — by symmetry, the *only*
rotation that preserves the point group's symmetry is one applied
**identically to all `d_h` partners simultaneously**. That's what
"exploiting degeneracy" means for the *parameters* — nothing new needs to
be built here, it's already how `so_orbitals.C`/`Orbs[h].ndocc_ir` are
stored.

## 4. Bug #1 (fixed earlier): the Hessian was block-diagonal by irrep

The original `Biajb` (orbital-rotation Hessian) was built via `BDMatrix`,
whose every method assumes "one block = one irrep `h`, uniformly across
every axis" (see `docs/degen_tensor.md` §2 for the general shape of this
assumption). That structurally **excludes** any coupling between two
same-irrep rotation pairs belonging to *different* irreps. But by character
orthogonality, `h⊗h` always contains the totally symmetric irrep for *any*
irrep `h` — so two same-irrep pairs from different irreps `h ≠ h'`
generically *do* couple through the two-electron integral `(ia|jb)`. This
holds even for abelian point groups with **no degenerate irreps at all**
(water, C2v) — it has nothing to do with degeneracy specifically.

Crucially, this didn't break *correctness*: `SO_RHF` had never been
exercised for real point-group symmetry before this investigation (zero
prior test coverage, confirmed via git history), and the block-diagonal
Hessian, while missing real physics, still gave a valid **descent
direction** for Newton's method — the fixed point (`gn = 0`) doesn't care
which Hessian approximation produced the step, only whether the gradient
is exact. It converged to the right energy every time, just slowly (13
iterations for methane/Td vs. 5 for the same molecule forced into C1
symmetry, where there's only one irrep and so no coupling could possibly
be missing).

**The fix** (`SO_RHF._build_soscf_hessian`): stop building the Hessian
per-irrep-block. Instead, combine every irrep's representative-partner
block into one dense array via `scipy.linalg.block_diag`, transform the
already-computed dense `bigERI` with it in a single `np.einsum`, and gather
the Hessian's entries directly by index:

```python
Biajb_dense = 4.0 * (fock_term + 4.0 * ia_jb - ib_ja - ij_ab)
```

where `ia_jb`/`ij_ab`/`ib_ja` are gathered from one shared `MO_dense`
tensor via fancy indexing — `(ia|jb)`, `(ij|ab)`, `(ib|ja)` for every pair
of active rotation parameters `p=(i,a)`, `q=(j,b)`, regardless of which
irrep each belongs to. No selection-rule reasoning is needed; building one
dense tensor and indexing into it captures every real coupling
automatically. `fock_term` (`δ_ij F_ab − δ_ab F_ij`) is included too, but —
important for what follows — it's **always exactly zero** whenever `p` and
`q` don't share the literal same MO index, since `δ_ij`/`δ_ab` test index
*equality*, not just same-irrep membership.

## 5. Bug #2: the `degen_h` gap (why `exploit_degen=True` was still slow)

Fixing cross-irrep coupling closed most of the gap (`exploit_degen=False`
now converges in as few iterations as forcing C1 symmetry, for every point
group tested). But `exploit_degen=True` was still noticeably slower:

| Molecule | Before any fix | Cross-irrep fix only | C1 baseline |
|---|---|---|---|
| Methane/STO-3G, `exploit_degen=True` | 24 | 18 | 5 |
| Ammonia/STO-3G, `exploit_degen=True` | 24 | 13 | 6 |
| Methane/cc-pVDZ, `exploit_degen=True` | (crashed) | 28 | 7 |

The reason: `degen_rhf_energy` weights the *energy* by `degen_h`, but
neither `gn` nor `Biajb_dense` applied any equivalent factor to the
gradient or Hessian. Since a rotation parameter for a degenerate irrep
represents an identical rotation applied to *all* `degen_h` partners at
once (§3), the energy's dependence on that shared parameter should, by the
chain rule, also carry a `degen_h`-sized effect — and it wasn't being
accounted for anywhere.

### Why "just multiply by `degen_h`" is wrong for the Hessian

The tempting fix — multiply `gn` and `Biajb_dense` by `degen_h` for
degenerate-irrep entries — is correct for the gradient but **quantitatively
wrong** for the Hessian. This was confirmed numerically (ammonia's `E`
irrep, `degen_h=2`), by building a fully-uncompressed reference (every
partner's data explicit, no compression) and computing the *true* energy
curvature for a rotation applied identically to both partners:

```
Hpp_shared (true, both partners rotated together) = 10.676
2 × Hpp_single_partner (what "just multiply by degen_h" would give) = 9.083
```

Not even close — the naive fix under-counts by about 15%. Decomposing the
true value: `Hpp_shared = degen_h · Hpp_single_partner + degen_h·(degen_h−1) · H_cross_partner`,
where `H_cross_partner ≈ +0.797` is a **previously entirely uncomputed
quantity** — the curvature contribution from partner 0's rotation coupling
to partner 1's rotation of *the same representative pair*. This is
structurally the *same* phenomenon as the cross-irrep coupling in §4, just
occurring **within** a degenerate irrep, between its own partners, rather
than between different irreps. The reasoning is identical too: `fock_term`
is exactly zero for a cross-partner combination (partner 0's occupied index
and partner 1's occupied index are never the *same* MO index, even though
they're "the same orbital" physically), so only the two-electron term
`4(ia|jb) − (ij|ab) − (ib|ja)` survives — and it's generically nonzero for
the same character-orthogonality reason as before.

The gradient has no such complication: a first derivative of a sum has no
cross terms (`d/dκ Σ_μ f(κ_μ) = Σ_μ f'(κ_μ)`, no product rule involved), so
`dE/dκ_shared = degen_h · dE/dκ_representative` holds exactly — confirmed
numerically (partner 0's and partner 1's individual gradients matched to 9
significant figures once compared correctly, see the phase note below).

### A red herring worth recording: eigenvector phase, not group theory

An early attempt to verify this by independently diagononalizing an
**uncompressed** (`exploit_degen=False`) reference and comparing partner
0's vs. partner 1's eigenvectors gave a confusing result — the "shared"
gradient came out identically zero, which looked like it might mean the
correct combination needs the point group's actual `d×d` irreducible
representation matrices (`symtext.irrep_mats`, confirmed available via
`molsym` — real matrices, verified unitary with correct group closure) to
combine partners correctly, rather than simple identical tiling.

That turned out to be an artifact of the test, not a real effect:
`numpy.linalg.eigh` returns an *arbitrary* orthonormal basis for a
degenerate eigenspace, so independently diagonalizing "both partners
together" gives partner 0 and partner 1 relatively-phased eigenvectors
with no fixed relationship. The **actual** compressed representation never
independently diagonalizes more than the one representative partner's
block in the first place — partners 1..`degen_h−1` are always *implicit
tiled copies* of partner 0 (exactly the same assumption
`MP2.run_degen_tensor()` and `DegenIntegralFactory` already rely on), so
there's no phase ambiguity to resolve in the real code. Redoing the
comparison with a properly *tiled* reference (copy partner 0's own,
already-fixed-phase data into partner 1's AO subspace, rather than
re-diagonalizing) gave a clean, self-consistent answer with no group
representation matrices needed. The lesson: identical tiling across
partners is the physically correct convention here — it just has to be
validated against a reference that doesn't introduce its own artificial
ambiguity.

## 6. The fix: tile, then pool

`_build_soscf_hessian` now builds a **tiled** combined space — every
irrep's representative-partner block duplicated `degen_h` times (`1` if
`exploit_degen=False`), located via the same "full SALC offset" bookkeeping
`DegenIntegralFactory` already uses for the analogous problem:

```python
for h in populated:
    degen_h = so_orbitals.symtext.irreps[h].d if self.options.exploit_degen else 1
    il_h = irreplength[h]
    for mu in range(degen_h):
        idx_tiled_parts.append(np.arange(full_offsets[h] + mu * il_h, full_offsets[h] + (mu + 1) * il_h))
        occ_C_tiled_blocks.append(occ_C.blocks[h])   # same representative block, tiled
        C_tiled_blocks.append(C.blocks[h])
        moF_tiled_blocks.append(moF.blocks[h])
```

The Hessian/gradient formula (§4) runs **unchanged** in this larger, tiled
space — every `(representative pair, partner)` combination becomes its own
row/column, so cross-partner terms fall out automatically, with no special
casing. The rotation-parameter count doesn't grow, though: each
representative pair's `degen_h` tiled copies are then **summed** back down
to a single `Biajb_dense`/`gn_flat` entry via a pooling matrix `P`
(`P[pair, tile] = 1` if that tile belongs to that representative pair):

```python
Biajb_dense = P @ Biajb_tiled @ P.T
gn_flat = P @ gn_tiled
```

This is exactly the "shared kappa, applied identically to every partner"
physics from §3 — summing the tiled entries *is* the chain rule, now
correctly including the cross-partner terms the sum brings in for free.
When `exploit_degen=False` (or for any nondegenerate irrep), tiling is a
no-op (`degen_h=1`, one tile per pair) and this reduces exactly to the
already-validated cross-irrep-only Hessian from §4 — confirmed by rerunning
every existing test unchanged.

### Why this *isn't* `MP2.run_degen_tensor()`'s tiling trick, despite looking similar

`MP2.run_degen_tensor()` also tiles degenerate irreps, but for a different
reason: MP2's `IJAB` energy expression is genuinely **quadratic and summed
over every partner combination independently** — each partner pair is a
*distinct, separately-nonzero* contribution to the correlation energy, so
tiling there creates new, independent output columns per partner (see
`docs/degen_tensor.md` §5). SOSCF's rotation parameter is different: it's
**one shared value**, not `degen_h` independent ones — tiling here is used
to correctly *evaluate the physics of the shared parameter*, then
**collapsed back down** via pooling. Both use the same "tile the
representative partner across all its copies" primitive, but MP2 keeps the
tiled axes as independent output dimensions, while SOSCF sums them away
again at the end.

## 7. Net effect: `exploit_degen=True` now costs nothing in iteration count

| Molecule | Before this session | After both fixes | C1 baseline |
|---|---|---|---|
| Methane/STO-3G, `exploit_degen=True` | 24 | **6** | 5 |
| Methane/cc-pVDZ, `exploit_degen=True` | crashed | **6** | 7 |
| Ammonia/STO-3G, `exploit_degen=True` | 24 | **6** | 6 |

Methane/cc-pVDZ is the hardest available case — `E`, `T1`, and `T2` are all
simultaneously populated and individually degenerate, so it exercises
cross-irrep coupling (§4) *and* cross-partner coupling (§6) at once, and
now converges in fewer iterations than even its own C1-forced baseline.

## 8. `exploit_degen=True` vs `False`, summarized

| | `exploit_degen=False` | `exploit_degen=True` |
|---|---|---|
| What's stored per degenerate irrep | All `degen_h` partners explicitly | One representative partner only |
| Total energy | Direct sum | `degen_h ×` representative contribution |
| Rotation parameters (`gn`/`x`/`U`) | One per same-irrep occ-virt pair | Same — never scales with `degen_h` |
| Gradient scaling needed | None (already explicit) | `degen_h ×` representative value (simple, exact) |
| Hessian scaling needed | None (already explicit) | `degen_h ×` same-partner term **+** `degen_h(degen_h−1) ×` a genuine cross-partner two-electron term (not a simple scalar) |
| Cost | `O(nbfxns)`-sized dense tensors at full AO count | Smaller per-irrep blocks, but the SOSCF Hessian build now tiles back up internally to get the physics right (see §6) — a real cost trade-off, though still cheap for the molecules tested here |

## 9. Where this is validated

- `test/test_soscf_hessian.py` — pytest-collected, fast: finite-difference
  gold-standard checks for both cross-irrep coupling (water/STO-3G,
  methane/cc-pVDZ) and cross-partner coupling (ammonia's `degen=2` `E`
  irrep, methane's `degen=3` `T2` irrep — the higher-degeneracy case caught
  a real bug in the *test's own* kappa construction, a good sign the check
  is doing its job), a diagonal-block regression against
  `DegenIntegralFactory`'s independently-derived oracle, and "did it
  actually engage" checks that off-diagonal/cross-partner terms aren't
  trivially zero.
- `test/smoke_so_rhf.py` — end-to-end Psi4 comparison plus iteration-count
  assertions (tight for both `exploit_degen` settings, calibrated against
  each molecule's own C1 baseline) across methane (Td, STO-3G and
  cc-pVDZ), water (C2v), and ammonia (C3v).
