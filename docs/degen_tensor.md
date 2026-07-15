# Exploiting degenerate irreps: `DegenTensor` and `DegenIntegralFactory`

This document explains the symmetry SRHF exploits when a molecule's point
group has a genuinely degenerate irrep (e.g. `E` in C3v/ammonia, `T1`/`T2` in
Td/methane), how `srhf/degen_tensor.py` generalizes that exploit into a
reusable tensor abstraction, and how the Hartree-Fock Fock build
(`srhf/rhf.py`) and MP2 correlation energy (`srhf/mp2.py`) each use it
differently — including a real bug MP2 exposed that the Fock build's design
could not.

## 1. The symmetry: representative partners

A degenerate irrep of dimension `d_mu` (e.g. `d=2` for `E`, `d=3` for `T1`
or `T2`) doesn't correspond to one symmetry-adapted linear combination
(SALC) per irrep — it corresponds to `d_mu` *partner* SALCs that transform
into each other under the group's operations. `so_orbitals.C`, `.eps`, and
friends only ever store **one representative partner** per degenerate irrep
(`irreplength[h]` functions, not `d_mu * irreplength[h]`) — by symmetry, the
MO coefficients and orbital energies built from any one partner are
*identical* to those built from any other partner. That compression is
already in place throughout the codebase (`SOrbitals.C.blocks[h]` has shape
`(irreplength[h], irreplength[h])`; `ORB.ndocc_ir`/`nvirt_ir` are counted in
this compressed basis) — `degen_tensor.py` doesn't introduce it, it exists
to correctly *account for* it wherever a calculation needs to sum over all
`d_mu` partners rather than just look one up.

That gives every tensor axis tied to a degenerate irrep one of two roles:

- **Operator-diagonal role** — `F`, `H`, `D`, `C`, orbital energies: every
  partner gives the identical value, so nothing needs to be summed. No
  scaling factor, ever.
- **Genuinely summed-over role** — e.g. the ket index in
  `J_pq = sum_r (pr|qs) D_rs`: `r` ranges over all `d_mu` partners of its
  irrep, and although each partner contributes an identical value (because
  `D` is itself operator-diagonal), the sum still needs exactly `d_mu`
  copies of it. Since only the representative partner was ever computed,
  that sum degenerates to *"take the one value you have and multiply by
  `d_mu`, once."*

That one multiplication is the entire exploit. Getting it applied to
exactly the right axes, exactly once, for an arbitrary contraction pattern
is what `degen_tensor.py` is for.

## 2. Where this started: the ad hoc special case in `SRHF.jk()`

Before `degen_tensor.py` existed, this exploit was hand-written into one
function, [`srhf/rhf.py:427`](../srhf/rhf.py):

```python
def jk(self, ERI, ERI_swap, d, braket, block):
    if braket == 2:
        degen = self.symtext.irreps[block[3]].d
        j = degen * np.einsum('pr,r->p', ERI, d)
        k = degen * np.einsum('pr,r->p', ERI_swap, d)
    else:
        j = np.einsum('pr,r->p', ERI, d)
        k = np.einsum('pr,r->p', ERI_swap, d)
    return j, k
```

`braket` is a code (0–3) computed elsewhere in `DPD` describing whether the
bra pair and/or ket pair of that ERI block belong to a degenerate irrep.
`braket == 2` means "ket is degenerate, bra is not" — the only case `jk()`
actually special-cases — and the fix is exactly the multiplication described
above: the density-contracted ket index `r` was summed over one
representative partner, so multiply by `degen` once. This works, but it's
locked to this one hand-written contraction (`'pr,r->p'`) and this one
axis-role assignment; extending it to a different contraction pattern (MP2's
4-index transform, eventually coupled-cluster amplitudes) would mean writing
a new hand-special-cased function each time.

## 3. Generalizing it: `AxisMeta` and `DegenTensor.einsum`

`srhf/degen_tensor.py` replaces the `braket` integer code with one
`AxisMeta` dataclass **per tensor axis**:

```python
@dataclass(frozen=True)
class AxisMeta:
    irrep: Optional[int]   # which irrep this axis is compressed against
    degen: int             # d_mu, the dimension of that irrep
    pending: bool          # True: this axis still owes one factor of `degen`
                            # to whichever contraction next sums over it
```

A `DegenTensor` is just an `ndarray` plus one `AxisMeta` per axis. Its
`einsum` classmethod is a drop-in `np.einsum` replacement: for every label
that gets *contracted away* (absent from the output subscript), if any
operand's axis carries that label with `pending=True`, the whole result is
multiplied by that axis's `degen` — **once per unique label**, not once per
operand that happens to share it (a `pair_groups` mechanism, described
below, prevents a bra pair or ket pair from being double-charged). Labels
that *survive* into the output keep their metadata for the next contraction
down the line, so a factor is never dropped or double-applied across a
chain of contractions. Plain `np.ndarray` operands (F, H, D, C, S — the
operator-diagonal objects) are treated as fully consumed, `degen=1`
everywhere, exactly matching their role above.

This is the direct generalization of `jk()`'s `if braket == 2: degen * ...`:
[`srhf/rhf.py:311`](../srhf/rhf.py) (`build_fock_degen_tensor`) builds the
same J and K contractions with no `if` at all —

```python
def build_fock_degen_tensor(self, H, Dp, repacked_bigERI, repacked_bigERI_swapped):
    ...
    for b, block in enumerate(self.dpd.nonzero_blocks):
        f_sym, d_sym = block[0], block[3]
        D_block = Dp.blocks[d_sym]
        J = DegenTensor.einsum('pqrs,rs->pq', repacked_bigERI[b], D_block)
        K = DegenTensor.einsum('pqrs,rs->pq', repacked_bigERI_swapped[b], D_block)
        oned_f[f_sym] += 2 * J.array - K.array
```

— because `repacked_bigERI[b]` already carries the right `AxisMeta` on its
`r`/`s` axes (built by `DegenIntegralFactory`, next section), and `D_block`
is a plain array (degen=1, operator-diagonal). Whichever of `r`/`s` is
`pending` gets charged automatically; `p`/`q` survive into the output
untouched, matching `F`'s own operator-diagonal role. `get_vhf_degen_tensor`
([`srhf/rhf.py:392`](../srhf/rhf.py)) is the same pattern for the SAD-guess
Fock build.

## 4. Building the tagged blocks: `DegenIntegralFactory`

`DegenIntegralFactory` ([`srhf/degen_tensor.py:211`](../srhf/degen_tensor.py))
builds the compressed, per-block-tagged SO-basis ERI directly from the raw
AO ERI, via the same "one representative partner per irrep" SALC row
selection `SOrbitals.ao_to_so()` already uses — it deliberately doesn't
touch the older sparse/packed-index compression code in `DPD`.

**Which blocks are even nonzero?** A two-electron integral `(pq|rs)` is
nonzero only if `Γp⊗Γq⊗Γr⊗Γs` contains the totally symmetric irrep. For real
irreps, the character-orthogonality theorem collapses `Γp⊗Γq` containing the
totally symmetric irrep down to the simple condition `p == q` (and likewise
`r == s` for the ket). `DPD._find_nonzero_blocks()`
([`srhf/srhf_helper.py:983`](../srhf/srhf_helper.py)) uses exactly that
per-pair test (`dp_contains_tsir(i, j)` and `dp_contains_tsir(k, l)`,
independently) to enumerate nonzero `(i, i, k, k)` blocks — i.e. the
Cartesian product of populated irreps, bra irrep crossed with ket irrep.
**This is a narrower condition than the true selection rule** (`Γp⊗Γq`
containing *any* irrep that `Γr⊗Γs` also contains would suffice — it doesn't
have to be specifically the totally symmetric one, and it doesn't force
`p==q`/`r==s` on its own). It happens to be exactly the set of blocks the
*Fock build* needs, though, because `D` and `F` are themselves
operator-diagonal (block-diagonal by irrep): any ERI contribution outside
`i==j`/`k==l` would get multiplied by `D_rs == 0` regardless of whether the
raw integral is "really" nonzero, so nothing is lost by skipping those
blocks for the Fock build specifically. (Section 6 below covers why this
same narrowing is *not* safe for MP2, which has no such shield.)

**The subtlety that isn't just "same irrep on both sides."**
`DegenIntegralFactory._make_block` ([`srhf/degen_tensor.py:308`](../srhf/degen_tensor.py))
picks between two strategies per block:

```python
bra_is_degen = bool(exploit and bra_degen > 1)
ket_is_degen = bool(exploit and ket_degen > 1)

if bra_is_degen and ket_is_degen:
    # full/uncompressed fallback -- see below
    ...
# else: select one representative partner per pair and tag it pending
idx = [self._axis_range(ir) for ir in block]
sub = E[np.ix_(*idx)]
return DegenTensor.from_irreps(sub, block, self.symtext, exploit,
                                pair_groups=[(0, 1), (2, 3)])
```

"Select one representative partner per pair, tag it pending, and let
`DegenTensor.einsum` multiply by `degen` once" is only valid when *at most
one side* of the block has genuine partner choice — e.g. bra nondegenerate,
ket degenerate: the ket's partner-dependence is guaranteed uniform (by the
bra being a fixed, single "partner"), so summing `degen` identical copies is
exactly `1 x degen`. The moment **both** bra and ket are degenerate irreps
(same irrep on both sides, e.g. methane's `T2 x T2`, or *different*
degenerate irreps, e.g. `E x T2`), that uniformity breaks: which ket partner
you'd sum against genuinely depends on which bra partner was selected, so
"pick partner 0 on both sides and multiply by a single `degen`" is
mathematically wrong. This was caught by brute-force comparison specifically
on methane/cc-pVDZ (whose `E`, `T1`, and `T2` irreps are all simultaneously
populated, producing cross-irrep degenerate-both blocks) — STO-3G methane
and ammonia each only populate one degenerate irrep, so they can't expose
this case at all. The fix is a full, uncompressed transform restricted to
just the two irreps involved (still cheap — not the whole molecule), with
the ket partner sum performed explicitly:

```python
E_full = self._full_transform(ERI_ao)          # uncompressed, this block's irreps only
...
sub = None
for k in range(ket_degen):
    ket_idx = range(full_offsets[ir3] + k * il_ket, full_offsets[ir3] + (k + 1) * il_ket)
    piece = E_full[np.ix_(bra_idx, bra_idx, ket_idx, ket_idx)]
    sub = piece if sub is None else sub + piece
```

The resulting block's ket axes are marked `pending=False` (already fully
summed); only the bra axis is left `pending=True`, matching `DPD`'s own
`lookup_degen()` convention for this same `braket == 3` case (bra AND ket
degenerate, checked independently — not "same irrep").

## 5. MP2: the same exploit hits a wall that HF never sees

`srhf/mp2.py`'s `run_symm()` computes the correlation energy from one
*combined* tensor spanning every irrep at once — `occ_C`/`virt_C` are
`block_diag`'d together across all irreps first, then a single
`(mnrs,mI,nA,rJ,sB->IAJB)` contraction and the standard
`E_2 = sum(IJAB * (2*IJAB - swap(IJAB)) / denom)` formula run once over the
whole thing. This assumes `so_orbitals.C` spans the full AO count per
irrep — true when `exploit_degen=False`, false the moment `C` is compressed
to `irreplength[h]` for a real degenerate irrep, which raises a
shape-mismatch error.

The first attempt to fix this reused the Fock build's own machinery:
decompose the MP2 energy into per-`(mu, nu)`-irrep-pair blocks via
`DPD.nonzero_blocks`/`DegenIntegralFactory`, exactly like
`build_fock_degen_tensor` does. **This was wrong, not just buggy** — and the
reason is precisely the caveat flagged in Section 4: `nonzero_blocks`'
`i==j`/`k==l` narrowing is only lossless for the Fock build because `D`/`F`
shield the dropped cross-irrep contributions with a zero. MP2 has no such
shield — it consumes the raw `(ia|jb)` integrals directly, and cross-irrep
bra pairs (`Γi != Γa`) are frequently *not* zero. This was confirmed by
directly inspecting `run_symm()`'s own already-correct combined `IJAB`
tensor: significant nonzero values showed up for cross-irrep `(I, A)` bra
pairs even for **water**, which has no degenerate irreps at all — proof the
gap has nothing to do with degeneracy specifically, only with reusing a
selection rule that was only ever valid for the Fock build's context.

### The fix: tile the compressed basis instead of decomposing into blocks

`run_degen_tensor()` ([`srhf/mp2.py:112`](../srhf/mp2.py)) never decomposes
into blocks at all. It reuses `run_symm()`'s combined-tensor formula
completely unchanged, and instead fixes the one thing that actually
breaks — the *size* of the combined `occ_C`/`virt_C`/orbital-energy arrays —
by tiling each irrep's compressed, `irreplength[h]`-sized block `degen`
times (once per partner) before combining:

```python
for h in range(len(so.symtext.irreps)):
    blk_o, blk_v = occ_C.blocks[h], virt_C.blocks[h]
    ...
    degen = so.symtext.irreps[h].d if self.options.exploit_degen else 1
    ndocc_h = so.Orbs[h].ndocc_ir
    for _ in range(degen):
        occ_tiled.append(blk_o)
        virt_tiled.append(blk_v)
        eocc_tiled.append(so.eps[h][:ndocc_h])
        evirt_tiled.append(so.eps[h][ndocc_h:])

occ_C_full = block_diag(*occ_tiled)
virt_C_full = block_diag(*virt_tiled)
```

This is valid for exactly the reason Section 1 gives: every partner of a
degenerate irrep has *identical* MO coefficients and orbital energies by
symmetry, so duplicating the representative partner's block `degen` times
along the diagonal reconstructs a full, `nbfxns`-sized coefficient matrix
that is dimensionally consistent with the *uncompressed* dense ERI
(`self.ERI`, already computed by the SCF step — no re-transform needed).
Because `run_symm()`'s formula makes no `nonzero_blocks`-style assumption at
all, it now automatically captures every genuinely nonzero `(ia|jb)`
contribution — cross-irrep or not — with no selection-rule reasoning
required. The tiling trick itself isn't new (it's the same "duplicate the
representative partner across all its partners" trick the Section 4
both-degenerate fallback also relies on); what changed is *where* it's
applied — to the input coefficients, not to a set of pre-filtered blocks.

### A second, independent hazard: `DegenTensor` is unsafe for a quadratic formula

While debugging the above, a second, purely mechanical problem was found and
is kept as a standalone regression test
(`test/test_mp2_degen_tensor.py::test_degen_tensor_einsum_then_square_gives_wrong_exponent`):
`DegenTensor.einsum` is designed for *linear* consumption — a `pending`
axis's `degen` factor is charged once into the array, correct if that array
is used exactly once afterward (e.g. added straight into a Fock matrix).
MP2's energy expression is *quadratic* in the same integral
(`IJAB * (2*IJAB - swap(IJAB))`): if `IJAB` were built via
`DegenTensor.einsum` (charging `degen` once into the array) and then
squared, the result would carry `degen**2`, not the physically correct
`degen**1` — squaring `degen` constant-valued partner contributions gives
`sum_u X_u**2 == degen * X**2`, not `(degen * X)**2`. This is exactly why
`run_degen_tensor()` never uses `DegenTensor`/`DegenIntegralFactory` at
all — `self.ERI` is a plain, already-uncompressed `ndarray`, so no
degeneracy factor is ever baked into it in the first place, and none needs
to be un-baked afterward.

## 6. Options

- `options.exploit_degen` — whether to compress degenerate irreps down to
  one representative partner at all (the SALC-selection step every path
  above depends on). `False` makes every irrep behave as `degen=1`
  everywhere in this document.
- `options.degen_tensor` — SCF-path selection: use
  `DegenIntegralFactory`/`build_fock_degen_tensor`/`get_vhf_degen_tensor`
  for the Fock build instead of the legacy `sparse_transform`/`jk()` path.
  Independent of `exploit_degen` (it can be `True` with `exploit_degen=False`,
  in which case every block behaves as the "at most one side degenerate"
  case with `degen=1`, a no-op multiplication).
  `MP2.run_degen_tensor()` does not depend on `options.degen_tensor` at
  all — it's a self-contained fix within `mp2.py` and works regardless of
  which Fock-build path was used for the underlying SCF.

## 7. Where this is validated

- `test/test_degen_tensor.py` — pure-math unit tests of `DegenTensor.einsum`
  and `from_irreps`/`pair_groups` (no chemistry).
- `test/test_degen_integral_factory.py` — block-level correctness of
  `DegenIntegralFactory` against a from-scratch brute-force reference, using
  NH3 and methane/cc-pVDZ (the case that exposed the both-degenerate
  subtlety in Section 4).
- `test/smoke_degen_tensor.py` — end-to-end SCF energies vs Psi4: water,
  methane (STO-3G and cc-pVDZ), ammonia, across core/gwh/sad guesses.
- `test/test_mp2_degen_tensor.py` — the `DegenTensor`-quadratic-reuse
  regression test (Section 5), plus end-to-end MP2 correlation energies vs
  Psi4 for water, ammonia/STO-3G, and methane/cc-pVDZ, each checked under
  both `exploit_degen=True` and `exploit_degen=False`.
- `test/smoke_mp2_degen_tensor.py` — the same end-to-end MP2 check as a
  standalone script, plus methane/STO-3G for cheap single-degenerate-irrep
  coverage.

Across all of the above, molecules with **no** degenerate irreps (e.g.
water) cannot exercise any of this code meaningfully — every degeneracy
factor collapses to `1`. Regression coverage for this subsystem specifically
requires ammonia (C3v, `E`) or methane (Td, `T1`/`T2`), and — for the
cross-irrep cases in Sections 4 and 5 — specifically a basis large enough to
populate more than one degenerate irrep at once (methane/cc-pVDZ, not
STO-3G).
