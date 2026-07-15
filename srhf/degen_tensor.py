"""
Degeneracy-aware tensor contraction subsystem.

Generalizes the ad hoc "braket == 2: multiply by degen" special case in
SRHF.jk() (srhf/rhf.py) into a tensor abstraction that tracks, per axis,
whether that axis is a compressed representative of a degenerate irrep and
whether its degenerate-partner sum has already been performed. A generic
einsum-style contraction can then apply the right scaling factor for any
contraction pattern, not just the one hand-written J/K pattern -- intended
to also serve MP2's 4-index IJAB transform and, eventually, coupled-cluster
amplitude tensors.

Confirmed rule: for a tensor axis compressed to one representative of a
d_mu-fold degenerate set,
  - an "operator-diagonal" role (F/H/D blocks -- all d_mu partners give
    identical values by symmetry) never needs a scaling factor.
  - a genuinely summed-over role (e.g. the ket index in J_pq = sum_r (p|r)
    D_r) needs exactly one factor of d_mu applied once, either baked in
    during construction or charged once at contraction time.

DegenTensor.einsum charges that factor automatically, once per unique
contracted label, from each operand's own per-axis metadata.
"""
from dataclasses import dataclass
from typing import Optional, Tuple
from collections import Counter

import numpy as np


@dataclass(frozen=True)
class AxisMeta:
    """
    Per-axis degeneracy bookkeeping for one axis of a DegenTensor.

    irrep:   index into symtext.irreps this axis is compressed against,
             or None if this axis carries no degeneracy information.
    degen:   d_mu, the dimension of that irrep (1 if untracked/nondegenerate).
    pending: True  -> this axis still owes exactly one factor of `degen`
                       to whichever contraction next sums over it.
             False -> already complete: either the axis plays an
                       operator-diagonal role and will never need a factor,
                       or a factor was already baked in during construction.
    """
    irrep: Optional[int]
    degen: int
    pending: bool


CONSUMED = AxisMeta(irrep=None, degen=1, pending=False)


class DegenTensor:
    """
    One dense ndarray (one irrep-block, e.g. one nonzero (ir1,ir2,ir3,ir4)
    ERI quadruple) plus one AxisMeta per array axis, in axis order.

    Deliberately not a subclass of, or replacement for, BDMatrix: BDMatrix's
    contract (a bare list of blocks, one per single irrep, 2-index-only
    einsum, no metadata) is a different, already-heavily-used abstraction
    for physically block-diagonal operators (F, H, D, S, C). DegenTensor is
    for higher-rank, irrep-quadruple-indexed (or later, higher-rank) tensors
    that need per-axis degeneracy metadata BDMatrix has no slot for.
    """

    def __init__(self, array, axes: Tuple[AxisMeta, ...]):
        array = np.asarray(array)
        if array.ndim != len(axes):
            raise ValueError(
                f"DegenTensor: array has {array.ndim} axes but {len(axes)} "
                f"AxisMeta entries were given"
            )
        self.array = array
        self.axes = tuple(axes)

    @property
    def shape(self):
        return self.array.shape

    @classmethod
    def from_irreps(cls, array, irrep_ids, symtext, exploit_degen, pair_groups=None):
        """
        Build axis metadata straight from a tuple of irrep indices, one per
        axis of `array`, in order. Construction only ever selects a
        representative partner, never sums.

        pair_groups: axis positions that are physically coupled through one
        shared degenerate-partner index (e.g. a bra pair or a ket pair of a
        4-index ERI block, which always share one irrep by group theory --
        the direct product of two real irreps contains the totally
        symmetric irrep iff the irreps are equal). All axes in such a group
        were compressed together by selecting the SAME partner index, so
        they owe exactly one shared factor of `degen`, not one per axis:
        only the first-listed member of each group is marked pending; the
        rest of that group is marked pending=False (already spoken for).
        Axes not mentioned in any group are treated as singleton groups
        (ordinary independent axes). Defaults to every axis independent,
        matching a tensor with no coupled axes.
        """
        n = len(irrep_ids)
        groups = list(pair_groups) if pair_groups is not None else []
        covered = {pos for group in groups for pos in group}
        for i in range(n):
            if i not in covered:
                groups.append((i,))  # unlisted axes are independent singletons
        rank_within_group = {}
        for group in groups:
            for rank, pos in enumerate(group):
                rank_within_group[pos] = rank

        axes = []
        for i, ir in enumerate(irrep_ids):
            degen = symtext.irreps[ir].d
            is_degen = bool(exploit_degen and degen > 1)
            pending = is_degen and rank_within_group[i] == 0
            axes.append(AxisMeta(irrep=ir, degen=degen, pending=pending))
        return cls(array, tuple(axes))

    @staticmethod
    def _parse(subscripts):
        """
        Single-character-label einsum strings only, matching every einsum
        call already in this codebase (e.g. 'pr,r->p', 'PQRS,Pp,Qq,Rr,Ss').
        '...' ellipsis and multi-letter labels are not supported.
        """
        if "..." in subscripts:
            raise NotImplementedError(
                "DegenTensor.einsum: ellipsis ('...') subscripts are not supported"
            )
        if "->" in subscripts:
            left, out = subscripts.split("->")
        else:
            left = subscripts
            counts = Counter(left.replace(",", ""))
            out = "".join(sorted(l for l, n in counts.items() if n == 1))
        in_spec = [s.strip() for s in left.split(",")]
        for spec in in_spec:
            if not spec.isalpha():
                raise NotImplementedError(
                    "DegenTensor.einsum: only plain alphabetic single-character "
                    f"labels are supported, got spec {spec!r}"
                )
        return in_spec, out

    @staticmethod
    def einsum(subscripts, *operands):
        """
        numpy.einsum-alike. Operands may be DegenTensor or plain np.ndarray
        (plain ndarrays are treated as fully "consumed"/untracked, i.e. every
        axis behaves as degen=1 -- this is how D, H, C, S, F stay untouched
        BDMatrix blocks and never need to be wrapped).

        For every contracted label (absent from the output spec): if any
        operand's axis for that label is tagged pending, multiply the WHOLE
        result by that label's degen exactly once -- once per unique label,
        never once per operand sharing the label (double-counting guard).

        For every surviving label (present in the output spec): no factor is
        applied here; that axis's metadata is carried forward unchanged onto
        the output DegenTensor, so a later contraction can still apply the
        factor exactly once when it is finally summed over.
        """
        in_spec, out_spec = DegenTensor._parse(subscripts)
        if len(in_spec) != len(operands):
            raise ValueError(
                f"einsum subscripts describe {len(in_spec)} operands but "
                f"{len(operands)} were given"
            )
        raw = [op.array if isinstance(op, DegenTensor) else op for op in operands]
        result = np.einsum(subscripts, *raw, optimize="optimal")

        label_meta = {}
        for spec, op in zip(in_spec, operands):
            if not isinstance(op, DegenTensor):
                continue
            if len(spec) != op.array.ndim:
                raise ValueError(
                    f"einsum spec {spec!r} does not match operand with "
                    f"{op.array.ndim} axes"
                )
            for label, meta in zip(spec, op.axes):
                if label in label_meta:
                    prev = label_meta[label]
                    if (
                        prev.irrep is not None
                        and meta.irrep is not None
                        and (prev.irrep, prev.degen) != (meta.irrep, meta.degen)
                    ):
                        raise ValueError(
                            f"Inconsistent degeneracy metadata for shared "
                            f"einsum label {label!r}: {prev} vs {meta}"
                        )
                    if meta.pending and not prev.pending:
                        label_meta[label] = meta
                else:
                    label_meta[label] = meta

        factor = 1
        for label in set(label_meta) - set(out_spec):
            meta = label_meta[label]
            if meta.pending:
                factor *= meta.degen

        if factor != 1:
            result = result * factor

        out_axes = tuple(label_meta.get(label, CONSUMED) for label in out_spec)
        return DegenTensor(result, out_axes)


class DegenIntegralFactory:
    """
    Builds degeneracy-compressed, per-axis-tagged SO-basis ERI blocks via an
    independent, from-first-principles route: apply the same
    representative-partner row-selection SOrbitals.ao_to_so() already uses
    for one-electron integrals (salc[:irreplength[h]]) to build one
    compressed AO->SO coefficient matrix, run the standard 4-einsum
    aotoso_2-style transform chain with it once, and slice out each nonzero
    (ir1,ir2,ir3,ir4) block. Deliberately does not touch DPD's packed-index
    sparse/legacy compression code.
    """

    def __init__(self, salcs, symtext, so_orbitals, options):
        self.salcs = salcs
        self.symtext = symtext
        self.irreplength = so_orbitals.irreplength
        self.options = options
        if options.exploit_degen:
            for h, salc in enumerate(salcs.salc_sets):
                degen = symtext.irreps[h].d
                if salc.shape[0] % degen != 0:
                    raise ValueError(
                        f"DegenIntegralFactory: irrep {h} has {salc.shape[0]} "
                        f"SALC rows, not evenly divisible by its degeneracy "
                        f"{degen}"
                    )
        self._offsets = self._compute_offsets(self.irreplength)
        self._s_compressed = self._compressed_salc_matrix()
        self._E_cache_key = None
        self._E_cache_val = None
        self._E_full_cache_key = None
        self._E_full_cache_val = None
        self._full_offsets_cache = None

    def _compute_offsets(self, irreplength):
        offsets, o = [], 0
        for il in irreplength:
            offsets.append(o)
            o += il
        return offsets

    def _compressed_salc_matrix(self):
        parts = []
        for h, salc in enumerate(self.salcs.salc_sets):
            il = self.irreplength[h]
            if il == 0:
                continue
            parts.append(salc[:il].T)  # one representative partner per irrep,
            # same convention as SOrbitals.ao_to_so()
        return np.concatenate(parts, axis=1)

    def _full_offsets(self):
        # offsets into the FULL (uncompressed, all-partner) SO basis, i.e.
        # cumulative sums of raw salc_sets row counts, not irreplength
        if self._full_offsets_cache is None:
            offsets, o = [], 0
            for salc in self.salcs.salc_sets:
                offsets.append(o)
                o += salc.shape[0]
            self._full_offsets_cache = offsets
        return self._full_offsets_cache

    def _full_transform(self, ERI_ao):
        # Full (uncompressed, all-partner) AO->SO ERI transform, mirroring
        # SRHF.aotoso_2 exactly. Only computed lazily, when a block with
        # both bra and ket degenerate needs it (see _make_block) -- avoids
        # paying its O(nbfxns^4) cost unless the molecule actually has such
        # a block (same irrep on both sides, e.g. T2 x T2, or two different
        # degenerate irreps, e.g. E x T2 -- both need it equally).
        if self._E_full_cache_key is ERI_ao:
            return self._E_full_cache_val
        parts = [salc.T for salc in self.salcs.salc_sets if salc.shape[0] != 0]
        s_full = np.concatenate(parts, axis=1)
        t1 = np.einsum("PQRS,Pp->pQRS", ERI_ao, s_full, optimize="optimal")
        t2 = np.einsum("pQRS,Qq->pqRS", t1, s_full, optimize="optimal")
        t3 = np.einsum("pqRS,Rr->pqrS", t2, s_full, optimize="optimal")
        E = np.einsum("pqrS,Ss->pqrs", t3, s_full, optimize="optimal")
        self._E_full_cache_key = ERI_ao
        self._E_full_cache_val = E
        return E

    def _transform(self, ERI_ao):
        if self._E_cache_key is ERI_ao:
            return self._E_cache_val
        s = self._s_compressed
        t1 = np.einsum("PQRS,Pp->pQRS", ERI_ao, s, optimize="optimal")
        t2 = np.einsum("pQRS,Qq->pqRS", t1, s, optimize="optimal")
        t3 = np.einsum("pqRS,Rr->pqrS", t2, s, optimize="optimal")
        E = np.einsum("pqrS,Ss->pqrs", t3, s, optimize="optimal")
        self._E_cache_key = ERI_ao
        self._E_cache_val = E
        return E

    def _axis_range(self, ir):
        o = self._offsets[ir]
        return range(o, o + self.irreplength[ir])

    def _make_block(self, E, block, ERI_ao, swap):
        ir1, ir2, ir3, ir4 = block  # ir1==ir2 (bra) and ir3==ir4 (ket), by group theory
        exploit = self.options.exploit_degen
        bra_degen = self.symtext.irreps[ir1].d
        ket_degen = self.symtext.irreps[ir3].d
        bra_is_degen = bool(exploit and bra_degen > 1)
        ket_is_degen = bool(exploit and ket_degen > 1)

        if bra_is_degen and ket_is_degen:
            # Both bra AND ket are degenerate irreps -- this is the
            # condition that actually matters, NOT whether they're the
            # SAME irrep (an earlier version of this code incorrectly
            # special-cased only ir1==ir3, e.g. T2 x T2; that missed cases
            # like E x T2, verified wrong by brute force against cc-pVDZ
            # methane, which has genuine cross-irrep blocks -- STO-3G
            # methane and ammonia only ever populate one degenerate irrep
            # each, so they couldn't expose this gap). This matches
            # DPD.lookup_degen()'s braket==3 condition (bra_degen>1 and
            # ket_degen>1, computed independently, with no requirement that
            # the two irreps match).
            #
            # "Select one representative axis and multiply by degen" is
            # only valid when AT MOST ONE side has genuine partner choice:
            # when bra is nondegenerate (a single trivial "partner"), the
            # ket-partner-dependence of the integral is guaranteed uniform
            # across ket partners, so summing degen identical copies equals
            # one copy times degen. The moment bra ALSO has real partner
            # choice, that uniformity breaks down (bra fixed at partner 0
            # does not integrate the same against every ket partner) --
            # verified by brute force to fail for both same-irrep and
            # cross-irrep degenerate-both cases alike.
            #
            # Fall back to the full/uncompressed transform (same recipe as
            # SRHF.aotoso_2, restricted to these two irreps' own AO
            # functions -- cheap, not the whole basis) and sum the ket over
            # its degenerate partners explicitly, bra fixed at partner 0 --
            # matching what the legacy compress_neri does for this same
            # (braket==3) condition. The ket sum is baked in here, so both
            # ket axes are fully consumed.
            E_full = self._full_transform(ERI_ao)
            if swap:
                E_full = np.swapaxes(E_full, 1, 2)
            full_offsets = self._full_offsets()
            il_bra = self.irreplength[ir1]
            il_ket = self.irreplength[ir3]
            bra_idx = range(full_offsets[ir1], full_offsets[ir1] + il_bra)
            sub = None
            for k in range(ket_degen):
                ket_idx = range(
                    full_offsets[ir3] + k * il_ket, full_offsets[ir3] + (k + 1) * il_ket
                )
                piece = E_full[np.ix_(bra_idx, bra_idx, ket_idx, ket_idx)]
                sub = piece if sub is None else sub + piece
            axes = (
                AxisMeta(irrep=ir1, degen=bra_degen, pending=True),
                AxisMeta(irrep=ir2, degen=bra_degen, pending=False),
                AxisMeta(irrep=ir3, degen=ket_degen, pending=False),
                AxisMeta(irrep=ir4, degen=ket_degen, pending=False),
            )
            return DegenTensor(sub, axes)

        # At most one of bra/ket is degenerate: selecting one
        # representative partner per pair (axes (0,1) for the bra, (2,3)
        # for the ket) and charging the pair's degen once at contraction
        # time is valid -- verified against brute-force sums.
        idx = [self._axis_range(ir) for ir in block]
        sub = E[np.ix_(*idx)]
        return DegenTensor.from_irreps(
            sub, block, self.symtext, exploit, pair_groups=[(0, 1), (2, 3)],
        )

    def degen_ERI_transform(self, ERI_ao, nonzero_blocks, swap=False):
        E = self._transform(ERI_ao)
        if swap:
            E = np.swapaxes(E, 1, 2)  # (pq|rs) -> (pr|qs), same convention
            # rhf.py's legacy (sparse_transform=False) path already uses to
            # build the K-contraction set
        return [self._make_block(E, block, ERI_ao, swap) for block in nonzero_blocks]
