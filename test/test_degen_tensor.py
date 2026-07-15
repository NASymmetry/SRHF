import numpy as np
import pytest

from srhf.degen_tensor import AxisMeta, DegenTensor, CONSUMED


def pending(degen, irrep=0):
    return AxisMeta(irrep=irrep, degen=degen, pending=True)


def consumed(degen=1, irrep=None):
    return AxisMeta(irrep=irrep, degen=degen, pending=False)


class _FakeIrrep:
    def __init__(self, d):
        self.d = d


class _FakeSymtext:
    """Minimal stand-in for molsym.Symtext, just enough to exercise
    DegenTensor.from_irreps's symtext.irreps[ir].d lookups."""
    def __init__(self, dims):
        self.irreps = [_FakeIrrep(d) for d in dims]


# ---------------------------------------------------------------------------
# Core scaling behavior
# ---------------------------------------------------------------------------

def test_contracted_pending_axis_applies_factor_once():
    np.random.seed(0)
    ERI = np.random.rand(4, 3)  # axes: p (bra), r (ket, degenerate)
    d = np.random.rand(3)
    t = DegenTensor(ERI, (consumed(), pending(degen=5)))

    result = DegenTensor.einsum("pr,r->p", t, d)

    expected = 5 * np.einsum("pr,r->p", ERI, d)
    assert np.allclose(result.array, expected)


def test_no_factor_when_axis_not_pending():
    np.random.seed(1)
    ERI = np.random.rand(4, 3)
    d = np.random.rand(3)
    t = DegenTensor(ERI, (consumed(), consumed()))

    result = DegenTensor.einsum("pr,r->p", t, d)

    expected = np.einsum("pr,r->p", ERI, d)
    assert np.allclose(result.array, expected)


def test_surviving_axis_carries_metadata_unchanged():
    np.random.seed(2)
    arr = np.random.rand(2, 3, 4)  # p (pending, degen=5), q (consumed), r (contracted)
    d = np.random.rand(4)
    t = DegenTensor(arr, (pending(degen=5), consumed(), consumed()))

    result = DegenTensor.einsum("pqr,r->pq", t, d)

    # r is contracted but not pending -> no factor. p survives -> no factor
    # charged yet either, but its pending/degen metadata must propagate.
    expected = np.einsum("pqr,r->pq", arr, d)
    assert np.allclose(result.array, expected)
    assert result.axes[0] == pending(degen=5)
    assert result.axes[1] == consumed()


def test_chained_contraction_applies_factor_exactly_once():
    np.random.seed(3)
    arr = np.random.rand(2, 3, 4)
    d1 = np.random.rand(4)
    t = DegenTensor(arr, (pending(degen=5), consumed(), consumed()))

    intermediate = DegenTensor.einsum("pqr,r->pq", t, d1)
    assert intermediate.axes[0].pending  # factor not yet charged

    d2 = np.random.rand(2)
    final = DegenTensor.einsum("pq,p->q", intermediate, d2)

    # Total: sum over r (no factor, not pending) then sum over p (pending,
    # degen=5) -> exactly one factor of 5 over the whole chain.
    unscaled = np.einsum("pq,p->q", np.einsum("pqr,r->pq", arr, d1), d2)
    assert np.allclose(final.array, 5 * unscaled)
    assert final.axes == (CONSUMED,)


def test_shared_pending_label_applies_factor_once():
    np.random.seed(4)
    a = np.random.rand(3)
    b = np.random.rand(3)
    A = DegenTensor(a, (pending(degen=3),))
    B = DegenTensor(b, (pending(degen=3),))

    result = DegenTensor.einsum("r,r->", A, B)

    # Physically this is one degenerate index being summed once -- must be
    # charged degen^1, not degen^2, even though BOTH operands tag it pending.
    expected = 3 * np.einsum("r,r->", a, b)
    assert np.isclose(result.array, expected)
    assert not np.isclose(result.array, 9 * np.einsum("r,r->", a, b))


def test_inconsistent_metadata_raises():
    a = np.random.rand(3)
    b = np.random.rand(3)
    A = DegenTensor(a, (pending(degen=3, irrep=0),))
    B = DegenTensor(b, (pending(degen=4, irrep=0),))

    with pytest.raises(ValueError):
        DegenTensor.einsum("r,r->", A, B)


def test_plain_ndarray_operand_untagged():
    np.random.seed(5)
    ERI = np.random.rand(4, 3)
    d = np.random.rand(3)  # plain ndarray, no metadata
    t = DegenTensor(ERI, (consumed(), pending(degen=2)))

    result = DegenTensor.einsum("pr,r->p", t, d)

    expected = 2 * np.einsum("pr,r->p", ERI, d)
    assert np.allclose(result.array, expected)


def test_both_operands_plain_ndarray_behaves_like_numpy():
    np.random.seed(6)
    a = np.random.rand(4, 3)
    b = np.random.rand(3)
    result = DegenTensor.einsum("pr,r->p", a, b)
    assert np.allclose(result.array, np.einsum("pr,r->p", a, b))


# ---------------------------------------------------------------------------
# from_irreps / pair_groups: axes coupled through one shared degenerate-
# partner index (e.g. an ERI block's bra pair or ket pair, which always
# share one irrep by group theory) must be charged once per GROUP, not once
# per axis -- this is the first of two related bugs found while validating
# DegenIntegralFactory against methane: this one affects the "at most one
# side degenerate" branch (DegenIntegralFactory._make_block's plain
# from_irreps path). A second, distinct bug in the SAME area (when BOTH bra
# and ket are degenerate, "select + multiply" is invalid at all, regardless
# of pair_groups) needed a different fix entirely -- see
# DegenIntegralFactory._make_block and test_degen_integral_factory.py.
# ---------------------------------------------------------------------------

def test_from_irreps_no_pair_groups_charges_every_axis_independently():
    symtext = _FakeSymtext([1, 3])  # irrep0 nondegenerate, irrep1 degen=3
    t = DegenTensor.from_irreps(np.random.rand(3, 3), (1, 1), symtext, exploit_degen=True)
    assert t.axes[0].pending and t.axes[0].degen == 3
    assert t.axes[1].pending and t.axes[1].degen == 3


def test_from_irreps_pair_group_charges_once_not_per_axis():
    np.random.seed(10)
    symtext = _FakeSymtext([1, 3])  # irrep1 has degen=3
    arr = np.random.rand(4, 3, 3)  # p (irrep0), r,s (coupled irrep1 pair)
    t = DegenTensor.from_irreps(
        arr, (0, 1, 1), symtext, exploit_degen=True, pair_groups=[(0,), (1, 2)]
    )
    d = np.random.rand(3, 3)
    result = DegenTensor.einsum("prs,rs->p", t, d)
    # exactly one factor of 3 for the coupled (r,s) pair, not 3*3=9
    expected = 3 * np.einsum("prs,rs->p", arr, d)
    assert np.allclose(result.array, expected)
    assert not np.allclose(result.array, 9 * np.einsum("prs,rs->p", arr, d))


def test_from_irreps_same_irrep_different_groups_charged_independently():
    # A bra pair (axes 0,1) and a ket pair (axes 2,3) that happen to share
    # the SAME irrep VALUE are still different physical couplings (different
    # pair_groups) and so must each be charged independently, not merged
    # into a single shared charge, if this from_irreps path were ever used
    # for such a block. In practice DegenIntegralFactory routes any block
    # with both bra and ket degenerate through a different construction
    # entirely (see _make_block) rather than through from_irreps/
    # pair_groups, but this still documents/guards the pair_groups
    # mechanism's own semantics in isolation.
    symtext = _FakeSymtext([3])  # single degenerate irrep, degen=3
    t = DegenTensor.from_irreps(
        np.zeros((2, 2, 2, 2)), (0, 0, 0, 0), symtext, exploit_degen=True,
        pair_groups=[(0, 1), (2, 3)],
    )
    assert t.axes[0].pending and not t.axes[1].pending
    assert t.axes[2].pending and not t.axes[3].pending


def test_from_irreps_partial_pair_groups_defaults_unlisted_axes_independent():
    symtext = _FakeSymtext([1, 4])  # irrep0 nondegenerate, irrep1 degen=4
    t = DegenTensor.from_irreps(
        np.zeros((2, 2, 2)), (1, 0, 1), symtext, exploit_degen=True,
        pair_groups=[(0, 2)],  # axis 1 deliberately not mentioned
    )
    assert t.axes[0].pending  # rank 0 of group (0,2)
    assert not t.axes[2].pending  # rank 1 of group (0,2)
    assert not t.axes[1].pending  # irrep0 is nondegenerate (d=1) regardless


# ---------------------------------------------------------------------------
# Parsing / validation edge cases
# ---------------------------------------------------------------------------

def test_ellipsis_subscripts_rejected():
    a = np.random.rand(2, 2)
    with pytest.raises(NotImplementedError):
        DegenTensor.einsum("...ij->...ji", a)


def test_multiletter_labels_rejected():
    a = np.random.rand(2, 2)
    with pytest.raises(NotImplementedError):
        DegenTensor.einsum("ab1,ab1->ab1", a, a, a)


def test_implicit_output_matches_numpy_convention():
    np.random.seed(7)
    a = np.random.rand(3, 4)
    b = np.random.rand(4, 5)
    result = DegenTensor.einsum("ij,jk", a, b)
    assert np.allclose(result.array, np.einsum("ij,jk", a, b))


def test_operand_count_mismatch_raises():
    a = np.random.rand(3)
    with pytest.raises(ValueError):
        DegenTensor.einsum("r,r->", a)


def test_spec_rank_mismatch_raises():
    t = DegenTensor(np.random.rand(3, 3), (consumed(), consumed()))
    with pytest.raises(ValueError):
        DegenTensor.einsum("r->r", t)


def test_axis_count_mismatch_raises_on_construction():
    with pytest.raises(ValueError):
        DegenTensor(np.random.rand(3, 3), (consumed(),))
