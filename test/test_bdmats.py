import numpy as np
import pytest
from srhf.bdmats import BDMatrix

# ---------------------------------------------------------------------------
# BASIC ARITHMETIC
# ---------------------------------------------------------------------------

def test_addition_subtraction(bd_eye_2x1):
    A = bd_eye_2x1
    B = bd_eye_2x1
    C = A + B
    D = C - A
    for b in C.blocks:
        assert np.allclose(b, 2 * np.eye(b.shape[0]))
    for b in D.blocks:
        assert np.allclose(b, np.eye(b.shape[0]))

def test_scalar_multiplication(bd_eye_2x1):
    A = 3 * bd_eye_2x1
    for b in A.blocks:
        assert np.allclose(b, 3 * np.eye(b.shape[0]))

# ---------------------------------------------------------------------------
# LINEAR ALGEBRA: @, dot, T, inv, pow, eigh
# ---------------------------------------------------------------------------

def test_dot_blockwise_multiplication(bd_scalar_blocks):
    """Ensure BDMatrix.dot performs blockwise matrix multiplication correctly."""
    A, B = bd_scalar_blocks, bd_scalar_blocks
    C = A.dot(B)
    # Blocks: [[2.]] * [[2.]] = [[4.]], [[3.]] * [[3.]] = [[9.]]
    assert np.allclose(C.blocks[0], np.array([[4.0]]))
    assert np.allclose(C.blocks[1], np.array([[9.0]]))

def test_transpose_and_inverse(bd_eye_2x1):
    A = bd_eye_2x1
    T = A.T()
    for i, block in enumerate(A.blocks):
        assert np.allclose(T.blocks[i], block.T)
    invA = A.inv()
    for i, block in enumerate(A.blocks):
        if block.size:
            prod = block @ invA.blocks[i]
            assert np.allclose(prod, np.eye(block.shape[0]))

def test_pow_and_eigh(bd_random_3x2):
    A = bd_random_3x2
    A_half = A ** 0.5
    for b in A_half.blocks:
        if b.size:
            assert np.all(np.isfinite(b))
    eigvals, eigvecs = A.eigh()
    for vals, vecs in zip(eigvals, eigvecs.blocks):
        if vals.size:
            assert np.allclose(vecs.T @ vecs, np.eye(vecs.shape[0]), atol=1e-10)

# ---------------------------------------------------------------------------
# STRUCTURAL AND UTILITY METHODS
# ---------------------------------------------------------------------------

def test_full_mat_and_back(bd_eye_2x1):
    full = bd_eye_2x1.full_mat()
    assert full.shape == (3, 3)
    A2 = BDMatrix.full_to_bd(full, [2, 1])
    assert np.allclose(A2.full_mat(), full)

def test_reshape_and_swapaxes():
    arr1 = np.arange(4).reshape(2, 2)
    arr2 = np.array([[5]])
    A = BDMatrix([arr1, arr2])
    B = A.reshape((4, 1), (1, 1))
    assert B.blocks[0].shape == (4, 1)
    C = A.swapaxes(0, 1)
    assert np.allclose(C.blocks[0], arr1.T)

def test_frob_norm(bd_random_3x2):
    A = bd_random_3x2
    n = A.frob_norm()
    expected = sum(np.linalg.norm(b, 'fro') for b in A.blocks if b.size)
    assert np.isclose(n, expected)

# ---------------------------------------------------------------------------
# EDGE CASES AND ERROR HANDLING
# ---------------------------------------------------------------------------

def test_empty_blocks_handled(bd_mixed_shapes):
    B = bd_mixed_shapes.inv()
    assert isinstance(B, BDMatrix)
    assert B.blocks[0].size == 0
    assert np.allclose(B.blocks[1], np.eye(2))

def test_size_mismatch_raises():
    A = BDMatrix([np.eye(2)])
    B = BDMatrix([np.eye(3)])
    with pytest.raises(ValueError):
        _ = A.dot(B)

def test_string_and_repr(bd_eye_2x1):
    s = str(bd_eye_2x1)
    r = repr(bd_eye_2x1)
    assert "array" in s
    assert "array" in r

