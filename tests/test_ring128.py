import numpy as np
import scipy.sparse as sp

from anchorflow.ring128 import R, SparseOp, rand

MOD = 1 << 128


def test_add_sub_neg_shift():
    rng = np.random.default_rng(1)
    a, b = rand(rng, 200), rand(rng, 200)
    ia, ib = a.to_ints(), b.to_ints()
    assert (a + b).to_ints() == [(x + y) % MOD for x, y in zip(ia, ib)]
    assert (a - b).to_ints() == [(x - y) % MOD for x, y in zip(ia, ib)]
    assert (-a).to_ints() == [(-x) % MOD for x in ia]
    assert a.shr(20).to_ints() == [x >> 20 for x in ia]
    assert a.add_const(-5).to_ints() == [(x - 5) % MOD for x in ia]


def test_bounded_masks_and_int64_roundtrip():
    rng = np.random.default_rng(2)
    r = rand(rng, 500, 118)
    assert all(x < (1 << 118) for x in r.to_ints())
    assert r.below_pow2(118)
    v = np.array([-3, 0, 7, -(1 << 40)], dtype=np.int64)
    assert np.array_equal(R.from_int64(v).to_int64(), v)


def test_sparse_product_matches_python():
    rng = np.random.default_rng(3)
    n = 60
    rows = rng.integers(0, n, 400)
    cols = rng.integers(0, n, 400)
    w = rng.integers(0, 1 << 20, 400).astype(np.uint64)
    M = sp.csr_matrix((w, (rows, cols)), shape=(n, n), dtype=np.uint64)
    x = rand(rng, n)
    got = (SparseOp(M) @ x).to_ints()
    ix = x.to_ints()
    want = [0] * n
    coo = M.tocoo()
    for r, c, d in zip(coo.row, coo.col, coo.data):
        want[r] = (want[r] + int(d) * ix[c]) % MOD
    assert got == want
