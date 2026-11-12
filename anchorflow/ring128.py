import numpy as np

U64 = np.uint64
LIMBS = 4
LIMB_BITS = 32
LMASK = U64((1 << 32) - 1)
ELEM_BYTES = 16
W_SPLIT = 11


class R:
    __slots__ = ("a",)

    def __init__(self, a):
        self.a = a

    def __len__(self):
        return self.a.shape[1]

    @staticmethod
    def zeros(m):
        return R(np.zeros((LIMBS, m), dtype=U64))

    @staticmethod
    def from_int64(v):
        v = np.asarray(v, dtype=np.int64)
        lo = v.view(U64)
        sign = np.where(v < 0, LMASK, U64(0))
        a = np.stack([lo & LMASK, lo >> U64(32), sign, sign.copy()])
        return R(a)

    @staticmethod
    def from_ints(values):
        values = [int(x) % (1 << 128) for x in values]
        a = np.array([[(x >> (32 * i)) & 0xFFFFFFFF for x in values] for i in range(LIMBS)], dtype=U64)
        return R(a.reshape(LIMBS, len(values)))

    def to_ints(self):
        a = self.a.astype(object)
        return [int(a[0, j]) | int(a[1, j]) << 32 | int(a[2, j]) << 64 | int(a[3, j]) << 96
                for j in range(self.a.shape[1])]

    def to_int64(self):
        return (self.a[0] | (self.a[1] << U64(32))).view(np.int64)

    def is_small_signed(self, bits=62):
        v = self.to_int64()
        hi_ok = np.where(v < 0, (self.a[2] == LMASK) & (self.a[3] == LMASK),
                         (self.a[2] == 0) & (self.a[3] == 0))
        return bool(np.all(hi_ok & (np.abs(v) < (1 << bits))))

    def copy(self):
        return R(self.a.copy())

    def __getitem__(self, idx):
        return R(self.a[:, idx])

    def __add__(self, o):
        return R(_norm(self.a + o.a))

    def __neg__(self):
        return R(_norm((~self.a & LMASK) + _one_low(self.a.shape[1])))

    def __sub__(self, o):
        return self + (-o)

    def add_const(self, c):
        c %= 1 << 128
        limbs = np.array([(c >> (32 * i)) & 0xFFFFFFFF for i in range(LIMBS)], dtype=U64)
        return R(_norm(self.a + limbs[:, None]))

    def shr(self, f):
        assert 0 < f < LIMB_BITS
        a = self.a
        out = np.empty_like(a)
        for i in range(LIMBS - 1):
            out[i] = (a[i] >> U64(f)) | ((a[i + 1] << U64(LIMB_BITS - f)) & LMASK)
        out[LIMBS - 1] = a[LIMBS - 1] >> U64(f)
        return R(out)

    def below_pow2(self, bits):
        top = bits // LIMB_BITS
        rem = bits % LIMB_BITS
        ok = np.ones(self.a.shape[1], dtype=bool)
        for i in range(top + 1, LIMBS):
            ok &= self.a[i] == 0
        if top < LIMBS:
            ok &= self.a[top] < U64(1 << rem)
        return bool(np.all(ok))


def _one_low(m):
    one = np.zeros((LIMBS, m), dtype=U64)
    one[0] = 1
    return one


def _norm(a):
    a = a.copy()
    for i in range(LIMBS - 1):
        c = a[i] >> U64(LIMB_BITS)
        a[i] &= LMASK
        a[i + 1] += c
    a[LIMBS - 1] &= LMASK
    return a


def rand(rng, m, bits=128):
    a = rng.integers(0, 1 << LIMB_BITS, size=(LIMBS, m), dtype=U64)
    top = bits // LIMB_BITS
    rem = bits % LIMB_BITS
    if top < LIMBS:
        a[top] &= U64((1 << rem) - 1)
        a[top + 1:] = 0
    return R(a)


def split_weights(M):
    M = M.tocsr()
    hi = M.copy()
    hi.data = (M.data >> U64(W_SPLIT)).astype(U64)
    lo = M.copy()
    lo.data = (M.data & U64((1 << W_SPLIT) - 1)).astype(U64)
    return hi, lo


class SparseOp:
    def __init__(self, M):
        M = M.tocsr()
        M.data = M.data.astype(U64)
        if M.nnz:
            assert int(M.data.max()) <= (1 << 21)
            assert int(np.diff(M.indptr).max()) < (1 << 20)
        self.hi, self.lo = split_weights(M)
        self.nnz = M.nnz
        self.shape = M.shape

    def __matmul__(self, x):
        a = x.a
        out = np.zeros((LIMBS, self.shape[0]), dtype=U64)
        lo_mask = U64((1 << (LIMB_BITS - W_SPLIT)) - 1)
        for i in range(LIMBS):
            pl = self.lo @ a[i]
            out[i] += pl & LMASK
            if i + 1 < LIMBS:
                out[i + 1] += pl >> U64(LIMB_BITS)
            ph = self.hi @ a[i]
            out[i] += (ph & lo_mask) << U64(W_SPLIT)
            if i + 1 < LIMBS:
                out[i + 1] += ph >> U64(LIMB_BITS - W_SPLIT)
            out = _norm(out)
        return R(out)
