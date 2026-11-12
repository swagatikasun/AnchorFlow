import numpy as np

U64 = np.uint64
FRAC_W = 20
FRAC = 24
SCALE = 1 << FRAC
MASK63 = U64((1 << 63) - 1)
ALL1 = U64((1 << 64) - 1)


def encode(x, frac=FRAC):
    return np.round(np.asarray(x, dtype=np.float64) * (1 << frac)).astype(np.int64).view(U64)


def decode(v, frac=FRAC):
    return np.asarray(v, dtype=U64).view(np.int64).astype(np.float64) / (1 << frac)


def to_signed(v):
    return np.asarray(v, dtype=U64).view(np.int64)


def rand_u64(rng, shape):
    return rng.integers(0, 1 << 64, size=shape, dtype=U64, endpoint=False)
