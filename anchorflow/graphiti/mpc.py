import numpy as np

from .ring import U64, ALL1, MASK63, rand_u64

W = 8


class Shared:
    __slots__ = ("s0", "s1")

    def __init__(self, s0, s1):
        self.s0 = s0
        self.s1 = s1

    def __add__(self, o):
        return Shared(self.s0 + o.s0, self.s1 + o.s1)

    def __sub__(self, o):
        return Shared(self.s0 - o.s0, self.s1 - o.s1)

    def __getitem__(self, idx):
        return Shared(self.s0[idx], self.s1[idx])

    def __len__(self):
        return len(self.s0)

    def add_public(self, c):
        return Shared(self.s0 + np.asarray(c, dtype=U64), self.s1.copy())

    def mul_public(self, c):
        c = np.asarray(c, dtype=U64)
        return Shared(self.s0 * c, self.s1 * c)

    def cumsum(self):
        return Shared(np.cumsum(self.s0, axis=0, dtype=U64), np.cumsum(self.s1, axis=0, dtype=U64))

    def permute(self, p):
        return Shared(self.s0[p], self.s1[p])

    def copy(self):
        return Shared(self.s0.copy(), self.s1.copy())

    @staticmethod
    def zeros(shape):
        return Shared(np.zeros(shape, dtype=U64), np.zeros(shape, dtype=U64))

    @staticmethod
    def stack(cols):
        return Shared(np.stack([c.s0 for c in cols], axis=1), np.stack([c.s1 for c in cols], axis=1))

    def col(self, j):
        return Shared(self.s0[:, j].copy(), self.s1[:, j].copy())


def inverse(p):
    q = np.empty_like(p)
    q[p] = np.arange(len(p), dtype=p.dtype)
    return q


class Context:
    def __init__(self, meter, seed=0):
        self.meter = meter
        root = np.random.SeedSequence(seed)
        s = root.spawn(5)
        self.rng_h = np.random.default_rng(s[0])
        self.rng_h0 = np.random.default_rng(s[1])
        self.rng_h1 = np.random.default_rng(s[2])
        self.rng_01 = np.random.default_rng(s[3])
        self.rng_party = np.random.default_rng(s[4])

    def share_from_owner(self, values, step="input"):
        v = np.asarray(values, dtype=U64)
        s1 = rand_u64(self.rng_party, v.shape)
        s0 = v - s1
        self.meter.comm(step, 1, O_S=v.size * W)
        return Shared(s0, s1)

    def share_public(self, values):
        v = np.asarray(values, dtype=U64)
        return Shared(v.copy(), np.zeros_like(v))

    def rerandomize(self, x):
        rho = rand_u64(self.rng_01, x.s0.shape)
        return Shared(x.s0 + rho, x.s1 - rho)

    def reveal(self, x, step="reveal"):
        self.meter.comm(step, 1, S0_S1=x.s0.size * W, S1_S0=x.s1.size * W)
        return x.s0 + x.s1

    def reveal_to_owner(self, x, step="output", nbytes=W):
        self.meter.comm(step, 1, S_O=2 * x.s0.size * nbytes)
        return x.s0 + x.s1

    def _triples(self, shape, boolean=False):
        with self.meter.in_phase("offline"):
            a0 = rand_u64(self.rng_h0, shape)
            b0 = rand_u64(self.rng_h0, shape)
            c0 = rand_u64(self.rng_h0, shape)
            a1 = rand_u64(self.rng_h1, shape)
            b1 = rand_u64(self.rng_h1, shape)
            if boolean:
                c1 = ((a0 ^ a1) & (b0 ^ b1)) ^ c0
            else:
                c1 = (a0 + a1) * (b0 + b1) - c0
            self.meter.comm("triples", 0, H_S1=c1.size * W)
        return (a0, b0, c0), (a1, b1, c1)

    def mul(self, x, y, step="mul"):
        (a0, b0, c0), (a1, b1, c1) = self._triples(x.s0.shape)
        d0, e0 = x.s0 - a0, y.s0 - b0
        d1, e1 = x.s1 - a1, y.s1 - b1
        self.meter.comm(step, 1, S0_S1=2 * d0.size * W, S1_S0=2 * d1.size * W)
        d = d0 + d1
        e = e0 + e1
        z0 = c0 + d * b0 + e * a0 + d * e
        z1 = c1 + d * b1 + e * a1
        return Shared(z0, z1)

    def band(self, x, y, step="and"):
        (a0, b0, c0), (a1, b1, c1) = self._triples(x.s0.shape, boolean=True)
        d0, e0 = x.s0 ^ a0, y.s0 ^ b0
        d1, e1 = x.s1 ^ a1, y.s1 ^ b1
        d = d0 ^ d1
        e = e0 ^ e1
        z0 = c0 ^ (d & b0) ^ (e & a0) ^ (d & e)
        z1 = c1 ^ (d & b1) ^ (e & a1)
        return Shared(z0, z1), (d0.size * 2 * W)

    def band_pair(self, x1, y1, x2, y2, step):
        r1, n1 = self.band(x1, y1, step)
        r2, n2 = self.band(x2, y2, step)
        self.meter.comm(step, 1, S0_S1=n1 + n2, S1_S0=n1 + n2)
        return r1, r2

    @staticmethod
    def trunc(x, f):
        s0 = x.s0 >> U64(f)
        s1 = U64(0) - ((U64(0) - x.s1) >> U64(f))
        return Shared(s0, s1)

    def msb(self, z, step="cmp"):
        shape = z.s0.shape
        with self.meter.in_phase("offline"):
            r = rand_u64(self.rng_h, shape)
            r0 = rand_u64(self.rng_h0, shape)
            rb0 = rand_u64(self.rng_h0, shape)
            r1 = r - r0
            rb1 = r ^ rb0
            self.meter.comm("cmp-prep", 0, H_S1=2 * r.size * W)
        c0 = z.s0 + r0
        c1 = z.s1 + r1
        self.meter.comm(step, 1, S0_S1=c0.size * W, S1_S0=c1.size * W)
        c = c0 + c1
        cm = c & MASK63
        nc = (~cm) & ALL1
        e = Shared(nc ^ (rb0 & MASK63), rb1 & MASK63)
        g = Shared(nc & (rb0 & MASK63), nc & (rb1 & MASK63))
        s = 1
        while s < 64:
            gs = Shared(g.s0 << U64(s), g.s1 << U64(s))
            if s < 32:
                es = Shared(e.s0 << U64(s), e.s1 << U64(s))
                ge, ee = self.band_pair(e, gs, e, es, step)
                g = Shared(g.s0 ^ ge.s0, g.s1 ^ ge.s1)
                e = ee
            else:
                ge, n = self.band(e, gs, step)
                self.meter.comm(step, 1, S0_S1=n, S1_S0=n)
                g = Shared(g.s0 ^ ge.s0, g.s1 ^ ge.s1)
            s *= 2
        top = U64(63)
        lt0 = (g.s0 >> top) & U64(1)
        lt1 = (g.s1 >> top) & U64(1)
        m0 = ((c >> top) & U64(1)) ^ ((rb0 >> top) & U64(1)) ^ lt0
        m1 = ((rb1 >> top) & U64(1)) ^ lt1
        return Shared(m0, m1)

    def ge_public(self, x, theta, step="cmp"):
        m = self.msb(x.add_public(U64(0) - np.asarray(theta, dtype=U64)), step)
        return Shared(m.s0 ^ U64(1), m.s1)


class SharedPerm:
    def __init__(self, ctx, sigma, step="perm"):
        self.ctx = ctx
        self.sigma = sigma
        n = len(sigma)
        with ctx.meter.in_phase("init"):
            self.i0 = ctx.rng_h0.permutation(n)
            self.i1 = ctx.rng_h1.permutation(n)
            self.o1 = inverse(self.i0)[sigma]
            self.o0 = inverse(self.i1)[sigma]
            idx = 4 if n < (1 << 32) else 8
            ctx.meter.comm(step, 1, H_S0=n * idx, H_S1=n * idx)
            self.inv = None

    def _inverse_material(self):
        if self.inv is None:
            self.inv = (inverse(self.o0), inverse(self.i1), inverse(self.o1), inverse(self.i0), inverse(self.sigma))
        return self.inv

    def apply(self, x, inverse_perm=False, step="shuffle"):
        ctx = self.ctx
        if inverse_perm:
            p0_in, p1_out, p1_in, p0_out, sig = self._inverse_material()
        else:
            p0_in, p1_out, p1_in, p0_out, sig = self.i0, self.o1, self.i1, self.o0, self.sigma
        shape = x.s0.shape
        with ctx.meter.in_phase("offline"):
            R = rand_u64(ctx.rng_h, shape)
            R0 = rand_u64(ctx.rng_h0, shape)
            R1 = rand_u64(ctx.rng_h1, shape)
            B0 = R1[sig] - R
            B1 = R0[sig] + R
            ctx.meter.comm("shuffle-prep", 0, H_S0=B0.size * W, H_S1=B1.size * W)
        a_to_p1 = (x.s0 + R0)[p0_in]
        a_to_p0 = (x.s1 + R1)[p1_in]
        ctx.meter.comm(step, 1, S0_S1=a_to_p1.size * W, S1_S0=a_to_p0.size * W)
        out0 = a_to_p0[p0_out] - B0
        out1 = a_to_p1[p1_out] - B1
        return Shared(out0, out1)


def random_shared_perm(ctx, n, step="perm"):
    with ctx.meter.in_phase("init"):
        sigma = ctx.rng_h.permutation(n)
    return SharedPerm(ctx, sigma, step)


def composed_inverse_perm(ctx, pa, pb, step="perm"):
    with ctx.meter.in_phase("init"):
        comp = pa.sigma[pb.sigma]
        sigma = inverse(comp)
    return SharedPerm(ctx, sigma, step)


def insecure_sort_perm(ctx, keys, step="isort"):
    n = len(keys)
    order = np.arange(n)
    seg_start = np.zeros(n, dtype=np.int64)
    seg_len = np.full(n, n, dtype=np.int64)
    levels = 0
    while True:
        pos = np.arange(n)
        active = (seg_len >= 2) & (pos != seg_start)
        if not active.any():
            break
        levels += 1
        act = np.nonzero(active)[0]
        piv = seg_start[act]
        ki = keys.permute(order[act])
        kp = keys.permute(order[piv])
        m = ctx.msb(ki - kp, step)
        lt = ctx.reveal(m, step).astype(np.int64) & 1
        cls = np.ones(n, dtype=np.int64)
        cls[act] = np.where(lt == 1, 0, 2)
        sorted_idx = np.lexsort((pos, cls, seg_start))
        order = order[sorted_idx]
        new_start_key = seg_start[sorted_idx] * 4 + cls[sorted_idx]
        brk = np.ones(n, dtype=bool)
        brk[1:] = new_start_key[1:] != new_start_key[:-1]
        starts = np.nonzero(brk)[0]
        lens = np.diff(np.append(starts, n))
        seg_id = np.cumsum(brk) - 1
        seg_start = starts[seg_id]
        seg_len = lens[seg_id]
    return order, levels
