import math
import secrets
import time
from collections import Counter

import numpy as np

from . import paillier
from .gen import F_W, matrix
from .protocol import K_BITS, SIGMA
from .ring128 import R

WMAX_BITS = 20


class Clock:
    def __init__(self):
        self.t = Counter()
        self.ops = Counter()

    def run(self, name, fn, *args):
        t0 = time.perf_counter()
        out = fn(*args)
        self.t[name] += time.perf_counter() - t0
        self.ops[name] += 1
        return out


def params(pub, n):
    beta, c_bits, rho_bits = paillier.slot_bits(K_BITS, SIGMA, n, WMAX_BITS, F_W)
    s = paillier.slots_per_ct(pub, beta)
    return beta, c_bits, rho_bits, s


def _columns(eco, k):
    M = matrix(eco, eco.col_owner == k).tocsr()
    return M


def offline_correlation(eco, key, rprime_parts, mode="L2"):
    pub = key.pub
    n = len(rprime_parts)
    m = eco.n
    K = eco.K
    beta, c_bits, rho_bits, s = params(pub, n)
    clk = Clock()
    nbytes = Counter()
    ctb = pub.ct_bytes()
    cols = np.nonzero(eco.owner >= 0)[0] if mode == "L1" else np.arange(m)
    enc_masks = []
    for j in range(n):
        vals = rprime_parts[j]
        enc_masks.append({int(u): clk.run("encrypt", pub.encrypt, vals[u]) for u in cols})
    nbytes["mask-broadcast"] += n * (n - 1) * len(cols) * ctb
    n_blocks = (m + s - 1) // s
    packed = [None] * n_blocks
    for k in range(K):
        Mk = _columns(eco, k)
        own_cols = np.nonzero(eco.owner == k)[0]
        enc_r = {}
        for u in own_cols:
            c = enc_masks[0][int(u)]
            for j in range(1, n):
                c = clk.run("mul", pub.add, c, enc_masks[j][int(u)])
            enc_r[int(u)] = c
        rows = {}
        coo = Mk.tocoo()
        for v, u, wgt in zip(coo.row.tolist(), coo.col.tolist(), coo.data.tolist()):
            term = clk.run("exp_w", pub.smul, enc_r[u], int(wgt))
            rows[v] = term if v not in rows else clk.run("mul", pub.add, rows[v], term)
        shift = 1 << beta
        for b in range(n_blocks):
            lo, hi = b * s, min(m, (b + 1) * s)
            acc = None
            for v in range(hi - 1, lo - 1, -1):
                if acc is not None:
                    acc = clk.run("exp_slot", pub.smul, acc, shift)
                if v in rows:
                    acc = rows[v] if acc is None else clk.run("mul", pub.add, acc, rows[v])
            fresh = clk.run("encrypt", pub.encrypt, 0)
            acc = fresh if acc is None else clk.run("mul", pub.add, acc, fresh)
            packed[b] = acc if packed[b] is None else clk.run("mul", pub.add, packed[b], acc)
        nbytes["owner-packed"] += n_blocks * ctb
    rho = [[secrets.randbits(rho_bits) for _ in range(m)] for _ in range(n)]
    for j in range(n):
        for b in range(n_blocks):
            lo, hi = b * s, min(m, (b + 1) * s)
            val = sum(rho[j][v] << (beta * (v - lo)) for v in range(lo, hi))
            packed[b] = clk.run("mul", pub.add, packed[b], clk.run("encrypt", pub.encrypt, val))
    nbytes["rho-masks"] += n * n_blocks * ctb
    nbytes["aggregate-broadcast"] += (n - 1) * n_blocks * ctb
    plain = []
    for b in range(n_blocks):
        parts = [clk.run("partial_dec", key.partial, j, packed[b]) for j in range(n)]
        plain.append(clk.run("combine", key.combine, parts))
    nbytes["partial-decryptions"] += n * (n - 1) * n_blocks * ctb
    slot_mask = (1 << beta) - 1
    opened = []
    for b in range(n_blocks):
        lo, hi = b * s, min(m, (b + 1) * s)
        for v in range(lo, hi):
            opened.append((plain[b] >> (beta * (v - lo))) & slot_mask)
    mod = 1 << 128
    shares = [R.from_ints([(opened[v] - rho[0][v]) % mod for v in range(m)])]
    for j in range(1, n):
        shares.append(R.from_ints([(-rho[j][v]) % mod for v in range(m)]))
    info = dict(beta=beta, slots=s, c_bits=c_bits, rho_bits=rho_bits, blocks=n_blocks,
                seconds=dict(clk.t), ops=dict(clk.ops), bytes=dict(nbytes), cols=len(cols))
    return shares, info


def unit_costs(bits=2048, n=5, trials=20):
    key = paillier.keygen(bits, n)
    pub = key.pub
    beta, _, _, s = params(pub, n)
    out = {}
    c = pub.encrypt(12345)

    def avg(fn, k=trials):
        t0 = time.perf_counter()
        for _ in range(k):
            fn()
        return (time.perf_counter() - t0) / k

    out["encrypt"] = avg(lambda: pub.encrypt(secrets.randbits(200)))
    out["mul"] = avg(lambda: pub.add(c, c), trials * 50)
    out["exp_w"] = avg(lambda: pub.smul(c, secrets.randbits(F_W) | 1), trials * 5)
    out["exp_slot"] = avg(lambda: pub.smul(c, 1 << beta), trials * 5)
    out["exp_scale"] = avg(lambda: pub.smul(c, 1 << (F_W * 5)), trials * 5)
    out["partial_dec"] = avg(lambda: key.partial(0, c), max(trials // 2, 3))
    out["combine"] = avg(lambda: key.combine([c] * n), trials * 5)
    return dict(bits=bits, n=n, beta=beta, slots=s, ct_bytes=pub.ct_bytes(), unit_s=out, backend=paillier.BACKEND)


def anchorflow_offline_counts(eco, n, tau, mode, slots, nnz_own=None):
    m = eco.n
    C = int((eco.owner >= 0).sum())
    cols = C if mode == "L1" else m
    nnz = unique_pairs(eco, eco.col_owner >= 0) if nnz_own is None else nnz_own
    rows_touched_k = sum(len(np.unique(eco.dst[eco.col_owner == k])) for k in range(eco.K))
    blocks = math.ceil(m / slots)
    slot_steps = 0
    for k in range(eco.K):
        r = np.unique(eco.dst[eco.col_owner == k])
        if len(r):
            top = np.zeros(blocks, dtype=np.int64) - 1
            np.maximum.at(top, r // slots, r % slots)
            slot_steps += int(top[top >= 0].sum())
    ops = Counter()
    ops["encrypt"] = n * cols + eco.K * blocks + n * blocks
    ops["mul"] = (n - 1) * C + nnz + (eco.K - 1) * blocks + n * blocks
    ops["exp_w"] = nnz
    ops["exp_slot"] = slot_steps
    ops["partial_dec"] = n * blocks
    ops["combine"] = blocks
    ops = Counter({k: v * tau for k, v in ops.items()})
    return ops, dict(blocks=blocks, cols=cols, C=C, nnz=nnz)


def anchorflow_offline_bytes(eco, n, tau, mode, slots, ct_bytes):
    m = eco.n
    C = int((eco.owner >= 0).sum())
    cols = C if mode == "L1" else m
    blocks = math.ceil(m / slots)
    per_iter = (n * (n - 1) * cols + eco.K * blocks + n * blocks + (n - 1) * blocks
                + n * (n - 1) * blocks) * ct_bytes
    return per_iter * tau


def unique_pairs(eco, sel=None):
    sel = np.ones(eco.m, dtype=bool) if sel is None else sel
    return len(np.unique(eco.dst[sel].astype(np.int64) * eco.n + eco.src[sel]))


def cost_from_counts(ops, unit):
    return sum(unit[k] * v for k, v in ops.items())


def party_of_vertex(eco):
    return np.where(eco.owner >= 0, eco.owner + 1, 0)


def egmond_counts(eco, n, tau):
    part = party_of_vertex(eco)
    ps, pd = part[eco.src], part[eco.dst]
    cross = ps != pd
    pairs = np.unique(np.stack([ps[cross], eco.dst[cross]]), axis=1).shape[1] if cross.any() else 0
    local_pairs = np.unique(np.stack([ps, eco.dst]), axis=1).shape[1]
    ops = Counter()
    ops["encrypt"] = eco.n + tau * pairs
    nnz = unique_pairs(eco)
    ops["exp_w"] = tau * nnz
    n_recv = len(np.unique(eco.dst))
    ops["mul"] = tau * (nnz - local_pairs) + tau * pairs + tau * (local_pairs - n_recv) + (tau - 1) * eco.n
    ops["exp_scale"] = tau * eco.n
    ops["partial_dec"] = n * eco.n
    ops["combine"] = eco.n
    return ops, dict(cross_pairs=int(pairs))


def egmond_bytes(eco, n, tau, ct_bytes):
    _, info = egmond_counts(eco, n, tau)
    initial = eco.n * ct_bytes
    per_iter = info["cross_pairs"] * ct_bytes
    final = n * (n - 1) * eco.n * ct_bytes
    return initial, per_iter * tau, final


def egmond_run(eco, key, tau, x0_fixed):
    pub = key.pub
    n = len(key.shares)
    clk = Clock()
    part = party_of_vertex(eco)
    M = matrix(eco).tocsc()
    enc = [clk.run("encrypt", pub.encrypt, int(v)) for v in x0_fixed]
    states = []
    for _ in range(tau):
        contrib = {}
        for u in range(eco.n):
            lo, hi = M.indptr[u], M.indptr[u + 1]
            for idx in range(lo, hi):
                v = int(M.indices[idx])
                term = clk.run("exp_w", pub.smul, enc[u], int(M.data[idx]))
                key_pv = (int(part[u]), v)
                contrib[key_pv] = term if key_pv not in contrib else clk.run("mul", pub.add, contrib[key_pv], term)
        new = [None] * eco.n
        for (p, v), c in contrib.items():
            if p != part[v]:
                c = clk.run("mul", pub.add, c, clk.run("encrypt", pub.encrypt, 0))
            new[v] = c if new[v] is None else clk.run("mul", pub.add, new[v], c)
        enc = [c if c is not None else pub.encrypt(0) for c in new]
        states.append(enc)
    acc = []
    for v in range(eco.n):
        a = None
        for t, st in enumerate(states):
            term = clk.run("exp_scale", pub.smul, st[v], 1 << (F_W * (tau - 1 - t)))
            a = term if a is None else clk.run("mul", pub.add, a, term)
        acc.append(a)
    out = []
    for v in range(eco.n):
        parts = [clk.run("partial_dec", key.partial, j, acc[v]) for j in range(n)]
        out.append(clk.run("combine", key.combine, parts))
    return out, dict(seconds=dict(clk.t), ops=dict(clk.ops))
