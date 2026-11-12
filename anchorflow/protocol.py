import hashlib
import math
import time

import numpy as np

from .gen import F_W, matrix, seed_vector
from .meter import Meter
from .ring128 import ELEM_BYTES, R, SparseOp, rand

F_X = 32
SIGMA = 64
K_BITS = F_X + F_W + 2


class Setup:
    def __init__(self, eco):
        self.eco = eco
        self.K = eco.K
        self.n = eco.K + 1
        co = eco.col_owner
        self.Mp = SparseOp(matrix(eco, co < 0))
        self.Mk = [SparseOp(matrix(eco, co == k)) for k in range(self.K)]
        self.custodial = np.nonzero(eco.owner >= 0)[0]
        self.owned = [np.nonzero(eco.owner == k)[0] for k in range(self.K)]

    def owner_party(self, k):
        return k + 1

    def owned_by_party(self, j):
        return [k for k in range(self.K) if self.owner_party(k) == j]


def encode_vec(x, frac=F_X):
    return R.from_int64(np.round(np.asarray(x) * (1 << frac)).astype(np.int64))


def decode_vec(v, frac=F_X):
    return v.to_int64().astype(np.float64) / (1 << frac)


def share(rng, v, n):
    parts = [rand(rng, len(v)) for _ in range(n - 1)]
    last = v
    for p in parts:
        last = last - p
    return [last] + parts


def reconstruct(parts):
    acc = parts[0]
    for p in parts[1:]:
        acc = acc + p
    return acc


def digest(v):
    return hashlib.sha256(np.ascontiguousarray(v.a).tobytes()).hexdigest()


def matrix_commitment(Mk):
    h = hashlib.sha256()
    for part in (Mk.lo, Mk.hi):
        h.update(part.indptr.tobytes())
        h.update(part.indices.tobytes())
        h.update(part.data.tobytes())
    return h.hexdigest()


def dealer_correlation(setup, rprime_sum, rng):
    c = None
    for Mk in setup.Mk:
        term = Mk @ rprime_sum
        c = term if c is None else c + term
    return share(rng, c, setup.n)


def opening_cost(n, m, mode):
    if mode == "all":
        return 1, n * (n - 1) * m * ELEM_BYTES, (n - 1) * m * ELEM_BYTES
    return 2, 2 * (n - 1) * m * ELEM_BYTES, 2 * (n - 1) * m * ELEM_BYTES


class Party:
    def __init__(self, j, seed, m):
        self.j = j
        self.seed = (seed, j)
        self.rng = np.random.default_rng([seed, 1000 + j])
        self.m = m

    def masks(self, bits):
        r = rand(self.rng, self.m, bits)
        return r, r.shr(F_W)


def input_shares(eco, seed, n, m):
    x0 = encode_vec(seed_vector(eco))
    pair = [rand(np.random.default_rng([seed, 2000 + j]), m) for j in range(1, n)]
    first = x0
    for p in pair:
        first = first - p
    return [first] + pair


def party_message(setup, j, w, x_share, mask, corr_share, delta, own_x=None, deviation=None):
    y = setup.Mp @ x_share
    for k in setup.owned_by_party(j):
        y = y + (setup.Mk[k] @ (w if own_x is None else own_x))
    if corr_share is not None:
        y = y - corr_share
    y = y + mask
    if j == 0:
        y = y.add_const(delta)
    if deviation is not None:
        y = y + deviation
    return y


def deliver(setup, s_shares, seed, meter=None):
    n, m, cust = setup.n, setup.eco.n, setup.custodial
    pos = np.full(m, -1, dtype=np.int64)
    pos[cust] = np.arange(len(cust))
    pads = []
    masked = [sh[cust] for sh in s_shares]
    for k in range(setup.K):
        full = rand(np.random.default_rng([seed, 3000 + k]), len(cust))
        keep = np.zeros(len(cust), dtype=bool)
        keep[pos[setup.owned[k]]] = True
        pad = R(np.where(keep, full.a, np.uint64(0)))
        pads.append(pad)
        for j, part in enumerate(share(np.random.default_rng([seed, 4000 + k]), pad, n)):
            masked[j] = masked[j] + part
    opened = reconstruct(masked)
    owners = []
    for k in range(setup.K):
        idx = pos[setup.owned[k]]
        owners.append(decode_vec(opened[idx] - pads[k][idx]))
    pub = np.nonzero(setup.eco.owner < 0)[0]
    investigator = decode_vec(reconstruct([sh[pub] for sh in s_shares]))
    if meter is not None:
        c = len(cust) * ELEM_BYTES
        meter.comm("output", "pad-shares", 1, setup.K * (n - 1) * c, (n - 1) * c)
        meter.comm("output", "masked-open", 1, n * (n - 1) * c, (n - 1) * c)
        meter.comm("output", "to-investigator", 1, (n - 1) * len(pub) * ELEM_BYTES, (n - 1) * len(pub) * ELEM_BYTES)
    return dict(owners=owners, investigator=investigator, public=pub)


def run(eco, tau, opening="all", debias=True, correlations=None, seed=0, setup=None, check_bounds=True,
        tier="H", record=False, deviations=None):
    setup = setup or Setup(eco)
    n, m = setup.n, eco.n
    meter = Meter()
    dealer_rng = np.random.default_rng([seed, 99])
    public_rng = np.random.default_rng([seed, 7])
    h = (n - 1) // 2 if debias else 0
    coin = debias and n % 2 == 0
    delta = 1 << K_BITS
    mask_bits = K_BITS + SIGMA
    open_bound = K_BITS + SIGMA + math.ceil(math.log2(n)) + 2
    parties = [Party(j, seed, m) for j in range(n)]
    deviations = deviations or {}
    transcript = dict(open=[], private=[], public_w=[], corr=[]) if record else None

    used_corr = []

    def correlation(rp):
        if tier == "V":
            return [None] * n
        if correlations is None:
            t0 = time.perf_counter()
            out = dealer_correlation(setup, reconstruct(rp), dealer_rng)
            meter.add_cpu("dealer", time.perf_counter() - t0)
        else:
            out = correlations(rp)
        if record:
            used_corr.append(out)
        return out

    xs = input_shares(eco, seed, n, m)
    masks = [p.masks(mask_bits) for p in parties]
    rprime = [mk[1] for mk in masks]
    rounds, total, busiest = opening_cost(n, m, opening)
    priv_bytes = (n - 1) * len(setup.custodial) * ELEM_BYTES if tier == "V" else 0
    largest = max((len(v) for v in setup.owned), default=0)
    priv_busiest = max(len(setup.custodial), (n - 1) * largest) * ELEM_BYTES if tier == "V" else 0
    t0 = time.perf_counter()
    msgs = [xs[j] + rprime[j] for j in range(n)]
    w = reconstruct(msgs)
    meter.add_cpu("online", time.perf_counter() - t0)
    meter.comm("online", "input-open", rounds, total + priv_bytes, busiest + priv_busiest)
    if record:
        transcript["open"].append([digest(mg) for mg in msgs])
        transcript["private"].append([digest(r) for r in rprime] if tier == "V" else None)
        transcript["public_w"].append(w)
    xs = [(w - rprime[0])] + [-rprime[j] for j in range(1, n)]
    acc = [R.zeros(m) for _ in range(n)]
    for t in range(tau):
        cs = correlation(rprime)
        own_x = (w - reconstruct(rprime)) if tier == "V" else None
        if record:
            transcript["corr"].append([digest(c) if c is not None else None for c in cs])
        masks = [p.masks(mask_bits) for p in parties]
        party_time, ys = [], []
        for j in range(n):
            t0 = time.perf_counter()
            ys.append(party_message(setup, j, w, xs[j], masks[j][0], cs[j], delta, own_x,
                                    deviations.get((t, j))))
            party_time.append(time.perf_counter() - t0)
        del cs
        rprime = [mk[1] for mk in masks]
        t0 = time.perf_counter()
        yhat = reconstruct(ys)
        if check_bounds and not deviations:
            assert yhat.below_pow2(open_bound), "masked opening exceeded its bound"
        w = yhat.shr(F_W).add_const(-(delta >> F_W) - h)
        if coin:
            w = w - R.from_int64(public_rng.integers(0, 2, size=m))
        xs = [w - rprime[0]] + [-rprime[j] for j in range(1, n)]
        acc = [a + x for a, x in zip(acc, xs)]
        rec_time = time.perf_counter() - t0
        meter.add_cpu("online", max(party_time) + rec_time)
        meter.add_cpu("online_total", sum(party_time) + rec_time)
        meter.comm("online", "iteration-open", rounds, total + priv_bytes, busiest + priv_busiest)
        if record:
            transcript["open"].append([digest(y) for y in ys])
            transcript["private"].append([digest(r) for r in rprime] if tier == "V" else None)
            transcript["public_w"].append(w)
        del ys
    outputs = deliver(setup, acc, seed, meter)
    s = decode_vec(reconstruct(acc))
    out = dict(exposure=s, outputs=outputs, meter=meter, n=n, m=m, nnz_pub=setup.Mp.nnz,
               nnz_own=[Mk.nnz for Mk in setup.Mk], tier=tier)
    if record:
        out["transcript"] = transcript
        out["corr_shares"] = used_corr
        out["commitments"] = dict(seed=[hashlib.sha256(repr(p.seed).encode()).hexdigest() for p in parties],
                                  matrix=[matrix_commitment(Mk) for Mk in setup.Mk])
    return out


def audit(eco, tau, j, seed, transcript, commitments, corr_shares=None, tier="H", debias=True, setup=None,
          received=None):
    setup = setup or Setup(eco)
    n, m = setup.n, eco.n
    found = []
    if hashlib.sha256(repr((seed, j)).encode()).hexdigest() != commitments["seed"][j]:
        return [("seed", None)]
    for k in setup.owned_by_party(j):
        if matrix_commitment(setup.Mk[k]) != commitments["matrix"][k]:
            return [("matrix", k)]
    delta = 1 << K_BITS
    mask_bits = K_BITS + SIGMA
    me = Party(j, seed, m)
    x_share = input_shares(eco, seed, n, m)[j]
    _, rp = me.masks(mask_bits)
    if digest(x_share + rp) != transcript["open"][0][j]:
        found.append(("input", -1))
    for t in range(tau):
        w = transcript["public_w"][t]
        my_x = (w - rp) if j == 0 else -rp
        corr = None
        if tier == "H":
            corr = corr_shares[t]
            if digest(corr) != transcript["corr"][t][j]:
                found.append(("correction", t))
        own_x = None
        if tier == "V" and setup.owned_by_party(j):
            parts = received[t]
            for i, part in enumerate(parts):
                if digest(part) != transcript["private"][t][i]:
                    found.append(("received", t))
            own_x = w - reconstruct(parts)
        mask, rp = me.masks(mask_bits)
        msg = party_message(setup, j, w, my_x, mask, corr, delta, own_x)
        if digest(msg) != transcript["open"][t + 1][j]:
            found.append(("message", t))
    return found
