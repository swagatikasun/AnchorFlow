from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp

F_W = 20
CHAINS = ("Ethereum", "Tron", "Bitcoin")
CHAIN_SHARE = np.array([0.5, 0.3, 0.2])
PUB, DEP, ACCT, SWAP, VIRT = 0, 1, 2, 3, 4


@dataclass
class Ecosystem:
    n: int
    chain: np.ndarray
    owner: np.ndarray
    kind: np.ndarray
    src: np.ndarray
    dst: np.ndarray
    amount: np.ndarray
    scale: np.ndarray
    bridge: np.ndarray
    seeds: np.ndarray
    endpoints: np.ndarray
    path_vertices: np.ndarray
    K: int
    w_float: np.ndarray = field(default=None)
    w_fp: np.ndarray = field(default=None)
    params: dict = field(default_factory=dict)

    @property
    def m(self):
        return len(self.src)

    @property
    def col_owner(self):
        return self.owner[self.src]

    def finalize(self):
        out = np.bincount(self.src, weights=self.amount, minlength=self.n)
        self.w_float = self.amount / out[self.src] * self.scale
        self.w_fp = np.floor(self.w_float * (1 << F_W)).astype(np.uint64)
        return self


class _Builder:
    def __init__(self, rng):
        self.rng = rng
        self.chain, self.owner, self.kind = [], [], []
        self.src, self.dst, self.amount, self.scale, self.bridge = [], [], [], [], []
        self.n = 0

    def add_vertices(self, count, chain, owner, kind):
        start = self.n
        self.n += count
        self.chain.append(np.broadcast_to(np.asarray(chain, dtype=np.int8), (count,)).copy())
        self.owner.append(np.broadcast_to(np.asarray(owner, dtype=np.int64), (count,)).copy())
        self.kind.append(np.full(count, kind, dtype=np.int8))
        return np.arange(start, start + count)

    def add_edges(self, s, d, amount=None, scale=1.0, bridge=False):
        s = np.asarray(s, dtype=np.int64)
        d = np.asarray(d, dtype=np.int64)
        if amount is None:
            amount = self.rng.lognormal(0.0, 1.0, size=len(s))
        self.src.append(s)
        self.dst.append(d)
        self.amount.append(np.asarray(amount, dtype=np.float64))
        self.scale.append(np.broadcast_to(np.asarray(scale, dtype=np.float64), (len(s),)).copy())
        self.bridge.append(np.full(len(s), bridge, dtype=bool))


def _popular(rng, pool, size, a=1.6):
    w = rng.zipf(a, size=len(pool)).astype(np.float64)
    w /= w.sum()
    return pool[rng.choice(len(pool), size=size, p=w)]


def generate(n_vertices=100_000, K=4, swap_share=0.25, pub_frac=0.7, out_degree=4.0,
             p_internal=0.25, p_cross=0.15, bridge_frac=0.02, virtual_frac=0.01,
             fee=0.02, n_seeds=8, seed=1):
    rng = np.random.default_rng(seed)
    b = _Builder(rng)
    K_swap = max(1, int(round(K * swap_share))) if K >= 2 else 0
    K_ex = K - K_swap
    n_pub = int(n_vertices * pub_frac)
    rest = n_vertices - n_pub
    n_swap_dep = int(rest * 0.1) if K_swap else 0
    n_acct = max((rest - n_swap_dep) // 2, K_ex * 4)

    pub_chain = rng.choice(3, size=n_pub, p=CHAIN_SHARE)
    pub = b.add_vertices(n_pub, pub_chain, -1, PUB)
    by_chain = [pub[pub_chain == c] for c in range(3)]

    ex_share = 1.0 / np.arange(1, K_ex + 1)
    ex_share /= ex_share.sum()
    acct_owner = rng.choice(K_ex, size=n_acct, p=ex_share)
    dep_chain = rng.choice(3, size=n_acct, p=CHAIN_SHARE)
    deps = b.add_vertices(n_acct, dep_chain, acct_owner, DEP)
    accts = b.add_vertices(n_acct, -1, acct_owner, ACCT)

    swap_owner = K_ex + rng.integers(0, K_swap, size=n_swap_dep) if K_swap else np.zeros(0, dtype=np.int64)
    swap_chain = rng.choice(3, size=n_swap_dep, p=CHAIN_SHARE)
    swaps = b.add_vertices(n_swap_dep, swap_chain, swap_owner, SWAP)

    m_pub = int(out_degree * n_pub)
    s_chain = rng.choice(3, size=m_pub, p=CHAIN_SHARE)
    ps, pd = [], []
    for c in range(3):
        cnt = int((s_chain == c).sum())
        ps.append(rng.choice(by_chain[c], size=cnt))
        pd.append(_popular(rng, by_chain[c], cnt))
    ps, pd = np.concatenate(ps), np.concatenate(pd)
    keep = ps != pd
    b.add_edges(ps[keep], pd[keep])

    m_br = int(bridge_frac * m_pub)
    bc = rng.choice(3, size=m_br, p=CHAIN_SHARE)
    bd_c = (bc + rng.integers(1, 3, size=m_br)) % 3
    bs = np.empty(m_br, dtype=np.int64)
    bd = np.empty(m_br, dtype=np.int64)
    for c in range(3):
        sel = bc == c
        bs[sel] = rng.choice(by_chain[c], size=int(sel.sum()))
        sel = bd_c == c
        bd[sel] = rng.choice(by_chain[c], size=int(sel.sum()))
    b.add_edges(bs, bd, bridge=True)

    n_in = rng.integers(1, 4, size=n_acct)
    dsrc = np.concatenate([rng.choice(by_chain[c], size=int(n_in[dep_chain == c].sum())) for c in range(3)])
    ddst = np.concatenate([np.repeat(deps[dep_chain == c], n_in[dep_chain == c]) for c in range(3)])
    b.add_edges(dsrc, ddst)
    b.add_edges(deps, accts, amount=np.ones(n_acct))

    n_out = rng.integers(1, 5, size=n_acct)
    osrc = np.repeat(accts, n_out)
    u = rng.random(len(osrc))
    is_int = u < p_internal
    is_cross = (u >= p_internal) & (u < p_internal + p_cross)
    is_virt = (u >= p_internal + p_cross) & (u < p_internal + p_cross + virtual_frac)
    is_wd = ~(is_int | is_cross | is_virt)
    own_of_src = acct_owner[osrc - accts[0]]
    int_dst = np.empty(int(is_int.sum()), dtype=np.int64)
    for k in range(K_ex):
        sel = own_of_src[is_int] == k
        pool = accts[acct_owner == k]
        int_dst[sel] = rng.choice(pool, size=int(sel.sum()))
    b.add_edges(osrc[is_int], int_dst)
    cross_dst = rng.choice(deps, size=int(is_cross.sum()))
    clash = acct_owner[cross_dst - deps[0]] == own_of_src[is_cross]
    cross_dst[clash] = rng.choice(deps, size=int(clash.sum()))
    b.add_edges(osrc[is_cross], cross_dst)
    wd_c = rng.choice(3, size=int(is_wd.sum()), p=CHAIN_SHARE)
    wd_src = osrc[is_wd]
    wd_dst = np.empty(len(wd_src), dtype=np.int64)
    for c in range(3):
        sel = wd_c == c
        wd_dst[sel] = _popular(rng, by_chain[c], int(sel.sum()))
    b.add_edges(wd_src, wd_dst)
    n_virt = int(is_virt.sum())
    if n_virt and K_ex >= 2:
        v_src = osrc[is_virt]
        v_from = own_of_src[is_virt]
        v_to = (v_from + rng.integers(1, K_ex, size=n_virt)) % K_ex
        virt = b.add_vertices(n_virt, -1, v_to, VIRT)
        b.add_edges(v_src, virt)
        v_dst = np.empty(n_virt, dtype=np.int64)
        for k in range(K_ex):
            sel = v_to == k
            v_dst[sel] = rng.choice(accts[acct_owner == k], size=int(sel.sum()))
        b.add_edges(virt, v_dst, amount=np.ones(n_virt))

    if n_swap_dep:
        s_in = np.concatenate([rng.choice(by_chain[c], size=int((swap_chain == c).sum())) for c in range(3)])
        s_dep = np.concatenate([swaps[swap_chain == c] for c in range(3)])
        b.add_edges(s_in, s_dep)
        out_c = (swap_chain + rng.integers(1, 3, size=n_swap_dep)) % 3
        s_out = np.empty(n_swap_dep, dtype=np.int64)
        for c in range(3):
            sel = out_c == c
            s_out[sel] = rng.choice(by_chain[c], size=int(sel.sum()))
        b.add_edges(swaps, s_out, amount=np.ones(n_swap_dep), scale=1.0 - fee)

    seeds, endpoints, path = _plant(b, rng, n_seeds, K_ex, K_swap, acct_owner, accts, by_chain, fee)

    eco = Ecosystem(n=b.n, chain=np.concatenate(b.chain), owner=np.concatenate(b.owner),
                    kind=np.concatenate(b.kind), src=np.concatenate(b.src), dst=np.concatenate(b.dst),
                    amount=np.concatenate(b.amount), scale=np.concatenate(b.scale),
                    bridge=np.concatenate(b.bridge), seeds=seeds,
                    endpoints=endpoints, path_vertices=path, K=K,
                    params=dict(n_vertices=n_vertices, K=K, swap_share=swap_share, pub_frac=pub_frac,
                                out_degree=out_degree, p_internal=p_internal, p_cross=p_cross,
                                bridge_frac=bridge_frac, virtual_frac=virtual_frac, fee=fee,
                                n_seeds=n_seeds, seed=seed))
    return eco.finalize()


def _plant(b, rng, n_seeds, K_ex, K_swap, acct_owner, accts, by_chain, fee):
    seeds, endpoints, path = [], [], []
    for i in range(n_seeds):
        hack = b.add_vertices(1, 0, -1, PUB)
        peel = b.add_vertices(3, 0, -1, PUB)
        b.add_edges(np.repeat(hack, 3), peel, amount=np.ones(3))
        mid = list(peel)
        if K_swap and i % 2 == 0:
            S = K_ex + int(rng.integers(0, K_swap))
            sdep = b.add_vertices(3, 0, S, SWAP)
            t1 = b.add_vertices(3, 1, -1, PUB)
            b.add_edges(peel, sdep, amount=np.ones(3))
            b.add_edges(sdep, t1, amount=np.ones(3), scale=1.0 - fee)
            mid += list(sdep) + list(t1)
            src_tron = t1
        else:
            t1 = b.add_vertices(3, 1, -1, PUB)
            b.add_edges(peel, t1, amount=np.ones(3), bridge=True)
            mid += list(t1)
            src_tron = t1
        t2 = b.add_vertices(1, 1, -1, PUB)
        b.add_edges(src_tron, np.repeat(t2, len(src_tron)), amount=np.ones(len(src_tron)))
        e5 = b.add_vertices(1, 0, -1, PUB)
        X = int(rng.integers(0, K_ex))
        Y = (X + 1) % K_ex if K_ex > 1 else X
        dx = b.add_vertices(1, 1, X, DEP)
        b.add_edges(t2, e5, amount=np.array([0.3]), bridge=True)
        b.add_edges(t2, dx, amount=np.array([0.7]))
        a1, a2 = b.add_vertices(2, -1, X, ACCT)
        b.add_edges(dx, [a1], amount=np.ones(1))
        btc = b.add_vertices(2, 2, -1, PUB)
        dy = b.add_vertices(1, 0, Y, DEP)
        b.add_edges([a1, a1], [a2, btc[0]], amount=np.array([0.5, 0.5]))
        b.add_edges([a2, a2], [btc[1], dy[0]], amount=np.array([0.5, 0.5]))
        a3 = b.add_vertices(1, -1, Y, ACCT)
        b.add_edges(dy, a3, amount=np.ones(1))
        out3 = b.add_vertices(1, 2, -1, PUB)
        b.add_edges(a3, out3, amount=np.ones(1))
        sink = np.concatenate([e5, btc, out3])
        b.add_edges(sink, _popular(rng, by_chain[0], len(sink)), amount=np.ones(len(sink)))
        seeds.append(int(hack[0]))
        endpoints += [int(e5[0]), int(btc[0]), int(btc[1]), int(out3[0]), int(a1), int(a2), int(a3[0])]
        path += mid + [int(t2[0]), int(dx[0]), int(dy[0])] + endpoints[-7:]
    return np.array(seeds), np.array(sorted(set(endpoints))), np.array(sorted(set(path)))


def seed_vector(eco):
    x = np.zeros(eco.n)
    x[eco.seeds] = 1.0 / len(eco.seeds)
    return x


def matrix(eco, cols=None, fixed=True):
    sel = np.ones(eco.m, dtype=bool) if cols is None else cols
    if fixed:
        return sp.csr_matrix((eco.w_fp[sel], (eco.dst[sel], eco.src[sel])), shape=(eco.n, eco.n), dtype=np.uint64)
    return sp.csr_matrix((eco.w_float[sel], (eco.dst[sel], eco.src[sel])), shape=(eco.n, eco.n))


def plaintext_exposure(eco, tau, cols=None, quantized=True, x0=None):
    if quantized:
        M = matrix(eco, cols, fixed=True).astype(np.float64) / (1 << F_W)
    else:
        M = matrix(eco, cols, fixed=False)
    x = seed_vector(eco) if x0 is None else x0
    acc = np.zeros(eco.n)
    for _ in range(tau):
        x = M @ x
        acc += x
    return acc
