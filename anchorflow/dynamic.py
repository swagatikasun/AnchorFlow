import math
from collections import Counter
from dataclasses import replace

import numpy as np
import scipy.sparse as sp

from .gen import ACCT, DEP, PUB, Ecosystem, matrix
from .protocol import share
from .ring128 import SparseOp


def apply_update(eco, n_tx, seed=0, new_vertex_frac=0.1):
    rng = np.random.default_rng([seed, 31])
    n_new = int(n_tx * new_vertex_frac)
    accts = np.nonzero(eco.kind == ACCT)[0]
    deps = np.nonzero(eco.kind == DEP)[0]
    pubs = np.nonzero(eco.kind == PUB)[0]
    chain_new = rng.choice(3, size=n_new)
    new_ids = np.arange(eco.n, eco.n + n_new)
    kind = rng.choice(3, size=n_tx, p=[0.4, 0.35, 0.25])
    src = np.empty(n_tx, dtype=np.int64)
    dst = np.empty(n_tx, dtype=np.int64)
    sel = kind == 0
    src[sel] = rng.choice(accts, size=int(sel.sum()))
    owner_src = eco.owner[src[sel]]
    dst_int = np.empty(int(sel.sum()), dtype=np.int64)
    for k in np.unique(owner_src):
        pool = accts[eco.owner[accts] == k]
        m = owner_src == k
        dst_int[m] = rng.choice(pool, size=int(m.sum()))
    dst[sel] = dst_int
    sel = kind == 1
    src[sel] = rng.choice(accts, size=int(sel.sum()))
    pool = np.concatenate([pubs, new_ids]) if n_new else pubs
    dst[sel] = rng.choice(pool, size=int(sel.sum()))
    sel = kind == 2
    src[sel] = rng.choice(pubs, size=int(sel.sum()))
    dst[sel] = rng.choice(deps, size=int(sel.sum()))
    new = replace(eco, n=eco.n + n_new,
                  chain=np.concatenate([eco.chain, chain_new.astype(np.int8)]),
                  owner=np.concatenate([eco.owner, np.full(n_new, -1, dtype=np.int64)]),
                  kind=np.concatenate([eco.kind, np.full(n_new, PUB, dtype=np.int8)]),
                  src=np.concatenate([eco.src, src]), dst=np.concatenate([eco.dst, dst]),
                  amount=np.concatenate([eco.amount, rng.lognormal(0.0, 1.0, size=n_tx)]),
                  scale=np.concatenate([eco.scale, np.ones(n_tx)]),
                  bridge=np.concatenate([eco.bridge, np.zeros(n_tx, dtype=bool)]))
    new.params = dict(eco.params, update_tx=n_tx)
    return new.finalize()


def padded_matrix(eco, n, cols):
    M = matrix(eco, cols).tocoo()
    return sp.csr_matrix((M.data, (M.row, M.col)), shape=(n, n), dtype=np.uint64)


def changed_columns(eco_old, eco_new):
    n = eco_new.n
    out = {}
    for k in range(eco_new.K):
        A = padded_matrix(eco_old, n, eco_old.col_owner == k).tocsc()
        B = padded_matrix(eco_new, n, eco_new.col_owner == k).tocsc()
        A.sort_indices()
        B.sort_indices()
        cols = []
        cand = np.unique(np.concatenate([np.nonzero(np.diff(A.indptr))[0], np.nonzero(np.diff(B.indptr))[0]]))
        for u in cand:
            a = slice(A.indptr[u], A.indptr[u + 1])
            b = slice(B.indptr[u], B.indptr[u + 1])
            if (A.indptr[u + 1] - A.indptr[u] != B.indptr[u + 1] - B.indptr[u]
                    or not np.array_equal(A.indices[a], B.indices[b]) or not np.array_equal(A.data[a], B.data[b])):
                cols.append(u)
        out[k] = np.array(cols, dtype=np.int64)
    return out


def column_restrict(M, cols, n):
    M = M.tocsc()
    keep = np.zeros(n, dtype=bool)
    keep[cols] = True
    M = M[:, :]
    D = sp.diags(keep.astype(np.uint64), dtype=np.uint64, format="csc")
    return (M @ D).tocsr()


def refreshed_dealer(eco_old, eco_new, rng):
    n_v = eco_new.n
    changed = changed_columns(eco_old, eco_new)
    old_ops = [SparseOp(padded_matrix(eco_old, n_v, eco_old.col_owner == k)) for k in range(eco_new.K)]
    plus, minus = [], []
    for k in range(eco_new.K):
        cols = changed[k]
        if len(cols) == 0:
            continue
        plus.append(SparseOp(column_restrict(padded_matrix(eco_new, n_v, eco_new.col_owner == k), cols, n_v)))
        minus.append(SparseOp(column_restrict(padded_matrix(eco_old, n_v, eco_old.col_owner == k), cols, n_v)))
    n_parties = eco_new.K + 1

    def make(rp):
        from .protocol import reconstruct
        r = reconstruct(rp)
        c_old = None
        for op in old_ops:
            term = op @ r
            c_old = term if c_old is None else c_old + term
        old_shares = share(rng, c_old, n_parties)
        delta = None
        for p, q in zip(plus, minus):
            term = (p @ r) - (q @ r)
            delta = term if delta is None else delta + term
        if delta is None:
            return old_shares
        d_shares = share(rng, delta, n_parties)
        return [a + b for a, b in zip(old_shares, d_shares)]

    return make, changed


def refresh_counts(eco_old, eco_new, n, slots, tau=1):
    changed = changed_columns(eco_old, eco_new)
    blocks = math.ceil(eco_new.n / slots)
    new_cols = int(((eco_new.owner >= 0)[eco_old.n:]).sum())
    owners_changed = sum(1 for k in changed if len(changed[k]))
    nnz_terms = 0
    slot_steps = 0
    for k, cols in changed.items():
        if len(cols) == 0:
            continue
        a = column_restrict(padded_matrix(eco_old, eco_new.n, eco_old.col_owner == k), cols, eco_new.n)
        b = column_restrict(padded_matrix(eco_new, eco_new.n, eco_new.col_owner == k), cols, eco_new.n)
        nnz_terms += a.nnz + b.nnz
        rows = np.unique(np.concatenate([a.tocoo().row, b.tocoo().row]))
        if len(rows):
            top = np.zeros(blocks, dtype=np.int64) - 1
            np.maximum.at(top, rows // slots, rows % slots)
            slot_steps += int(top[top >= 0].sum())
    ops = Counter()
    ops["encrypt"] = n * new_cols + owners_changed * blocks + n * blocks
    ops["mul"] = (n - 1) * new_cols + nnz_terms + max(owners_changed - 1, 0) * blocks + n * blocks + blocks
    ops["exp_w"] = nnz_terms
    ops["exp_slot"] = slot_steps
    ops["partial_dec"] = n * blocks
    ops["combine"] = blocks
    ops = Counter({k: v * tau for k, v in ops.items()})
    info = dict(changed_cols=int(sum(len(c) for c in changed.values())), new_cols=new_cols,
                owners_changed=owners_changed, blocks=blocks)
    return ops, info
