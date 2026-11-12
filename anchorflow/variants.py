from dataclasses import replace

import numpy as np

from .gen import PUB


def _rebuild(eco, src, dst, amount, scale, bridge):
    new = replace(eco, src=np.asarray(src, dtype=np.int64), dst=np.asarray(dst, dtype=np.int64),
                  amount=np.asarray(amount, dtype=np.float64), scale=np.asarray(scale, dtype=np.float64),
                  bridge=np.asarray(bridge, dtype=bool))
    return new.finalize()


def uncertain_matching(eco, mode, sigma, decoys=2, sharpness=4.0, seed=0):
    if mode == "oracle" or sigma is None:
        return eco
    rng = np.random.default_rng([seed, 41])
    idx = np.nonzero(eco.bridge)[0]
    keep = np.ones(eco.m, dtype=bool)
    keep[idx] = False
    pools = [np.nonzero((eco.kind == PUB) & (eco.chain == c))[0] for c in range(3)]
    add_s, add_d, add_a, add_sc = [], [], [], []
    for e in idx:
        u, v = int(eco.src[e]), int(eco.dst[e])
        cand = np.concatenate([[v], rng.choice(pools[int(eco.chain[v])], size=decoys)])
        score = rng.normal(0.0, sigma, size=len(cand))
        score[0] += 1.0
        conf = np.exp(sharpness * (score - score.max()))
        conf /= conf.sum()
        if mode == "hard":
            add_s.append(u)
            add_d.append(int(cand[int(np.argmax(conf))]))
            add_a.append(eco.amount[e])
            add_sc.append(eco.scale[e])
        else:
            for c, q in zip(cand, conf):
                add_s.append(u)
                add_d.append(int(c))
                add_a.append(eco.amount[e] * q)
                add_sc.append(eco.scale[e])
    k = len(add_s)
    return _rebuild(eco, np.concatenate([eco.src[keep], add_s]), np.concatenate([eco.dst[keep], add_d]),
                    np.concatenate([eco.amount[keep], add_a]), np.concatenate([eco.scale[keep], add_sc]),
                    np.concatenate([eco.bridge[keep], np.ones(k, dtype=bool)]))


def omit_custodial(eco, frac, targeted=False, seed=0):
    rng = np.random.default_rng([seed, 43])
    cand = eco.col_owner >= 0
    drop = np.zeros(eco.m, dtype=bool)
    if targeted:
        on_path = np.isin(eco.src, eco.path_vertices) & np.isin(eco.dst, eco.path_vertices)
        drop = cand & on_path & (eco.kind[eco.src] == 2)
    else:
        idx = np.nonzero(cand)[0]
        drop[rng.choice(idx, size=int(frac * len(idx)), replace=False)] = True
    keep = ~drop
    return _rebuild(eco, eco.src[keep], eco.dst[keep], eco.amount[keep], eco.scale[keep], eco.bridge[keep])


def anchor_links(eco):
    so, do = eco.owner[eco.src], eco.owner[eco.dst]
    return (so >= 0) & (do >= 0) & (so != do)


def missing_anchors(eco, frac, seed=0):
    rng = np.random.default_rng([seed, 47])
    idx = np.nonzero(anchor_links(eco))[0]
    pick = rng.choice(idx, size=int(round(frac * len(idx))), replace=False)
    sink = eco.n
    dst = eco.dst.copy()
    dst[pick] = sink
    new = replace(eco, n=eco.n + 1, chain=np.concatenate([eco.chain, [np.int8(0)]]),
                  owner=np.concatenate([eco.owner, [-1]]), kind=np.concatenate([eco.kind, [np.int8(PUB)]]))
    return _rebuild(new, eco.src, dst, eco.amount, eco.scale, eco.bridge)


def wrong_bridges(eco, frac, seed=0):
    rng = np.random.default_rng([seed, 53])
    idx = np.nonzero(eco.bridge)[0]
    pick = rng.choice(idx, size=int(round(frac * len(idx))), replace=False)
    pools = [np.nonzero((eco.kind == PUB) & (eco.chain == c))[0] for c in range(3)]
    dst = eco.dst.copy()
    for e in pick:
        dst[e] = rng.choice(pools[int(eco.chain[eco.dst[e]])])
    return _rebuild(eco, eco.src, dst, eco.amount, eco.scale, eco.bridge)


def mix_paths(eco, phi, fan_out=5, seed=0):
    if phi <= 0:
        return eco
    rng = np.random.default_rng([seed, 59])
    inner = np.setdiff1d(eco.path_vertices, eco.endpoints)
    out = np.bincount(eco.src, weights=eco.amount, minlength=eco.n)
    pubs = np.nonzero(eco.kind == PUB)[0]
    add_s = np.repeat(inner, fan_out)
    add_d = rng.choice(pubs, size=len(add_s))
    add_a = np.repeat(out[inner] * phi / (1.0 - phi) / fan_out, fan_out)
    return _rebuild(eco, np.concatenate([eco.src, add_s]), np.concatenate([eco.dst, add_d]),
                    np.concatenate([eco.amount, add_a]), np.concatenate([eco.scale, np.ones(len(add_s))]),
                    np.concatenate([eco.bridge, np.zeros(len(add_s), dtype=bool)]))


def compare(eco, s, oracle, thresholds=(1e-3, 1e-4, 1e-5, 1e-6)):
    out = {}
    n_extra = len(s) - len(oracle)
    if n_extra > 0:
        oracle = np.concatenate([oracle, np.zeros(n_extra)])
    for th in thresholds:
        f, t = s >= th, oracle >= th
        tp = int((f & t).sum())
        fp = int((f & ~t).sum())
        fn = int((~f & t).sum())
        neg = int((~t).sum())
        out[f"{th:g}"] = dict(precision=tp / max(tp + fp, 1), recall=tp / max(tp + fn, 1), fpr=fp / max(neg, 1),
                              endpoint_recall=float((s[eco.endpoints] >= th).mean()), flagged=int(f.sum()),
                              tp=tp, fp=fp, fn=fn)
    return out


def time_expanded_size(eco, windows, seed=0):
    rng = np.random.default_rng([seed, 61])
    w = rng.integers(0, windows, size=eco.m)
    send = eco.src.astype(np.int64) * windows + w
    recv = eco.dst.astype(np.int64) * windows + w
    active = np.unique(np.concatenate([send, recv]))
    owner_of = active // windows
    carry = int((np.diff(owner_of) == 0).sum())
    return dict(windows=int(windows), vertices=int(len(active)), transfer_edges=int(eco.m), carry_edges=carry,
                custodial_vertices=int((eco.owner[owner_of] >= 0).sum()))
