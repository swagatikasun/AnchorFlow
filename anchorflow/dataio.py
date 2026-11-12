import csv
import os

import numpy as np

from .gen import Ecosystem


def save(eco, folder):
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, "vertices.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "chain", "owner", "kind"])
        for i in range(eco.n):
            w.writerow([i, int(eco.chain[i]), int(eco.owner[i]), int(eco.kind[i])])
    with open(os.path.join(folder, "transfers.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["src", "dst", "amount", "value_kept", "bridge"])
        for s, d, a, sc, b in zip(eco.src, eco.dst, eco.amount, eco.scale, eco.bridge):
            w.writerow([int(s), int(d), repr(float(a)), repr(float(sc)), int(b)])
    with open(os.path.join(folder, "labels.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "label"])
        for v in eco.seeds:
            w.writerow([int(v), "seed"])
        for v in eco.endpoints:
            w.writerow([int(v), "endpoint"])


def load(folder, K=None):
    with open(os.path.join(folder, "vertices.csv")) as f:
        rows = list(csv.DictReader(f))
    ids = np.array([int(r["id"]) for r in rows])
    order = np.argsort(ids)
    if not np.array_equal(ids[order], np.arange(len(ids))):
        raise ValueError("vertex ids must be 0..n-1")
    chain = np.array([int(r["chain"]) for r in rows], dtype=np.int8)[order]
    owner = np.array([int(r["owner"]) for r in rows], dtype=np.int64)[order]
    kind = np.array([int(r["kind"]) for r in rows], dtype=np.int8)[order]
    with open(os.path.join(folder, "transfers.csv")) as f:
        tr = list(csv.DictReader(f))
    src = np.array([int(r["src"]) for r in tr], dtype=np.int64)
    dst = np.array([int(r["dst"]) for r in tr], dtype=np.int64)
    amount = np.array([float(r["amount"]) for r in tr])
    scale = np.array([float(r.get("value_kept") or 1.0) for r in tr])
    bridge = np.array([int(r.get("bridge") or 0) for r in tr], dtype=bool)
    seeds, endpoints = [], []
    path = os.path.join(folder, "labels.csv")
    if os.path.exists(path):
        with open(path) as f:
            for r in csv.DictReader(f):
                (seeds if r["label"] == "seed" else endpoints).append(int(r["id"]))
    eco = Ecosystem(n=len(ids), chain=chain, owner=owner, kind=kind, src=src, dst=dst, amount=amount, scale=scale,
                    bridge=bridge, seeds=np.array(seeds, dtype=np.int64), endpoints=np.array(sorted(endpoints), dtype=np.int64),
                    path_vertices=np.array(sorted(endpoints), dtype=np.int64),
                    K=int(owner.max()) + 1 if K is None else K, params=dict(source=folder))
    return eco.finalize()
