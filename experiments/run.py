import argparse
import json
import os
import platform
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from anchorflow import dynamic, he, paillier, protocol, variants
from anchorflow.gen import generate, plaintext_exposure, seed_vector
from anchorflow.ring128 import R
from anchorflow.graphiti import run as graphiti
from anchorflow.meter import LAN, WAN

TAU = 10
THRESHOLDS = (1e-4, 1e-5, 1e-6)

BATCHES = {
    "accuracy": [dict(n_vertices=100_000, K=k, tau=12) for k in (2, 3, 4, 7, 15)],
    "scale-a": [dict(n_vertices=v, K=4) for v in (10_000, 30_000, 100_000)],
    "scale-b": [dict(n_vertices=300_000, K=4)],
    "scale-c": [dict(n_vertices=1_000_000, K=4)],
    "scale-d": [dict(n_vertices=3_000_000, K=4, graphiti=False)],
    "parties": [dict(n_vertices=100_000, K=k) for k in (2, 4, 9, 14, 24)],
    "density": [dict(n_vertices=100_000, K=4, out_degree=d) for d in (2.0, 4.0, 8.0, 16.0)],
    "forensic": [dict(n_vertices=100_000, K=4, seed=s, forensic=True) for s in (1, 2, 3, 4, 5)],
    "uncertain": [dict(sigma=s) for s in (0.25, 0.5, 1.0, 1.5, 2.0)],
    "omission": [dict(frac=f) for f in (0.05, 0.1, 0.2, 0.4)] + [dict(frac=0.0, targeted=True)],
    "dynamic": [dict(update_tx=u) for u in (100, 1_000, 10_000)],
    "anchors-missing": [dict(missing=f) for f in (0.0, 0.05, 0.1, 0.2)],
    "anchors-wrong": [dict(wrong=f) for f in (0.05, 0.1, 0.2)],
    "mixing": [dict(phi=p) for p in (0.0, 0.25, 0.5, 0.75)],
    "hops": [dict(n_vertices=100_000, K=4, tau=t, graphiti=False) for t in (2, 5, 10, 15, 20)],
    "timeexp": [dict(windows=w) for w in (1, 4, 16, 64, 256)],
    "audit": [dict(n_vertices=20_000, K=4, tier=t, deviate=d) for t in ("H", "V") for d in (False, True)],
    "he-units-2048": [dict(bits=2048)],
    "he-units-3072": [dict(bits=3072)],
    "he-validate": [dict(bits=2048)],
}


def opening_metrics(n, m, mode, tau):
    rounds, total, busiest = protocol.opening_cost(n, m, mode)
    out = dict(rounds=rounds * (tau + 1), bytes=total * (tau + 1))
    for name, (lat, bw) in (("lan", LAN), ("wan", WAN)):
        out[name + "_net_s"] = (tau + 1) * (rounds * lat + 8.0 * busiest / bw)
    return out


def accuracy(ref, got):
    out = dict(max_abs_err=float(np.abs(ref - got).max()))
    for th in THRESHOLDS:
        out[f"disagree@{th:g}"] = int(((ref >= th) != (got >= th)).sum())
        out[f"alerts@{th:g}"] = int((ref >= th).sum())
    return out


def eco_stats(eco):
    return dict(V=int(eco.n), E=int(eco.m), C=int((eco.owner >= 0).sum()),
                nnz=he.unique_pairs(eco), nnz_own=he.unique_pairs(eco, eco.col_owner >= 0),
                nnz_pub=he.unique_pairs(eco, eco.col_owner < 0), K=int(eco.K),
                max_in_degree=int(np.bincount(eco.dst, minlength=eco.n).max()))


def cost_models(eco, n, tau):
    out = {}
    for bits in (2048, 3072):
        pub = paillier.PublicKey((1 << (bits - 1)) + 1)
        beta, _, _, s = he.params(pub, n)
        ctb = pub.ct_bytes()
        for mode in ("L1", "L2"):
            ops, meta = he.anchorflow_offline_counts(eco, n, tau, mode, s)
            out[f"anchorflow-{mode}-{bits}"] = dict(ops=dict(ops), slots=s, beta=beta,
                                                    bytes=he.anchorflow_offline_bytes(eco, n, tau, mode, s, ctb))
        ops, meta = he.egmond_counts(eco, n, tau)
        init_b, iter_b, final_b = he.egmond_bytes(eco, n, tau, ctb)
        out[f"egmond-{bits}"] = dict(ops=dict(ops), cross_pairs=meta["cross_pairs"], bytes_initial=init_b,
                                     bytes_iterations=iter_b, bytes_final=final_b, rounds=tau + 2)
    return out


def run_protocol_config(cfg, seed):
    cfg = dict(cfg)
    tau = cfg.pop("tau", TAU)
    use_graphiti = cfg.pop("graphiti", True)
    forensic = cfg.pop("forensic", False)
    gseed = cfg.pop("seed", seed)
    t0 = time.perf_counter()
    eco = generate(seed=gseed, **cfg)
    gen_s = time.perf_counter() - t0
    n = eco.K + 1
    row = dict(config=cfg, tau=tau, seed=gseed, n=n, gen_s=gen_s, stats=eco_stats(eco))
    ref = plaintext_exposure(eco, tau)
    setup = protocol.Setup(eco)
    t0 = time.perf_counter()
    res = protocol.run(eco, tau, debias=True, seed=seed, setup=setup)
    row["anchorflow_wall_s"] = time.perf_counter() - t0
    on = res["meter"].summary("online")
    row["anchorflow"] = dict(cpu_crit_s=on["cpu_s"], cpu_total_s=res["meter"].cpu.get("online_total", 0.0),
                             all=opening_metrics(n, eco.n, "all", tau),
                             relay=opening_metrics(n, eco.n, "relay", tau),
                             output=res["meter"].summary("output"), **accuracy(ref, res["exposure"]))
    t0 = time.perf_counter()
    resv = protocol.run(eco, tau, debias=True, seed=seed, setup=setup, tier="V")
    onv = resv["meter"].summary("online")
    row["anchorflow_v"] = dict(wall_s=time.perf_counter() - t0, cpu_crit_s=onv["cpu_s"], bytes=onv["bytes"],
                               rounds=onv["rounds"], lan_net_s=onv["lan_net_s"], wan_net_s=onv["wan_net_s"],
                               **accuracy(ref, resv["exposure"]))
    del resv
    if cfg.get("n_vertices", 0) <= 300_000 and not forensic:
        biased = protocol.run(eco, tau, debias=False, seed=seed, setup=setup)
        row["anchorflow_no_offset"] = accuracy(ref, biased["exposure"])
    row["cost_models"] = cost_models(eco, n, tau)
    if use_graphiti and not forensic:
        t0 = time.perf_counter()
        g = graphiti.run(eco.src, eco.dst, eco.w_fp, eco.n, seed_vector(eco), tau, seed=seed)
        row["graphiti_wall_s"] = time.perf_counter() - t0
        row["graphiti"] = dict(dag_N=g["dag_N"], init=g["init"], online=g["online"], offline=g["offline"],
                               isort_levels=g["isort_levels"], **accuracy(ref, g["exposure"]))
    if forensic:
        row["forensic"] = forensic_metrics(eco)
    return row


def reachability(eco, tau):
    import scipy.sparse as sp
    A = sp.csr_matrix((np.ones(eco.m), (eco.dst, eco.src)), shape=(eco.n, eco.n))
    seen = np.zeros(eco.n, dtype=bool)
    front = np.zeros(eco.n)
    front[eco.seeds] = 1.0
    for _ in range(tau):
        front = (A @ front > 0).astype(np.float64)
        seen |= front > 0
    return seen


def forensic_metrics(eco):
    out = dict(endpoints=int(len(eco.endpoints)), by_tau={})
    reach = reachability(eco, 12)
    s12 = plaintext_exposure(eco, 12)
    out["poison"] = dict(reachable=int(reach.sum()),
                         **{f"below_{th:g}": int((reach & (s12 < th)).sum()) for th in (1e-4, 1e-6, 1e-8)})
    pub_cols = eco.col_owner < 0
    for tau in (4, 6, 8, 10, 12, 14):
        full = plaintext_exposure(eco, tau)
        public = plaintext_exposure(eco, tau, cols=pub_cols)
        rec = {}
        for th in (1e-3, 1e-4, 1e-5):
            rec[f"{th:g}"] = dict(
                recall_full=float((full[eco.endpoints] >= th).mean()),
                recall_public=float((public[eco.endpoints] >= th).mean()),
                flagged_full=int((full >= th).sum()), flagged_public=int((public >= th).sum()),
                flagged_custodial_full=int(((full >= th) & (eco.owner >= 0)).sum()),
                flagged_custodial_public=int(((public >= th) & (eco.owner >= 0)).sum()))
        out["by_tau"][str(tau)] = rec
    return out


FORENSIC_BASE = dict(n_vertices=100_000, K=4)


def recall_stats(eco, s, ref, th=1e-4):
    flagged, flagged_ref = s >= th, ref >= th
    return dict(recall=float((s[eco.endpoints] >= th).mean()), flagged=int(flagged.sum()),
                extra=int((flagged & ~flagged_ref).sum()), missed=int((flagged_ref & ~flagged).sum()))


def run_uncertain(cfg, seeds):
    out = dict(sigma=cfg["sigma"], runs=[])
    for seed in seeds:
        eco = generate(seed=seed, **FORENSIC_BASE)
        ref = plaintext_exposure(eco, 12)
        row = dict(seed=seed, bridges=int(eco.bridge.sum()))
        for mode in ("hard", "soft"):
            v = variants.uncertain_matching(eco, mode, cfg["sigma"], seed=seed)
            row[mode] = recall_stats(eco, plaintext_exposure(v, 12), ref)
        out["runs"].append(row)
    return out


def run_omission(cfg, seeds):
    out = dict(cfg, runs=[])
    for seed in seeds:
        eco = generate(seed=seed, **FORENSIC_BASE)
        ref = plaintext_exposure(eco, 12)
        v = variants.omit_custodial(eco, cfg["frac"], targeted=cfg.get("targeted", False), seed=seed)
        out["runs"].append(dict(seed=seed, dropped=int(eco.m - v.m), **recall_stats(eco, plaintext_exposure(v, 12), ref)))
    return out


def run_anchor_failure(cfg, seeds):
    out = dict(cfg, runs=[])
    for seed in seeds:
        eco = generate(seed=seed, **FORENSIC_BASE)
        oracle = plaintext_exposure(eco, 12)
        if "missing" in cfg:
            v = variants.missing_anchors(eco, cfg["missing"], seed=seed)
        else:
            v = variants.wrong_bridges(eco, cfg["wrong"], seed=seed)
        out["runs"].append(dict(seed=seed, anchor_links=int(variants.anchor_links(eco).sum()),
                                bridges=int(eco.bridge.sum()),
                                metrics=variants.compare(eco, plaintext_exposure(v, 12), oracle)))
    return out


def run_mixing(cfg, seeds):
    out = dict(cfg, runs=[])
    for seed in seeds:
        eco = variants.mix_paths(generate(seed=seed, **FORENSIC_BASE), cfg["phi"], seed=seed)
        full = plaintext_exposure(eco, 12)
        public = plaintext_exposure(eco, 12, cols=eco.col_owner < 0)
        out["runs"].append(dict(seed=seed, full=variants.compare(eco, full, full),
                                public=variants.compare(eco, public, full)))
    return out


def run_timeexp(cfg, seeds):
    out = dict(cfg, runs=[])
    for seed in seeds:
        eco = generate(seed=seed, **FORENSIC_BASE)
        out["runs"].append(dict(seed=seed, V=int(eco.n), E=int(eco.m),
                                **variants.time_expanded_size(eco, cfg["windows"], seed=seed)))
    return out


def run_dynamic(cfg, seed):
    tau = TAU
    eco = generate(seed=seed, **FORENSIC_BASE)
    new = dynamic.apply_update(eco, cfg["update_tx"], seed=seed)
    n = new.K + 1
    make, changed = dynamic.refreshed_dealer(eco, new, np.random.default_rng([seed, 5]))
    res = protocol.run(new, tau, correlations=make, seed=seed)
    ref = plaintext_exposure(new, tau)
    row = dict(update_tx=cfg["update_tx"], V=int(new.n), E=int(new.m), n=n,
               max_abs_err=float(np.abs(res["exposure"] - ref).max()))
    for bits in (2048,):
        pub = paillier.PublicKey((1 << (bits - 1)) + 1)
        _, _, _, s = he.params(pub, n)
        ops, info = dynamic.refresh_counts(eco, new, n, s, tau=tau)
        full, _ = he.anchorflow_offline_counts(new, n, tau, "L1", s)
        row[f"refresh-{bits}"] = dict(ops=dict(ops), **info)
        row[f"full-L1-{bits}"] = dict(ops=dict(full))
    t0 = time.perf_counter()
    g = graphiti.run(new.src, new.dst, new.w_fp, new.n, seed_vector(new), 1, seed=seed)
    row["graphiti_reinit"] = dict(init=g["init"], wall_s=time.perf_counter() - t0)
    return row


def run_audit(cfg, seed):
    eco = generate(n_vertices=cfg["n_vertices"], K=cfg["K"], seed=seed)
    tau, tier = 6, cfg["tier"]
    n = eco.K + 1
    rng = np.random.default_rng([seed, 77])
    deviations = {}
    if cfg["deviate"]:
        for _ in range(5):
            t, j = int(rng.integers(0, tau)), int(rng.integers(0, n))
            v = np.zeros(eco.n, dtype=np.int64)
            v[int(rng.integers(0, eco.n))] = 1 << 36
            deviations[(t, j)] = R.from_int64(v)
    t0 = time.perf_counter()
    r = protocol.run(eco, tau, seed=seed, tier=tier, record=True, deviations=deviations)
    run_s = time.perf_counter() - t0
    parties = [protocol.Party(i, seed, eco.n) for i in range(n)]
    received = [[p.masks(protocol.K_BITS + protocol.SIGMA)[1] for p in parties] for _ in range(tau + 1)] if tier == "V" else None
    flagged = {}
    t0 = time.perf_counter()
    for j in range(n):
        corr = [c[j] for c in r["corr_shares"]] if tier == "H" else None
        found = protocol.audit(eco, tau, j, seed, r["transcript"], r["commitments"], corr_shares=corr, tier=tier,
                               received=received)
        flagged[j] = sorted({t for kind, t in found if kind == "message"})
    audit_s = time.perf_counter() - t0
    expected = {}
    for (t, j) in deviations:
        expected.setdefault(j, set()).add(t)
    detected = all(set(flagged.get(j, [])) >= ts for j, ts in expected.items())
    false_flags = sum(len(v) for j, v in flagged.items() if j not in expected)
    return dict(cfg, deviations=[list(k) for k in deviations], flagged={str(k): v for k, v in flagged.items()},
                detected=bool(detected), false_flags=int(false_flags), run_s=run_s, audit_s=audit_s,
                transcript_hashes=sum(len(x) for x in r["transcript"]["open"]))


def run_he_units(cfg):
    res = he.unit_costs(bits=cfg["bits"], n=5)
    res["machine"] = platform.processor() or platform.machine()
    return res


def run_he_validate(cfg):
    bits = cfg["bits"]
    units = he.unit_costs(bits=bits, n=4, trials=10)["unit_s"]
    rows = {}
    eco = generate(n_vertices=300, K=3, seed=11, n_seeds=2)
    n = eco.K + 1
    key = paillier.keygen(bits, n)
    for mode in ("L1", "L2"):
        infos = []

        def make(rp):
            sh, info = he.offline_correlation(eco, key, [r.to_ints() for r in rp], mode=mode)
            infos.append(info)
            return sh

        t0 = time.perf_counter()
        res = protocol.run(eco, 1, correlations=make, seed=5)
        wall = time.perf_counter() - t0
        ops, _ = he.anchorflow_offline_counts(eco, n, 1, mode, infos[0]["slots"])
        rows[f"anchorflow-{mode}"] = dict(V=eco.n, C=int((eco.owner >= 0).sum()), measured_s=sum(infos[0]["seconds"].values()),
                                          predicted_s=he.cost_from_counts(ops, units), wall_s=wall,
                                          ops_match=infos[0]["ops"] == dict(ops),
                                          max_abs_err=float(np.abs(res["exposure"] - plaintext_exposure(eco, 1)).max()))
    eco = generate(n_vertices=120, K=3, seed=12, n_seeds=2)
    tau = 2
    x0 = np.round(seed_vector(eco) * (1 << protocol.F_X)).astype(np.int64)
    t0 = time.perf_counter()
    out, info = he.egmond_run(eco, key, tau, x0)
    ops, _ = he.egmond_counts(eco, n, tau)
    got = np.array(out, dtype=float) / 2.0 ** (protocol.F_X + 20 * tau)
    rows["egmond"] = dict(V=eco.n, E=eco.m, measured_s=sum(info["seconds"].values()),
                          predicted_s=he.cost_from_counts(ops, units), wall_s=time.perf_counter() - t0,
                          ops_match=info["ops"] == dict(ops),
                          max_abs_err=float(np.abs(got - plaintext_exposure(eco, tau)).max()))
    return dict(bits=bits, units=units, checks=rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("batch", choices=sorted(BATCHES))
    ap.add_argument("--seeds", type=int, nargs="+", default=[1])
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "results"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"{args.batch}.jsonl")
    with open(path, "a") as f:
        for cfg in BATCHES[args.batch]:
            for seed in args.seeds:
                t0 = time.perf_counter()
                print(f"[{args.batch}] {cfg} seed={seed}", flush=True)
                if args.batch.startswith("he-units"):
                    row = run_he_units(cfg)
                elif args.batch == "uncertain":
                    row = run_uncertain(cfg, [1, 2, 3])
                elif args.batch == "omission":
                    row = run_omission(cfg, [1, 2, 3])
                elif args.batch == "dynamic":
                    row = run_dynamic(cfg, seed)
                elif args.batch == "audit":
                    row = run_audit(cfg, seed)
                elif args.batch.startswith("anchors"):
                    row = run_anchor_failure(cfg, [1, 2, 3])
                elif args.batch == "mixing":
                    row = run_mixing(cfg, [1, 2, 3])
                elif args.batch == "timeexp":
                    row = run_timeexp(cfg, [1, 2, 3])
                elif args.batch == "he-validate":
                    row = run_he_validate(cfg)
                else:
                    row = run_protocol_config(cfg, seed)
                row["batch"] = args.batch
                row["elapsed_s"] = time.perf_counter() - t0
                f.write(json.dumps(row) + "\n")
                f.flush()
                print(f"  done in {row['elapsed_s']:.1f}s", flush=True)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
