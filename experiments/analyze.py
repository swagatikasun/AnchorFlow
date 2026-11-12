import json
import os
import sys
from collections import defaultdict

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(__file__), "..")
RES = os.path.join(ROOT, "results")
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "paper")
FIG = os.path.join(OUT, "figs")
TAB = os.path.join(OUT, "tables")
os.makedirs(FIG, exist_ok=True)
os.makedirs(TAB, exist_ok=True)

plt.rcParams.update({"font.family": "serif", "font.size": 8.5, "axes.linewidth": 0.6,
                     "lines.linewidth": 1.0, "lines.markersize": 3.5, "legend.fontsize": 7,
                     "legend.frameon": False, "figure.dpi": 200})
WAN_LAT, WAN_BW = 100e-3, 100e6
LAN_LAT, LAN_BW = 0.5e-3, 1e9
OFFLINE_ROUNDS_PER_ITER = 4


def load(name):
    p = os.path.join(RES, name + ".jsonl")
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []


def units(bits):
    rows = load(f"he-units-{bits}")
    return rows[-1] if rows else None


U = {b: units(b) for b in (2048, 3072)}


def he_seconds(ops, unit):
    return sum(unit["unit_s"][k] * v for k, v in ops.items())


def af_offline(row, mode="L1", bits=2048):
    cm = row["cost_models"][f"anchorflow-{mode}-{bits}"]
    cpu = he_seconds(cm["ops"], U[bits])
    n = row["n"]
    net = OFFLINE_ROUNDS_PER_ITER * row["tau"] * WAN_LAT + 8.0 * cm["bytes"] / n / WAN_BW
    return dict(cpu_total_h=cpu / 3600, per_party_h=(cpu / n + net) / 3600, bytes=cm["bytes"])


def egmond_online(row, bits=2048):
    cm = row["cost_models"][f"egmond-{bits}"]
    cpu = he_seconds(cm["ops"], U[bits])
    n = row["n"]
    total_bytes = cm["bytes_initial"] + cm["bytes_iterations"] + cm["bytes_final"]
    net_wan = cm["rounds"] * WAN_LAT + 8.0 * total_bytes / n / WAN_BW
    return dict(cpu_total_h=cpu / 3600, wan_s=cpu / n + net_wan, bytes=total_bytes)


def af_online(row, mode, net):
    a = row["anchorflow"]
    return a["cpu_crit_s"] + a[mode][f"{net}_net_s"]


def af_online_v(row, net):
    a = row["anchorflow_v"]
    return a["cpu_crit_s"] + a[f"{net}_net_s"]


scale = sorted(load("scale-a") + load("scale-b") + load("scale-c") + load("scale-d"), key=lambda r: r["stats"]["V"])
parties = sorted(load("parties"), key=lambda r: r["n"])
density = sorted(load("density"), key=lambda r: r["stats"]["E"])
accuracy = sorted(load("accuracy"), key=lambda r: r["n"])
forensic = load("forensic")
validate = load("he-validate")
uncertain = sorted(load("uncertain"), key=lambda r: r["sigma"])
omission = load("omission")
dyn = sorted(load("dynamic"), key=lambda r: r["update_tx"])
audit_rows = load("audit")
anch_miss = sorted(load("anchors-missing"), key=lambda r: r["missing"])
anch_wrong = sorted(load("anchors-wrong"), key=lambda r: r["wrong"])
mixing = sorted(load("mixing"), key=lambda r: r["phi"])
hops = sorted(load("hops"), key=lambda r: r["tau"])
timeexp = sorted(load("timeexp"), key=lambda r: r["windows"])
N = {}


def fmt(v, d=2):
    if v is None:
        return "--"
    if v >= 1000:
        return f"{v:,.0f}".replace(",", "{,}")
    if v >= 100:
        return f"{v:.0f}"
    if v >= 10:
        return f"{v:.1f}"
    return f"{v:.{d}f}"


def sci(v):
    if v == 0:
        return "0"
    e = int(np.floor(np.log10(abs(v))))
    m = v / 10 ** e
    return f"{m:.1f}\\times 10^{{{e}}}"


def vlabel(V):
    e = int(np.floor(np.log10(V)))
    m = V / 10 ** e
    return f"$10^{{{e}}}$" if round(m, 1) == 1.0 else f"${m:.0f}\\cdot10^{{{e}}}$"


def fig_scale():
    if not scale:
        return
    V = np.array([r["stats"]["V"] for r in scale])
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.55))
    ax = axes[0]
    ax.plot(V, [r["anchorflow"]["relay"]["bytes"] / 1e6 for r in scale], "o-", color="black", label="AnchorFlow, relay")
    ax.plot(V, [r["anchorflow"]["all"]["bytes"] / 1e6 for r in scale], "o--", color="black", markerfacecolor="white", label="AnchorFlow, all-to-all")
    g = [r for r in scale if "graphiti" in r]
    ax.plot([r["stats"]["V"] for r in g], [r["graphiti"]["online"]["bytes"] / 1e6 for r in g], "s-", color="0.5", label="Graphiti-style (2+1), online")
    ax.plot([r["stats"]["V"] for r in g], [r["graphiti"]["init"]["bytes"] / 1e6 for r in g], "s:", color="0.5", markerfacecolor="white", label="Graphiti-style (2+1), init")
    ax.set_ylabel("Communication (MB)")
    ax = axes[1]
    ax.plot(V, [af_online(r, "all", "wan") for r in scale], "o-", color="black", label="AnchorFlow")
    ax.plot([r["stats"]["V"] for r in g], [r["graphiti"]["online"]["wan_s"] for r in g], "s-", color="0.5", label="Graphiti-style (2+1)")
    if U[2048]:
        ax.plot(V, [egmond_online(r)["wan_s"] for r in scale], "^--", color="0.3", markerfacecolor="white", label="HE propagation")
    ax.set_ylabel("Online WAN time per query (s)")
    ax = axes[2]
    ax.plot([r["stats"]["V"] for r in g], [r["graphiti"]["init"]["wan_s"] / 3600 for r in g], "s:", color="0.5", markerfacecolor="white", label="Graphiti-style init, WAN")
    if U[2048]:
        ax.plot(V, [af_offline(r, "L1")["per_party_h"] for r in scale], "o-", color="black", label="AnchorFlow offline, L1")
        ax.plot(V, [af_offline(r, "L2")["per_party_h"] for r in scale], "o--", color="black", markerfacecolor="white", label="AnchorFlow offline, L2")
    ax.set_ylabel("Preprocessing per party (h)")
    for ax, t in zip(axes, "abc"):
        ax.set_xscale("log")
        ax.set_yscale("log")
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, hi * (hi / lo) ** 0.9)
        ax.legend(loc="upper left", fontsize=6.2, handlelength=2.2)
        ax.set_xlabel(f"({t}) Vertices $|V|$")
        ax.grid(True, which="major", linewidth=0.3, color="0.85")
    fig.tight_layout(w_pad=0.8)
    fig.savefig(os.path.join(FIG, "scale.pdf"))
    plt.close(fig)


def fig_parties_density():
    if not parties or not density:
        return
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.3))
    n = [r["n"] for r in parties]
    ax = axes[0]
    ax.plot(n, [r["anchorflow"]["all"]["bytes"] / 1e6 for r in parties], "o--", color="black", markerfacecolor="white", label="all-to-all")
    ax.plot(n, [r["anchorflow"]["relay"]["bytes"] / 1e6 for r in parties], "o-", color="black", label="relay")
    ax.set_ylabel("Online comm. per query (MB)")
    ax.set_xlabel("(a) Parties $n$")
    ax.legend()
    ax = axes[1]
    ax.plot(n, [af_online(r, "all", "wan") for r in parties], "o--", color="black", markerfacecolor="white", label="all-to-all")
    ax.plot(n, [af_online(r, "relay", "wan") for r in parties], "o-", color="black", label="relay")
    ax.plot(n, [af_online(r, "all", "lan") for r in parties], "^:", color="0.4", label="all-to-all, LAN")
    ax.set_ylabel("Online time per query (s)")
    ax.set_xlabel("(b) Parties $n$")
    ax.legend()
    ax = axes[2]
    d = [r["stats"]["E"] / r["stats"]["V"] for r in density]
    ax.plot(d, [r["anchorflow"]["relay"]["bytes"] / 1e6 for r in density], "o-", color="black", label="AnchorFlow, relay")
    ax.plot(d, [r["graphiti"]["online"]["bytes"] / 1e6 for r in density], "s-", color="0.5", label="Graphiti-style (2+1), online")
    ax.set_ylim(0, None)
    ax.set_ylabel("Online comm. per query (MB)")
    ax.set_xlabel("(c) Average degree $|E|/|V|$")
    ax.legend(loc="upper left", fontsize=6.5)
    for ax in axes:
        ax.grid(True, which="major", linewidth=0.3, color="0.85")
    fig.tight_layout(w_pad=0.8)
    fig.savefig(os.path.join(FIG, "parties.pdf"))
    plt.close(fig)


def fig_accuracy():
    if not accuracy:
        return
    fig, axes = plt.subplots(1, 2, figsize=(3.5, 1.9))
    n = [r["n"] for r in accuracy]
    ax = axes[0]
    ax.plot(n, [r["anchorflow"]["max_abs_err"] for r in accuracy], "o-", color="black", label="with offset $h$")
    ax.plot(n, [r["anchorflow_no_offset"]["max_abs_err"] for r in accuracy], "s--", color="0.5", markerfacecolor="white", label="without $h$")
    ax.set_yscale("log")
    ax.set_ylabel("Max. exposure error")
    ax.set_xlabel("(a) Parties $n$")
    ax.legend(loc="center right", fontsize=6.5)
    ax = axes[1]
    w = 0.38
    x = np.arange(len(n))
    ax.bar(x - w / 2, [r["anchorflow"]["disagree@1e-06"] for r in accuracy], w, color="black", label="with $h$")
    ax.bar(x + w / 2, [r["anchorflow_no_offset"]["disagree@1e-06"] for r in accuracy], w, color="0.7", label="without $h$")
    ax.set_xticks(x)
    ax.set_xticklabels(n)
    ax.set_ylabel(r"Wrong alerts, $\theta=10^{-6}$")
    ax.set_xlabel("(b) Parties $n$")
    ax.legend(fontsize=6.5)
    for ax in axes:
        ax.grid(True, axis="y", linewidth=0.3, color="0.85")
    fig.tight_layout(w_pad=0.5)
    fig.savefig(os.path.join(FIG, "accuracy.pdf"))
    plt.close(fig)


def forensic_curves(th="0.0001"):
    taus = sorted({int(t) for r in forensic for t in r["forensic"]["by_tau"]})
    full = [np.mean([r["forensic"]["by_tau"][str(t)][th]["recall_full"] for r in forensic]) for t in taus]
    pub = [np.mean([r["forensic"]["by_tau"][str(t)][th]["recall_public"] for r in forensic]) for t in taus]
    return taus, full, pub


def fig_forensic():
    if not forensic:
        return
    fig, ax = plt.subplots(figsize=(3.5, 1.9))
    for th, style in (("0.0001", "-"), ("1e-05", ":")):
        taus, full, pub = forensic_curves(th)
        ax.plot(taus, full, "o" + style, color="black", label=f"AnchorFlow, $\\theta={sci(float(th))}$".replace("\\times", "\\!\\times\\!"))
        ax.plot(taus, pub, "s" + style, color="0.55", markerfacecolor="white", label=f"Public only, $\\theta={sci(float(th))}$".replace("\\times", "\\!\\times\\!"))
    ax.set_xlabel(r"Hops $\tau$")
    ax.set_ylabel("Endpoint recall")
    ax.set_ylim(-0.03, 1.05)
    ax.grid(True, linewidth=0.3, color="0.85")
    ax.legend(fontsize=6, loc="center right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "forensic.pdf"))
    plt.close(fig)


def table_scale():
    lines = []
    for r in scale:
        a = r["anchorflow"]
        g = r.get("graphiti")
        cells = [vlabel(r["stats"]["V"]), fmt(r["stats"]["E"] / 1e6, 2), fmt(a["cpu_crit_s"]),
                 fmt(a["relay"]["bytes"] / 1e6), fmt(af_online(r, "relay", "lan")), fmt(af_online(r, "all", "wan")),
                 fmt(af_online_v(r, "wan")),
                 fmt(g["online"]["bytes"] / 1e6) if g else "--", fmt(g["online"]["wan_s"]) if g else "--",
                 fmt(g["init"]["bytes"] / 1e9) if g else "--", fmt(g["init"]["wan_s"] / 60) if g else "--"]
        lines.append(" & ".join(cells) + r" \\")
    open(os.path.join(TAB, "scale.tex"), "w").write("\n".join(lines) + "\n")


def table_he_units():
    names = [("encrypt", "Encryption (with fresh $r^N$)"), ("partial_dec", "Partial decryption"),
             ("exp_slot", r"Slot shift $c^{2^\beta}$"), ("exp_scale", r"Scale shift $c^{2^{100}}$"),
             ("exp_w", "Weight exponent (20 bits)"), ("mul", "Ciphertext multiplication")]
    lines = []
    for key, label in names:
        cells = [label] + [fmt(U[b]["unit_s"][key] * 1e3, 2) if U[b] else "--" for b in (2048, 3072)]
        lines.append(" & ".join(cells) + r" \\")
    open(os.path.join(TAB, "he_units.tex"), "w").write("\n".join(lines) + "\n")


def table_offline():
    lines = []
    for r in scale:
        V = r["stats"]["V"]
        if V not in {min(scale, key=lambda q: abs(q["stats"]["V"] - t))["stats"]["V"] for t in (1e4, 1e5, 1e6)}:
            continue
        l1, l2 = af_offline(r, "L1"), af_offline(r, "L2")
        eg = egmond_online(r)
        cells = [vlabel(V), fmt(r["stats"]["C"] / 1e3, 1), fmt(l1["cpu_total_h"]), fmt(l1["per_party_h"]), fmt(l1["bytes"] / 1e9),
                 fmt(l2["per_party_h"]), fmt(l2["bytes"] / 1e9), fmt(eg["cpu_total_h"]), fmt(eg["wan_s"] / 3600),
                 fmt(af_online(r, "all", "wan"))]
        lines.append(" & ".join(cells) + r" \\")
    open(os.path.join(TAB, "offline.tex"), "w").write("\n".join(lines) + "\n")


def table_validate():
    if not validate:
        return
    v = validate[-1]["checks"]
    lines = []
    for key, label in (("anchorflow-L1", "\\textsf{AnchorFlow}, L1"), ("anchorflow-L2", "\\textsf{AnchorFlow}, L2"),
                       ("egmond", "HE propagation, $\\tau=2$")):
        c = v[key]
        lines.append(" & ".join([label, str(c["V"]), fmt(c["measured_s"]), fmt(c["predicted_s"]),
                                 f"${sci(c['max_abs_err'])}$" if c["max_abs_err"] else "0"]) + r" \\")
    open(os.path.join(TAB, "validate.tex"), "w").write("\n".join(lines) + "\n")


def fig_uncertain():
    if not uncertain:
        return
    sig = [r["sigma"] for r in uncertain]
    fig, axes = plt.subplots(1, 2, figsize=(3.5, 1.9))
    for mode, style, label in (("hard", "s--", "best match only"), ("soft", "o-", "confidence-weighted")):
        axes[0].plot(sig, [np.mean([x[mode]["recall"] for x in r["runs"]]) for r in uncertain], style, color="black" if mode == "soft" else "0.5",
                     markerfacecolor="white" if mode == "hard" else "black", label=label)
        axes[1].plot(sig, [np.mean([x[mode]["extra"] for x in r["runs"]]) for r in uncertain], style, color="black" if mode == "soft" else "0.5",
                     markerfacecolor="white" if mode == "hard" else "black", label=label)
    axes[0].set_ylabel("Endpoint recall")
    axes[0].set_xlabel(r"(a) Matching noise $\sigma$")
    axes[0].set_ylim(0.4, 1.03)
    axes[0].legend(fontsize=5.8, loc="lower left", handlelength=1.8)
    axes[1].set_ylabel("Extra flags")
    axes[1].set_xlabel(r"(b) Matching noise $\sigma$")
    for ax in axes:
        ax.grid(True, linewidth=0.3, color="0.85")
    fig.tight_layout(w_pad=1.2)
    fig.savefig(os.path.join(FIG, "uncertain.pdf"))
    plt.close(fig)


def table_omission():
    if not omission:
        return
    lines = []
    for r in sorted(omission, key=lambda r: (r.get("targeted", False), r["frac"])):
        label = "planted links only" if r.get("targeted") else f"{100 * r['frac']:.0f}\\% at random"
        dropped = int(round(np.mean([x["dropped"] for x in r["runs"]])))
        lines.append(" & ".join([label, fmt(dropped) if dropped >= 1000 else str(dropped),
                                 fmt(100 * np.mean([x["recall"] for x in r["runs"]]), 0),
                                 fmt(np.mean([x["missed"] for x in r["runs"]]), 0),
                                 fmt(np.mean([x["extra"] for x in r["runs"]]), 0)]) + r" \\")
    open(os.path.join(TAB, "omission.tex"), "w").write("\n".join(lines) + "\n")


def table_dynamic():
    if not dyn or not U[2048]:
        return
    lines = []
    for r in dyn:
        n = r["n"]
        ref = he_seconds(r["refresh-2048"]["ops"], U[2048]) / n / 3600
        full = he_seconds(r["full-L1-2048"]["ops"], U[2048]) / n / 3600
        g = r["graphiti_reinit"]["init"]
        lines.append(" & ".join([fmt(r["update_tx"], 0), fmt(r["refresh-2048"]["changed_cols"]) if r["refresh-2048"]["changed_cols"] >= 1000 else str(r["refresh-2048"]["changed_cols"]), fmt(ref), fmt(full),
                                 fmt(100 * ref / full, 0) + "\\%", fmt(g["wan_s"] / 60), f"${sci(r['max_abs_err'])}$"]) + r" \\")
    open(os.path.join(TAB, "dynamic.tex"), "w").write("\n".join(lines) + "\n")


def mean_metric(rows, key, th="0.0001", part="metrics"):
    return float(np.mean([x[part][th][key] for x in rows]))


def table_comm():
    big = max(scale, key=lambda r: r["stats"]["V"] if r["stats"]["V"] <= 1.5e6 else 0) if scale else None
    p25 = parties[-1] if parties else None
    if not big or not p25:
        return
    def mb(r, mode):
        return fmt(r["anchorflow"][mode]["bytes"] / 1e6)
    def busiest_mb(r, mode):
        n, V, tau = r["n"], r["stats"]["V"], r["tau"]
        b = (n - 1) * V * 16 if mode == "all" else 2 * (n - 1) * V * 16
        return fmt(b * (tau + 1) / 1e6)
    lines = [
        " & ".join([r"All-to-all", "1", r"$n(n-1)|V|$", r"$(n-1)|V|$", r"$(n-1)|V|$", mb(big, "all"), busiest_mb(big, "all"), mb(p25, "all"), busiest_mb(p25, "all")]) + r" \\",
        " & ".join([r"Relay", "2", r"$2(n-1)|V|$", r"$|V|$; relay $(n-1)|V|$", r"$2(n-1)|V|$", mb(big, "relay"), busiest_mb(big, "relay"), mb(p25, "relay"), busiest_mb(p25, "relay")]) + r" \\",
        " & ".join([r"Tier V, added", "0", r"$(n-1)|C|$", r"$\le|C|$", r"$\max(|C|,(n-1)\max_k|V_k|)$",
                    fmt((big["anchorflow_v"]["bytes"] - big["anchorflow"]["all"]["bytes"]) / 1e6),
                    "--", fmt((p25["anchorflow_v"]["bytes"] - p25["anchorflow"]["all"]["bytes"]) / 1e6), "--"]) + r" \\"]
    open(os.path.join(TAB, "comm.tex"), "w").write("\n".join(lines) + "\n")
    N["CommBigV"] = fmt(big["stats"]["V"] / 1e6, 1)
    N["CommPartiesV"] = fmt(p25["stats"]["V"] / 1e3, 0)
    N["CommPartiesN"] = str(p25["n"])


def table_total():
    if not scale or not U[2048]:
        return
    lines = []
    for target in (1e5, 1e6):
        r = min((q for q in scale if "graphiti" in q), key=lambda q: abs(q["stats"]["V"] - target))
        h_q = (af_offline(r, "L1")["per_party_h"] * 3600 + af_online(r, "all", "wan")) / 3600
        v_q = af_online_v(r, "wan") / 3600
        e_q = egmond_online(r)["wan_s"] / 3600
        g_init = r["graphiti"]["init"]["wan_s"] / 3600
        g_on = r["graphiti"]["online"]["wan_s"] / 3600
        for i, Q in enumerate((1, 10, 100, 1000)):
            cells = [vlabel(r["stats"]["V"]) if i == 0 else "", str(Q), fmt(Q * h_q), fmt(Q * v_q), fmt(Q * e_q),
                     fmt(g_init + Q * g_on), fmt(Q * (g_init + g_on))]
            lines.append(" & ".join(cells) + r" \\")
        lines.append(r"\midrule")
        if target == 1e5:
            N["TotHQ"] = fmt(h_q)
            N["TotVQ"] = fmt(v_q * 3600)
            N["TotEQ"] = fmt(e_q)
            N["TotGStatic"] = fmt(g_init + 1000 * g_on)
            N["TotHThousand"] = fmt(1000 * h_q)
            N["TotEThousand"] = fmt(1000 * e_q)
            N["TotVThousand"] = fmt(1000 * v_q)
            N["BreakEvenQ"] = fmt(g_init / max(h_q - g_on, 1e-9), 1) if h_q < g_init + g_on else "none"
    lines = lines[:-1]
    open(os.path.join(TAB, "total.tex"), "w").write("\n".join(lines) + "\n")


def table_anchors():
    if not anch_miss or not anch_wrong:
        return
    lines = []
    for label, rows, key in (("Missing anchors", anch_miss, "missing"), ("Wrong matches", anch_wrong, "wrong")):
        for i, r in enumerate(rows):
            if r[key] == 0 and key == "wrong":
                continue
            lines.append(" & ".join([label if i == 0 or (key == "wrong" and i == 0) else "", f"{100 * r[key]:.0f}\\%",
                                     fmt(100 * mean_metric(r["runs"], "precision"), 1), fmt(100 * mean_metric(r["runs"], "recall"), 1),
                                     f"{100 * mean_metric(r['runs'], 'fpr'):.3f}", fmt(100 * mean_metric(r["runs"], "endpoint_recall"), 1)]) + r" \\")
        lines.append(r"\midrule")
    open(os.path.join(TAB, "anchors.tex"), "w").write("\n".join(lines[:-1]) + "\n")
    hi_m, hi_w = anch_miss[-1], anch_wrong[-1]
    N["MissHiPct"] = f"{100 * hi_m['missing']:.0f}"
    N["MissHiRecall"] = fmt(100 * mean_metric(hi_m["runs"], "recall"), 1)
    N["MissHiPrec"] = fmt(100 * mean_metric(hi_m["runs"], "precision"), 1)
    N["MissHiEnd"] = fmt(100 * mean_metric(hi_m["runs"], "endpoint_recall"), 1)
    N["WrongHiPct"] = f"{100 * hi_w['wrong']:.0f}"
    N["WrongHiRecall"] = fmt(100 * mean_metric(hi_w["runs"], "recall"), 1)
    N["WrongHiPrec"] = fmt(100 * mean_metric(hi_w["runs"], "precision"), 1)
    N["AnchorLinks"] = fmt(np.mean([x["anchor_links"] for x in hi_m["runs"]]), 0)


def table_mixing():
    if not mixing:
        return
    lines = []
    for r in mixing:
        cells = [f"{r['phi']:.2f}"]
        for th in ("0.0001", "1e-05"):
            cells += [fmt(100 * mean_metric(r["runs"], "endpoint_recall", th, "full"), 0),
                      fmt(100 * mean_metric(r["runs"], "endpoint_recall", th, "public"), 0)]
        cells += [fmt(100 * mean_metric(r["runs"], "recall", "0.0001", "public"), 1),
                  f"{100 * mean_metric(r['runs'], 'fpr', '0.0001', 'public'):.3f}"]
        lines.append(" & ".join(cells) + r" \\")
    open(os.path.join(TAB, "mixing.tex"), "w").write("\n".join(lines) + "\n")
    hi = mixing[-1]
    N["MixHiPhi"] = f"{hi['phi']:.2f}"
    N["MixHiFullFour"] = fmt(100 * mean_metric(hi["runs"], "endpoint_recall", "0.0001", "full"), 0)
    N["MixHiFullFive"] = fmt(100 * mean_metric(hi["runs"], "endpoint_recall", "1e-05", "full"), 0)
    lo = mixing[0]
    N["PubRecallVsOracle"] = fmt(100 * mean_metric(lo["runs"], "recall", "0.0001", "public"), 1)


def table_hops():
    if not hops:
        return
    lines = []
    for r in hops:
        lines.append(" & ".join([str(r["tau"]), f"${sci(r['anchorflow']['max_abs_err'])}$",
                                 f"${sci(r['anchorflow_no_offset']['max_abs_err'])}$",
                                 str(r["anchorflow_no_offset"]["disagree@1e-06"]),
                                 fmt(af_online(r, "all", "wan"))]) + r" \\")
    open(os.path.join(TAB, "hops.tex"), "w").write("\n".join(lines) + "\n")
    N["HopsMaxTau"] = str(hops[-1]["tau"])
    N["HopsMaxErr"] = f"${sci(hops[-1]['anchorflow']['max_abs_err'])}$"
    N["HopsMaxErrNoOff"] = f"${sci(hops[-1]['anchorflow_no_offset']['max_abs_err'])}$"


def table_custodians():
    if not parties or not U[2048]:
        return
    lines = []
    for r in parties:
        st = r["stats"]
        mem = (6 * 16 * st["V"] + 24 * st["nnz_pub"] + 24 * st["nnz_own"] / max(st["K"], 1) + 16 * st["V"]) / 1e6
        lines.append(" & ".join([str(r["n"]), fmt(af_online(r, "all", "wan")), fmt(af_online(r, "relay", "wan")),
                                 fmt(r["anchorflow"]["relay"]["bytes"] / 1e6), fmt(af_online_v(r, "wan")),
                                 fmt(af_offline(r, "L1")["per_party_h"]), fmt(mem)]) + r" \\")
    open(os.path.join(TAB, "custodians.tex"), "w").write("\n".join(lines) + "\n")


def table_timeexp():
    if not timeexp or not scale:
        return
    mid = min(scale, key=lambda r: abs(r["stats"]["V"] - 1e5))
    lines = []
    for r in timeexp:
        vt = np.mean([x["vertices"] for x in r["runs"]])
        v = np.mean([x["V"] for x in r["runs"]])
        ce = np.mean([x["carry_edges"] for x in r["runs"]])
        lines.append(" & ".join([str(r["windows"]), fmt(vt / 1e3, 1), fmt(ce / 1e3, 1), fmt(vt / v, 2),
                                 fmt(af_online(mid, "all", "wan") * vt / mid["stats"]["V"])]) + r" \\")
    open(os.path.join(TAB, "timeexp.tex"), "w").write("\n".join(lines) + "\n")
    r = timeexp[-1]
    N["TimeExpRatio"] = fmt(np.mean([x["vertices"] for x in r["runs"]]) / np.mean([x["V"] for x in r["runs"]]), 1)
    N["TimeExpW"] = str(r["windows"])


def numbers():
    def put(name, value):
        N[name] = value

    if scale:
        big_g = max((r for r in scale if "graphiti" in r), key=lambda r: r["stats"]["V"])
        top = max(scale, key=lambda r: r["stats"]["V"])
        put("MaxV", fmt(top["stats"]["V"] / 1e6, 1))
        put("MaxE", fmt(top["stats"]["E"] / 1e6, 1))
        put("MaxVWan", fmt(af_online(top, "all", "wan")))
        put("MaxVCpu", fmt(top["anchorflow"]["cpu_crit_s"]))
        put("GV", fmt(big_g["stats"]["V"] / 1e6, 1))
        put("GE", fmt(big_g["stats"]["E"] / 1e6, 1))
        put("GOnlineWan", fmt(big_g["graphiti"]["online"]["wan_s"]))
        put("GInitWanMin", fmt(big_g["graphiti"]["init"]["wan_s"] / 60))
        put("GInitGB", fmt(big_g["graphiti"]["init"]["bytes"] / 1e9))
        put("AFOnlineWanAtG", fmt(af_online(big_g, "all", "wan")))
        put("AFRelayMBAtG", fmt(big_g["anchorflow"]["relay"]["bytes"] / 1e6))
        put("GOnlineMBAtG", fmt(big_g["graphiti"]["online"]["bytes"] / 1e6))
        put("CommRatio", fmt(big_g["graphiti"]["online"]["bytes"] / big_g["anchorflow"]["relay"]["bytes"], 1))
        put("WanRatio", fmt(big_g["graphiti"]["online"]["wan_s"] / af_online(big_g, "all", "wan"), 1))
        put("MaxErr", f"${sci(max(r['anchorflow']['max_abs_err'] for r in scale + parties + density + accuracy))}$")
        if U[2048]:
            mid = min(scale, key=lambda r: abs(r["stats"]["V"] - 1e5))
            eg = egmond_online(mid)
            put("EgOnlineHMid", fmt(eg["wan_s"] / 3600))
            put("AFOnlineSMid", fmt(af_online(mid, "all", "wan")))
            put("AFOfflineHMid", fmt(af_offline(mid, "L1")["per_party_h"]))
            put("EgCpuHMid", fmt(eg["cpu_total_h"]))
            put("AFCpuHMid", fmt(af_offline(mid, "L1")["cpu_total_h"]))
            put("MidV", fmt(mid["stats"]["V"] / 1e3, 0))
            put("HEWorkRatio", fmt(af_offline(mid, "L1")["cpu_total_h"] / eg["cpu_total_h"], 1))
            put("SpeedupVsEg", fmt(eg["wan_s"] / af_online(mid, "all", "wan"), 0))
    allrows = scale + parties + density + accuracy + forensic
    put("Runs", str(len(allrows)))
    put("Decisions", fmt(sum(3 * r["stats"]["V"] for r in allrows) / 1e6, 1))
    put("AFWrong", str(sum(sum(v for k, v in r["anchorflow"].items() if k.startswith("disagree")) for r in allrows)))
    put("GWrong", str(sum(sum(v for k, v in r["graphiti"].items() if k.startswith("disagree")) for r in allrows if "graphiti" in r)))
    if density:
        put("DensLo", fmt(density[0]["stats"]["E"] / density[0]["stats"]["V"], 1))
        put("DensHi", fmt(density[-1]["stats"]["E"] / density[-1]["stats"]["V"], 1))
        put("GDensGrowth", fmt(density[-1]["graphiti"]["online"]["bytes"] / density[0]["graphiti"]["online"]["bytes"], 1))
        put("AFDensGrowth", fmt(density[-1]["anchorflow"]["relay"]["bytes"] / density[0]["anchorflow"]["relay"]["bytes"], 2))
    if accuracy:
        put("NoOffsetMaxErr", f"${sci(max(r['anchorflow_no_offset']['max_abs_err'] for r in accuracy))}$")
        put("NoOffsetWrong", str(max(r["anchorflow_no_offset"]["disagree@1e-06"] for r in accuracy)))
        put("OffsetMaxErrAcc", f"${sci(max(r['anchorflow']['max_abs_err'] for r in accuracy))}$")
    if forensic and "poison" in forensic[0]["forensic"]:
        po = [r["forensic"]["poison"] for r in forensic]
        N["PoisonReach"] = fmt(np.mean([p["reachable"] for p in po]), 0)
        N["PoisonBelowSix"] = fmt(100 * np.mean([p["below_1e-06"] / p["reachable"] for p in po]), 1)
        N["PoisonBelowFour"] = fmt(100 * np.mean([p["below_0.0001"] / p["reachable"] for p in po]), 1)
    if forensic:
        taus, full, pub = forensic_curves("0.0001")
        i = taus.index(12)
        put("RecallFull", fmt(100 * full[i], 0))
        put("RecallPublic", fmt(100 * pub[i], 0))
        put("FlaggedCustodial", fmt(np.mean([r["forensic"]["by_tau"]["12"]["0.0001"]["flagged_custodial_full"] for r in forensic]), 0))
    if parties:
        p16 = parties[-1]
        put("MaxParties", str(p16["n"]))
        put("WanMaxParties", fmt(af_online(p16, "all", "wan")))
        put("RelayWanMaxParties", fmt(af_online(p16, "relay", "wan")))
        put("RelayMaxPartiesMB", fmt(p16["anchorflow"]["relay"]["bytes"] / 1e6))
        put("AllMaxPartiesMB", fmt(p16["anchorflow"]["all"]["bytes"] / 1e6))
    vrows = [r for r in allrows if "anchorflow_v" in r]
    if vrows:
        put("VWrong", str(sum(sum(v for k, v in r["anchorflow_v"].items() if k.startswith("disagree")) for r in vrows)))
        put("VMaxErr", f"${sci(max(r['anchorflow_v']['max_abs_err'] for r in vrows))}$")
    if scale:
        top = max(scale, key=lambda r: r["stats"]["V"])
        put("MaxVWanV", fmt(af_online_v(top, "wan")))
        mid = min(scale, key=lambda r: abs(r["stats"]["V"] - 1e5))
        put("VWanMid", fmt(af_online_v(mid, "wan")))
        put("VExtraPct", fmt(100 * (mid["anchorflow_v"]["bytes"] / mid["anchorflow"]["all"]["bytes"] - 1), 0))
    if uncertain:
        hi = uncertain[-1]
        put("UncSigmaHi", fmt(hi["sigma"], 1))
        put("UncHardRecall", fmt(100 * np.mean([x["hard"]["recall"] for x in hi["runs"]]), 0))
        put("UncSoftRecall", fmt(100 * np.mean([x["soft"]["recall"] for x in hi["runs"]]), 0))
        put("UncHardExtra", fmt(np.mean([x["hard"]["extra"] for x in hi["runs"]]), 0))
        put("UncSoftExtra", fmt(np.mean([x["soft"]["extra"] for x in hi["runs"]]), 0))
    if omission:
        tg = [r for r in omission if r.get("targeted")]
        if tg:
            put("TargetRecall", fmt(100 * np.mean([x["recall"] for x in tg[0]["runs"]]), 0))
        r10 = [r for r in omission if not r.get("targeted") and abs(r["frac"] - 0.1) < 1e-9]
        if r10:
            put("OmitTenRecall", fmt(100 * np.mean([x["recall"] for x in r10[0]["runs"]]), 0))
    if dyn and U[2048]:
        r = dyn[-1]
        n = r["n"]
        ref = he_seconds(r["refresh-2048"]["ops"], U[2048]) / n / 3600
        full = he_seconds(r["full-L1-2048"]["ops"], U[2048]) / n / 3600
        put("DynTx", fmt(r["update_tx"], 0))
        put("DynRefreshPct", fmt(100 * ref / full, 0))
        put("DynGReinitMin", fmt(r["graphiti_reinit"]["init"]["wan_s"] / 60))
        put("DynSmallPct", fmt(100 * he_seconds(dyn[0]["refresh-2048"]["ops"], U[2048]) / he_seconds(dyn[0]["full-L1-2048"]["ops"], U[2048]), 0))
    if audit_rows:
        put("AuditRuns", str(len(audit_rows)))
        put("AuditDetected", str(sum(1 for r in audit_rows if r["deviate"] and r["detected"])))
        put("AuditDeviating", str(sum(1 for r in audit_rows if r["deviate"])))
        put("AuditFalse", str(sum(r["false_flags"] for r in audit_rows)))
        put("AuditSec", fmt(max(r["audit_s"] / (r["K"] + 1) for r in audit_rows), 2))
    if U[2048]:
        put("EncMs", fmt(U[2048]["unit_s"]["encrypt"] * 1e3, 1))
        put("PdecMs", fmt(U[2048]["unit_s"]["partial_dec"] * 1e3, 1))
        put("Backend", "pure-Python" if U[2048]["backend"] == "python" else "GMP")
    if validate:
        c = validate[-1]["checks"]
        put("ValErr", f"${sci(max(c[k]['max_abs_err'] for k in c if c[k]['max_abs_err']) )}$")
    with open(os.path.join(OUT, "numbers.tex"), "w") as f:
        for k, v in sorted(N.items()):
            f.write(f"\\newcommand{{\\num{k}}}{{{v}}}\n")
    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump(N, f, indent=1)


table_comm()
table_total()
table_anchors()
table_mixing()
table_hops()
table_custodians()
table_timeexp()
fig_scale()
fig_uncertain()
table_omission()
table_dynamic()
fig_parties_density()
fig_accuracy()
fig_forensic()
table_scale()
table_he_units()
table_offline()
table_validate()
numbers()
print("ok", json.dumps(N, indent=1))
