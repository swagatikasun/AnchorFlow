import time

import numpy as np

from .dag import DagList
from .meter import Meter, LAN
from .mpc import Context, Shared
from .ring import FRAC, FRAC_W, U64, encode, decode

WAN = (100e-3, 100e6)


def _phase(meter, phase):
    s = meter.summary(phase)
    out = dict(rounds=s["rounds"], bytes=s["bytes"], cpu_s=s["cpu_s"])
    for name, (lat, bw) in (("lan", LAN), ("wan", WAN)):
        out[name + "_net_s"] = meter.network_time(phase, lat, bw)
        out[name + "_s"] = out[name + "_net_s"] + s["cpu_s"]
    return out


def run(src, dst, w_fp, n, x0, tau, seed=0):
    meter = Meter()
    ctx = Context(meter, seed)
    meter.reset_clock()
    dag = DagList(ctx, n, np.asarray(src), np.asarray(dst), np.asarray(w_fp, dtype=U64))
    with meter.in_phase("init"):
        x = ctx.share_from_owner(encode(x0), "seed-input")
    acc = Shared.zeros(n)
    for _ in range(tau):
        y = dag.iterate(x)
        x = Context.trunc(ctx.rerandomize(y), FRAC_W)
        acc = acc + x
    meter._tick()
    exposure = decode(acc.s0 + acc.s1)
    return dict(exposure=exposure, dag_N=dag.N, init=_phase(meter, "init"),
                online=_phase(meter, "online"), offline=_phase(meter, "offline"),
                isort_levels=(dag.levels_s, dag.levels_d), frac=FRAC)
