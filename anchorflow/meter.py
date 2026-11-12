import time
from collections import defaultdict
from contextlib import contextmanager

LAN = (0.5e-3, 1e9)
WAN = (100e-3, 100e6)


class Meter:
    def __init__(self):
        self.records = []
        self.cpu = defaultdict(float)

    def comm(self, phase, step, rounds, total_bytes, busiest_bytes):
        self.records.append(dict(phase=phase, step=step, rounds=rounds, bytes=int(total_bytes),
                                 busiest=int(busiest_bytes)))

    @contextmanager
    def timed(self, phase):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.cpu[phase] += time.perf_counter() - t0

    def add_cpu(self, phase, seconds):
        self.cpu[phase] += seconds

    def summary(self, phase, lan=LAN, wan=WAN):
        recs = [r for r in self.records if r["phase"] == phase]
        out = dict(rounds=sum(r["rounds"] for r in recs), bytes=sum(r["bytes"] for r in recs),
                   cpu_s=self.cpu.get(phase, 0.0))
        for name, (lat, bw) in (("lan", lan), ("wan", wan)):
            out[name + "_net_s"] = sum(r["rounds"] * lat + 8.0 * r["busiest"] / bw for r in recs)
            out[name + "_s"] = out[name + "_net_s"] + out["cpu_s"]
        steps = defaultdict(int)
        for r in recs:
            steps[r["step"]] += r["bytes"]
        out["per_step"] = dict(steps)
        return out
