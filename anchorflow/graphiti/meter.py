import time
from collections import defaultdict
from contextlib import contextmanager

LINKS = ("S0>S1", "S1>S0", "H>S0", "H>S1", "S>O", "O>S")


class Meter:
    def __init__(self):
        self.stack = ["online"]
        self.records = []
        self.cpu = defaultdict(float)
        self.mark = time.perf_counter()
        self.lane = None

    @property
    def phase(self):
        return self.stack[-1]

    def _tick(self):
        now = time.perf_counter()
        self.cpu[self.stack[-1]] += now - self.mark
        self.mark = now

    def comm(self, step, rounds, busiest=None, **links):
        rec = {"phase": self.phase, "step": step, "rounds": rounds}
        if busiest is not None:
            rec["busiest"] = int(busiest)
        if self.lane is not None and self.phase == "online":
            rec["lane"] = self.lane
        for k, v in links.items():
            rec[k.replace("_", ">")] = int(v)
        self.records.append(rec)

    @contextmanager
    def in_phase(self, phase):
        self._tick()
        self.stack.append(phase)
        try:
            yield
        finally:
            self._tick()
            self.stack.pop()

    def reset_clock(self):
        self.mark = time.perf_counter()

    def summary(self, phase):
        rounds = 0
        total = 0
        per_link = defaultdict(int)
        steps = defaultdict(int)
        for r in self.records:
            if r["phase"] != phase:
                continue
            rounds += r["rounds"]
            for k in LINKS:
                if k in r:
                    per_link[k] += r[k]
                    total += r[k]
                    steps[r["step"]] += r[k]
        return {"rounds": rounds, "bytes": total, "per_link": dict(per_link),
                "per_step": dict(steps), "cpu_s": self.cpu.get(phase, 0.0)}

    def _critical(self, phase, cost):
        total = 0.0
        lanes = defaultdict(lambda: defaultdict(float))
        for r in self.records:
            if r["phase"] != phase:
                continue
            c = cost(r)
            if "lane" in r:
                lanes[r["lane"][0]][r["lane"][1]] += c
            else:
                total += c
        return total + sum(max(v.values()) for v in lanes.values())

    def network_time(self, phase, latency_s, bandwidth_bps):
        def cost(r):
            busiest = r["busiest"] if "busiest" in r else max([r.get(k, 0) for k in LINKS] + [0])
            return r["rounds"] * latency_s + 8.0 * busiest / bandwidth_bps
        return self._critical(phase, cost)

    def critical_rounds(self, phase):
        return int(self._critical(phase, lambda r: r["rounds"]))


LAN = (0.5e-3, 1e9)
WAN = (40e-3, 100e6)
