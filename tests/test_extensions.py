import tempfile

import numpy as np
import pytest

from anchorflow import dataio, dynamic, protocol, variants
from anchorflow.gen import generate, plaintext_exposure
from anchorflow.ring128 import R


def test_owner_visible_tier_matches_plaintext():
    eco = generate(n_vertices=4000, K=4, seed=21)
    res = protocol.run(eco, 10, seed=2, tier="V")
    assert np.abs(res["exposure"] - plaintext_exposure(eco, 10)).max() < 1e-6


def test_masked_output_delivery_gives_each_owner_its_accounts():
    eco = generate(n_vertices=3000, K=3, seed=24)
    res = protocol.run(eco, 6, seed=4)
    setup = protocol.Setup(eco)
    ref = plaintext_exposure(eco, 6)
    for k, vals in enumerate(res["outputs"]["owners"]):
        assert np.abs(vals - ref[setup.owned[k]]).max() < 1e-6
    assert np.abs(res["outputs"]["investigator"] - ref[res["outputs"]["public"]]).max() < 1e-6


def test_twenty_five_parties():
    eco = generate(n_vertices=6000, K=24, seed=22)
    res = protocol.run(eco, 8, seed=3)
    assert np.abs(res["exposure"] - plaintext_exposure(eco, 8)).max() < 1e-6


@pytest.mark.parametrize("tier", ["H", "V"])
def test_audit_detects_deviation_and_accepts_honest_parties(tier):
    eco = generate(n_vertices=3000, K=3, seed=23)
    tau, seed, n = 5, 4, eco.K + 1
    dev = np.zeros(eco.n, dtype=np.int64)
    dev[7] = 1 << 36
    r = protocol.run(eco, tau, seed=seed, tier=tier, record=True, deviations={(2, 1): R.from_int64(dev)})
    parties = [protocol.Party(i, seed, eco.n) for i in range(n)]
    received = [[p.masks(protocol.K_BITS + protocol.SIGMA)[1] for p in parties] for _ in range(tau + 1)]
    for j in range(n):
        corr = [c[j] for c in r["corr_shares"]] if tier == "H" else None
        found = protocol.audit(eco, tau, j, seed, r["transcript"], r["commitments"], corr_shares=corr, tier=tier,
                               received=received if tier == "V" else None)
        assert found == ([("message", 2)] if j == 1 else [])


def test_refreshed_correction_after_update():
    eco = generate(n_vertices=5000, K=4, seed=24)
    new = dynamic.apply_update(eco, 400, seed=1)
    make, changed = dynamic.refreshed_dealer(eco, new, np.random.default_rng(1))
    assert sum(len(c) for c in changed.values()) > 0
    res = protocol.run(new, 10, correlations=make, seed=5)
    assert np.abs(res["exposure"] - plaintext_exposure(new, 10)).max() < 1e-6


def test_matching_and_omission_variants():
    eco = generate(n_vertices=20000, K=4, seed=25)
    assert variants.uncertain_matching(eco, "hard", 0.0).m == eco.m
    soft = variants.uncertain_matching(eco, "soft", 1.0, seed=1)
    assert soft.m == eco.m + 2 * int(eco.bridge.sum())
    gone = variants.omit_custodial(eco, 0.0, targeted=True)
    rec = (plaintext_exposure(gone, 12)[eco.endpoints] >= 1e-4).mean()
    assert rec < (plaintext_exposure(eco, 12)[eco.endpoints] >= 1e-4).mean()


def test_csv_roundtrip():
    eco = generate(n_vertices=1500, K=3, seed=26)
    with tempfile.TemporaryDirectory() as d:
        dataio.save(eco, d)
        back = dataio.load(d, K=3)
    assert np.array_equal(back.w_fp, eco.w_fp)
    assert np.array_equal(back.seeds, eco.seeds)
