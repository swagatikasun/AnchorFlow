import numpy as np
import pytest

from anchorflow import protocol
from anchorflow.gen import generate, plaintext_exposure


@pytest.mark.parametrize("K", [2, 4, 7])
def test_matches_plaintext(K):
    eco = generate(n_vertices=4000, K=K, seed=K)
    ref = plaintext_exposure(eco, 10)
    res = protocol.run(eco, 10, seed=1)
    assert np.abs(res["exposure"] - ref).max() < 1e-6


def test_relay_and_all_to_all_give_same_values():
    eco = generate(n_vertices=2000, K=3, seed=9)
    a = protocol.run(eco, 6, opening="all", seed=4)
    b = protocol.run(eco, 6, opening="relay", seed=4)
    assert np.array_equal(a["exposure"], b["exposure"])
    sa, sb = a["meter"].summary("online"), b["meter"].summary("online")
    assert sa["rounds"] == 7 and sb["rounds"] == 14


def test_rounding_offset_prevents_false_alerts():
    eco = generate(n_vertices=60000, K=4, seed=5)
    ref = plaintext_exposure(eco, 12)
    fixed = protocol.run(eco, 12, debias=True, seed=2)["exposure"]
    biased = protocol.run(eco, 12, debias=False, seed=2)["exposure"]
    assert np.abs(fixed - ref).max() < 1e-6
    assert np.abs(biased - ref).max() > 10 * np.abs(fixed - ref).max()


def test_planted_endpoints_need_custodial_columns():
    eco = generate(n_vertices=5000, K=4, seed=3)
    full = plaintext_exposure(eco, 12)
    public = plaintext_exposure(eco, 12, cols=eco.col_owner < 0)
    assert (full[eco.endpoints] >= 1e-4).all()
    custodial_ends = eco.endpoints[eco.owner[eco.endpoints] >= 0]
    assert len(custodial_ends) > 0
    assert (public[custodial_ends] >= 1e-4).sum() == 0
    assert (public[eco.endpoints] >= 1e-4).sum() < len(eco.endpoints) // 2
