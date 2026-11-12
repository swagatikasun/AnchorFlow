import numpy as np

from anchorflow import he, paillier, protocol
from anchorflow.gen import generate, plaintext_exposure, seed_vector


def test_threshold_paillier_roundtrip():
    key = paillier.keygen(512, 3)
    for m in (0, 1, 123456789, (1 << 300) + 7):
        m %= key.pub.n
        c = key.pub.encrypt(m)
        assert key.decrypt(c) == m
    c = key.pub.add(key.pub.encrypt(5), key.pub.smul(key.pub.encrypt(7), 3))
    assert key.decrypt(c) == 26


def test_homomorphic_offline_phase_end_to_end():
    eco = generate(n_vertices=80, K=2, seed=4, n_seeds=1)
    n = eco.K + 1
    key = paillier.keygen(1024, n)
    infos = []

    def make(rp):
        sh, info = he.offline_correlation(eco, key, [r.to_ints() for r in rp], mode="L1")
        infos.append(info)
        return sh

    res = protocol.run(eco, 2, correlations=make, seed=3)
    assert np.abs(res["exposure"] - plaintext_exposure(eco, 2)).max() < 1e-6
    ops, _ = he.anchorflow_offline_counts(eco, n, 1, "L1", infos[0]["slots"])
    assert infos[0]["ops"] == dict(ops)


def test_he_propagation_baseline():
    eco = generate(n_vertices=60, K=2, seed=6, n_seeds=1)
    n = eco.K + 1
    key = paillier.keygen(512, n)
    tau = 2
    x0 = np.round(seed_vector(eco) * (1 << 32)).astype(np.int64)
    out, info = he.egmond_run(eco, key, tau, x0)
    got = np.array(out, dtype=float) / 2.0 ** (32 + 20 * tau)
    assert np.abs(got - plaintext_exposure(eco, tau)).max() < 1e-9
    ops, _ = he.egmond_counts(eco, n, tau)
    assert info["ops"] == dict(ops)
