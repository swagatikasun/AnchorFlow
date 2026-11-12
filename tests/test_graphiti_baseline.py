import numpy as np

from anchorflow.gen import generate, plaintext_exposure, seed_vector
from anchorflow.graphiti import run as graphiti


def test_graphiti_style_baseline_matches_plaintext():
    eco = generate(n_vertices=3000, K=3, seed=8)
    res = graphiti.run(eco.src, eco.dst, eco.w_fp, eco.n, seed_vector(eco), 8)
    assert np.abs(res["exposure"] - plaintext_exposure(eco, 8)).max() < 1e-5
