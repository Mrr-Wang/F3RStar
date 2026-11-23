"""Unit tests for BaseStarDataset.build_posterior_stars (InfoStar posterior
star selection dataset-side helper).

Uses a bare duck-typed stand-in instead of constructing a full
``BaseStarDataset`` (which needs images/paths/etc.) -- the method only
touches ``self.neighbors_all``.
"""

from __future__ import annotations

import numpy as np

from gluemap.datasets.star import BaseStarDataset


class _FakeDataset:
    def __init__(self, neighbors_all: dict[int, set[int]]):
        self.neighbors_all = neighbors_all


def test_build_posterior_stars_matches_center_plus_neighbors():
    fake = _FakeDataset({0: {1, 2}, 1: {0, 2}, 2: {0, 1}, 3: {4}, 4: {3}})
    out = BaseStarDataset.build_posterior_stars(fake, [1, 3])

    assert len(out) == 2
    star1, star3 = out
    assert star1[0] == 1
    assert set(star1[1:].tolist()) == {0, 2}
    assert star3[0] == 3
    assert set(star3[1:].tolist()) == {4}
    assert star1.dtype == np.int64


def test_build_posterior_stars_skips_isolated_candidates():
    fake = _FakeDataset({0: {1}, 1: {0}, 2: set()})
    out = BaseStarDataset.build_posterior_stars(fake, [2])
    assert out == []


def test_build_posterior_stars_empty_input():
    fake = _FakeDataset({0: {1}, 1: {0}})
    assert BaseStarDataset.build_posterior_stars(fake, []) == []
