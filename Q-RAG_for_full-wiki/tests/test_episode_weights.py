"""Weighted episode sampling: the holdout is excluded by weight, not a filter.

Episodes are drawn by the environment, not a dataloader, so the sampler lives
in ``DenseSearchEnv.reset``. The tests check exactly what it exists for: equal
weights keep the distribution uniform, given weights converge to the target
shares, and a zero weight is never drawn.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from envs.search_env import DenseSearchEnv, load_weights


class ListDataset:
    def __init__(self, keys):
        self.samples = [
            {"key": key, "question": key, "answer": key, "id": key}
            for key in keys
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


class NoFeedback:
    def reset(self, obs, info):
        pass

    def copy(self):
        return NoFeedback()


def make_env(dataset, weights=None, seed=0):
    return DenseSearchEnv(
        dataset=dataset,
        max_steps=2,
        feedback_model=NoFeedback(),
        seed=seed,
        weights=weights,
    )


def draw(env, count):
    return Counter(env.sample_index() for _ in range(count))


def test_equal_weights_stay_uniform() -> None:
    """χ² of equal weights against uniform: any deviation must be noise.

    Eight seeds are pooled into one sample on purpose: on a single seed χ² is a
    lottery (28 for seed 0 against an expectation of 7), and the test would
    catch an unlucky draw instead of sampler bias.
    """
    dataset = ListDataset([f"k{i}" for i in range(8)])
    per_seed = 20000
    counts = Counter()
    for seed in range(8):
        counts.update(draw(make_env(dataset, np.ones(8), seed=seed), per_seed))
    observed = np.array([counts[i] for i in range(8)], dtype=float)
    expected = 8 * per_seed / 8
    chi2 = float(((observed - expected) ** 2 / expected).sum())
    # 7 degrees of freedom: the critical value at the 0.001 level is 24.32.
    assert chi2 < 24.32, (chi2, observed)


def test_given_weights_converge_to_their_shares() -> None:
    dataset = ListDataset([f"k{i}" for i in range(4)])
    weights = np.array([1.0, 3.0, 0.0, 4.0])
    draws = 40000
    counts = draw(make_env(dataset, weights), draws)
    shares = np.array([counts[i] for i in range(4)], dtype=float) / draws
    assert np.allclose(shares, weights / weights.sum(), atol=0.01)


def test_zero_weight_is_never_drawn() -> None:
    """This is how the holdout is excluded: weight 0 is never drawn."""
    dataset = ListDataset([f"k{i}" for i in range(5)])
    counts = draw(make_env(dataset, np.array([1.0, 0.0, 1.0, 0.0, 1.0])), 20000)
    assert counts[1] == 0 and counts[3] == 0
    assert set(counts) == {0, 2, 4}


def test_without_weights_the_old_uniform_path_is_kept() -> None:
    """Without a weights file, sampling uses the old ``integers`` path, bit for bit."""
    dataset = ListDataset([f"k{i}" for i in range(6)])
    env = make_env(dataset, weights=None, seed=7)
    expected = np.random.default_rng(7).integers(6, size=50).tolist()
    assert [env.sample_index() for _ in range(50)] == expected


def test_copies_share_the_weights(tmp_path: Path) -> None:
    dataset = ListDataset([f"k{i}" for i in range(4)])
    env = make_env(dataset, np.array([0.0, 1.0, 0.0, 0.0]))
    clone = env.copy()
    clone._rng = np.random.default_rng(3)
    assert set(draw(clone, 200)) == {1}


def test_load_weights_follows_dataset_order(tmp_path: Path) -> None:
    dataset = ListDataset(["b", "a", "c"])
    path = tmp_path / "w.jsonl"
    path.write_text(
        "\n".join(
            json.dumps({"key": key, "w": weight})
            for key, weight in (("a", 0.0), ("b", 1.0), ("c", 2.0))
        )
        + "\n",
        encoding="utf-8",
    )
    assert load_weights(path, dataset).tolist() == [1.0, 0.0, 2.0]


def test_missing_key_is_an_error_not_a_default(tmp_path: Path) -> None:
    """A forgotten line would quietly extend the training set."""
    dataset = ListDataset(["a", "b"])
    path = tmp_path / "w.jsonl"
    path.write_text(json.dumps({"key": "a", "w": 1.0}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no weight"):
        load_weights(path, dataset)


def test_all_zero_weights_are_an_error(tmp_path: Path) -> None:
    dataset = ListDataset(["a"])
    path = tmp_path / "w.jsonl"
    path.write_text(json.dumps({"key": "a", "w": 0.0}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="total weight is zero"):
        load_weights(path, dataset)
