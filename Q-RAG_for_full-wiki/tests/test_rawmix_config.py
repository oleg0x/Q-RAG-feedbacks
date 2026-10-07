"""NQ + HotpotQA raw mix: the config, holdout and weights agree with each other.

No GPU and no matrix: only the data seam is checked, i.e. that eval runs on the
holdout, that the holdout is excluded from training by weight, and that both
halves of the mix get their own curves. Skipped if the parquet is not present.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import pytest
from hydra import compose, initialize
from hydra.utils import instantiate

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from envs.search_env import SearchDatasetAdapter, load_weights


LAB = Path(__file__).resolve().parents[2]
TRAIN_DATA = LAB / "train_data"
PARQUET = TRAIN_DATA / "raw" / "train.parquet"

pytestmark = pytest.mark.skipif(
    not PARQUET.is_file(), reason=f"{PARQUET} not found: python src/build_train_mix.py"
)


@pytest.fixture(scope="module")
def cfg():
    with initialize(version_base="1.3", config_path="../configs"):
        return compose(config_name="training_fullwiki_rawmix")


def test_config_points_at_the_raw_mix(cfg) -> None:
    assert cfg.envs.task == "NQ+HotPotQA"
    assert cfg.eval_episodes == 2000
    assert str(cfg.envs.train_weights).endswith("weights/a1.jsonl")
    assert str(cfg.envs.eval_dataset.ids_file).endswith("holdout.json")
    # Compared variants must match in everything except the reward model.
    assert cfg.steps_count == 30000
    assert cfg.seed == 42


def test_eval_is_the_holdout_and_covers_both_halves(cfg) -> None:
    dataset = SearchDatasetAdapter(instantiate(cfg.envs.eval_dataset), None)
    assert len(dataset) == 2000
    sources = Counter(dataset[i]["source"] for i in range(len(dataset)))
    assert sources == {"nq": 1000, "hotpotqa": 1000}

    holdout = set(
        json.loads((TRAIN_DATA / "holdout.json").read_text(encoding="utf-8"))["ids"]
    )
    assert {dataset[i]["key"] for i in range(len(dataset))} == holdout


def test_holdout_is_excluded_from_training_by_weight(cfg) -> None:
    dataset = SearchDatasetAdapter(instantiate(cfg.envs.train_dataset), None)
    weights = load_weights(cfg.envs.train_weights, dataset)
    holdout = set(
        json.loads((TRAIN_DATA / "holdout.json").read_text(encoding="utf-8"))["ids"]
    )

    assert len(dataset) == 169615
    assert float(weights.sum()) == len(dataset) - len(holdout)
    zeroed = {
        dataset[index]["key"] for index in range(len(dataset)) if weights[index] == 0
    }
    assert zeroed == holdout


def test_gold_titles_come_from_the_parquet_for_the_hotpotqa_half(cfg) -> None:
    """The HotpotQA half carries gold titles in metadata, the NQ half does not.

    `pool/gold_title_recall` depends on this: without titles pool drift cannot
    be measured, and the titles are in the file itself, so no join on question
    text is needed.
    """
    dataset = SearchDatasetAdapter(instantiate(cfg.envs.eval_dataset), None)
    by_source = {"nq": [], "hotpotqa": []}
    for index in range(len(dataset)):
        sample = dataset[index]
        by_source[sample["source"]].append(sample)

    hotpot = by_source["hotpotqa"]
    assert all(len(item["gold_titles"]) == 2 for item in hotpot)
    # The title repeats for every supporting sentence; deduplication must
    # leave exactly two articles.
    assert all(
        len(set(item["gold_titles"])) == len(item["gold_titles"]) for item in hotpot
    )
    assert all(not item["gold_titles"] for item in by_source["nq"])
    # Without a title table the coverage flag is unknown for both halves; with
    # one it must stay None for NQ: "not checked", not "not covered".
    assert all(item["gold_titles_covered"] is None for item in by_source["nq"])


def test_answer_variants_survive_to_the_episode(cfg) -> None:
    """Examples with several aliases reach the environment as a list, not joined."""
    dataset = SearchDatasetAdapter(instantiate(cfg.envs.eval_dataset), None)
    variants = [dataset[i]["answer_variants"] for i in range(len(dataset))]
    assert all(isinstance(item, list) and item for item in variants)
    assert max(len(item) for item in variants) > 1
    assert all(
        dataset[i]["answer"] == variants[i][0] for i in range(len(dataset))
    )
