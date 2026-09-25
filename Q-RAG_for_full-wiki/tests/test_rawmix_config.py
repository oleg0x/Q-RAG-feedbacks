"""Сырая смесь NQ + HotpotQA: конфиг, holdout и веса сходятся между собой.

Без GPU и без матрицы: проверяется ровно стык данных — что eval идёт по
holdout, что holdout исключён из обучения весом, и что раздельные кривые
получат обе половины смеси. Пропускается, если parquet не скачан.
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
    not PARQUET.is_file(), reason=f"нет {PARQUET}: python build_train_mix.py"
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
    # Руки ветки обязаны совпадать во всём, кроме модели награды.
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
    """Половина HotpotQA несёт gold-титулы в metadata, половина NQ — нет.

    От этого зависит `pool/gold_title_recall`: без титулов дрейф пула нечем
    мерить, а титулы лежат прямо в файле — джойн по тексту вопроса не нужен.
    """
    dataset = SearchDatasetAdapter(instantiate(cfg.envs.eval_dataset), None)
    by_source = {"nq": [], "hotpotqa": []}
    for index in range(len(dataset)):
        sample = dataset[index]
        by_source[sample["source"]].append(sample)

    hotpot = by_source["hotpotqa"]
    assert all(len(item["gold_titles"]) == 2 for item in hotpot)
    # Титул повторяется на каждый supporting sentence — дедупликация обязана
    # оставить ровно две статьи.
    assert all(
        len(set(item["gold_titles"])) == len(item["gold_titles"]) for item in hotpot
    )
    assert all(not item["gold_titles"] for item in by_source["nq"])
    # Без таблицы титулов флаг покрытия неизвестен у обеих половин; с ней он
    # обязан остаться None у NQ — «не проверяли», а не «не покрыто».
    assert all(item["gold_titles_covered"] is None for item in by_source["nq"])


def test_answer_variants_survive_to_the_episode(cfg) -> None:
    """Многовариантные примеры доезжают до среды списком, а не склейкой."""
    dataset = SearchDatasetAdapter(instantiate(cfg.envs.eval_dataset), None)
    variants = [dataset[i]["answer_variants"] for i in range(len(dataset))]
    assert all(isinstance(item, list) and item for item in variants)
    assert max(len(item) for item in variants) > 1
    assert all(
        dataset[i]["answer"] == variants[i][0] for i in range(len(dataset))
    )
