#!/usr/bin/env python3
"""Сырая смесь NQ + HotpotQA для обучения: манифест, holdout и веса эпизодов.

Вход — `train.parquet` датасета `PeterJinGo/nq_hotpotqa_train` (тот же, на
котором учится Search-R1), лежащий в `train_data/raw/`. Скрипт ничего не
фильтрует и не дедуплицирует намеренно: ветка обучалась на полном наборе — том
же, что у Search-R1, — чтобы разница с ним не объяснялась подготовкой данных.
Он производит три артефакта.

``manifest.json`` — число строк, разбивка и доли по ``data_source``, sha256
обоих parquet-файлов. Нужен, чтобы «на чём училась эта ветка» отвечалось без
пересчёта по 170 тыс. строк.

``holdout.json`` — по ``--holdout-per-source`` примеров каждой половины
смеси, фиксированный сид. Один и тот же во всех руках ветки: eval обязан
покрывать ту же смесь, что и обучение, раздельными кривыми, а сравнивать руки
между собой можно только на общем наборе.

``weights/a1.jsonl`` — вес каждого примера: 1 везде, 0 на holdout. Через этот
файл среда и тянет эпизоды (``envs/search_env.py``), поэтому исключение
holdout из обучения и будущие фильтрованные руки A2/A3 — это один механизм и
один формат.

Ключ примера — ``"{data_source}:{id}"``, а не голый ``id``: обе половины
смеси нумеруются с нуля, и все 79 168 идентификаторов NQ совпадают с чужими
идентификаторами HotpotQA.

    python src/build_train_mix.py --check   # только пересчёт и сверка манифеста
    python src/build_train_mix.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
from pathlib import Path
from typing import Any

import pandas as pd

import runlib


LOG = logging.getLogger("build-train-mix")

# train_data/ лежит в корне репозитория, а этот модуль — в src/.
DEFAULT_ROOT = runlib.REPO / "train_data"
# sha256 `test.parquet` семи бенчмарков Search-R1: строки в
# `runs/shared/searchr1/` нарезаны именно из него, и расхождение означало бы,
# что опубликованные семь строк сняты не с того файла.
SEARCHR1_TEST_SHA256 = (
    "30aa887b6d47e06e8c0f6f5307c88fe4e13461ac25a20ec0a5433ad7a4fe25dc"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def example_key(data_source: str, sample_id: str) -> str:
    return f"{data_source}:{sample_id}"


def load_mix(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(
        path, columns=["id", "question", "golden_answers", "data_source"]
    )
    frame["key"] = [
        example_key(source, sample_id)
        for source, sample_id in zip(frame["data_source"], frame["id"])
    ]
    if frame["key"].nunique() != len(frame):
        raise ValueError(
            f"{path}: ключи примеров не уникальны — holdout и веса адресовать нечем"
        )
    return frame


def describe(frame: pd.DataFrame) -> dict[str, Any]:
    """Разбивка смеси: сколько примеров и вариантов ответа у каждой половины."""
    counts = frame["data_source"].value_counts()
    variants = frame["golden_answers"].apply(len)
    sources = {}
    for source in sorted(counts.index):
        subset = variants[frame["data_source"] == source]
        sources[source] = {
            "rows": int(counts[source]),
            "share": float(counts[source] / len(frame)),
            "answer_variants_mean": float(subset.mean()),
            "answer_variants_max": int(subset.max()),
            "multi_answer_share": float((subset > 1).mean()),
        }
    return {
        "rows": int(len(frame)),
        "answer_variants_mean": float(variants.mean()),
        "answer_variants_max": int(variants.max()),
        "multi_answer_share": float((variants > 1).mean()),
        "sources": sources,
    }


def pick_holdout(frame: pd.DataFrame, per_source: int, seed: int) -> list[str]:
    """По ``per_source`` ключей каждой половины смеси, фиксированный сид.

    Выборка идёт по отсортированным ключам, а не по порядку строк parquet:
    порядок в файле — это чужое решение, и привязываться к нему значит
    получить другой holdout при пересохранении датасета.
    """
    chosen: list[str] = []
    for source in sorted(frame["data_source"].unique()):
        keys = sorted(frame.loc[frame["data_source"] == source, "key"])
        if per_source > len(keys):
            raise ValueError(
                f"{source}: просили {per_source} примеров, всего {len(keys)}"
            )
        rng = random.Random(f"{seed}:{source}")
        chosen.extend(sorted(rng.sample(keys, per_source)))
    return chosen


def write_weights(path: Path, frame: pd.DataFrame, holdout: set[str]) -> dict[str, int]:
    path.parent.mkdir(parents=True, exist_ok=True)
    drawable = 0
    with path.open("w", encoding="utf-8") as sink:
        for key in frame["key"]:
            weight = 0.0 if key in holdout else 1.0
            drawable += weight > 0
            sink.write(json.dumps({"key": key, "w": weight}) + "\n")
    return {"rows": int(len(frame)), "drawable": int(drawable)}


def build(root: Path, per_source: int, seed: int, check: bool) -> dict[str, Any]:
    raw = root / "raw"
    train_file = raw / "train.parquet"
    test_file = raw / "test.parquet"
    for path in (train_file, test_file):
        if not path.is_file():
            raise SystemExit(f"Нет {path}: сначала скачайте датасет")

    test_sha = sha256_file(test_file)
    if test_sha != SEARCHR1_TEST_SHA256:
        raise SystemExit(
            f"{test_file}: sha256 {test_sha} против ожидаемого "
            f"{SEARCHR1_TEST_SHA256}. Семь строк Search-R1 в runs/shared/ "
            "сняты с другого файла — остановитесь и разберитесь."
        )

    frame = load_mix(train_file)
    holdout = pick_holdout(frame, per_source, seed)
    manifest: dict[str, Any] = {
        "dataset": "PeterJinGo/nq_hotpotqa_train",
        "revision": "b7d80abfee334a7a91cb377544f09180d58b34f6",
        "created_utc": runlib.utc_now(),
        "files": {
            "train.parquet": {
                "sha256": sha256_file(train_file),
                "bytes": train_file.stat().st_size,
            },
            "test.parquet": {"sha256": test_sha, "bytes": test_file.stat().st_size},
        },
        "test_parquet_matches_searchr1": True,
        "train": describe(frame),
        "holdout": {
            "seed": seed,
            "per_source": per_source,
            "size": len(holdout),
        },
    }

    if check:
        stored = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        stored_holdout = json.loads(
            (root / "holdout.json").read_text(encoding="utf-8")
        )["ids"]
        same = (
            stored["train"] == manifest["train"]
            and stored["files"] == manifest["files"]
            and stored_holdout == holdout
        )
        LOG.info("Сверка манифеста и holdout: %s", "совпали" if same else "РАЗОШЛИСЬ")
        if not same:
            raise SystemExit("Артефакты разошлись с пересчётом")
        return manifest

    (root / "holdout.json").write_text(
        json.dumps(
            {
                "seed": seed,
                "per_source": per_source,
                "source": "train_data/raw/train.parquet",
                "key_format": "{data_source}:{id}",
                "ids": holdout,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    stats = write_weights(root / "weights" / "a1.jsonl", frame, set(holdout))
    manifest["weights"] = {"a1.jsonl": stats}
    (root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    LOG.info(
        "Смесь: %d строк, holdout %d, тянется %d",
        manifest["train"]["rows"],
        len(holdout),
        stats["drawable"],
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--holdout-per-source", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--check",
        action="store_true",
        help="только пересчитать и сверить с тем, что уже лежит",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    build(args.root, args.holdout_per_source, args.seed, args.check)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
