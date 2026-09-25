#!/usr/bin/env python3
"""Срез рана для чтения человеком: вопрос, голд, ответ модели, чанки.

``answer_judge.json`` содержит всё нужное, но неудобен: он тащит
``supporting_facts``, ``sf_idx`` и покандидатные дампы каждого хопа, титул
статьи спрятан первой строкой внутри текста чанка, а алиасов ответа в нём нет
вовсе — ``answer_judge_llms.py`` сравнивает предсказание с единственным
``answer``.

Здесь запись раскладывается по полям, а рядом с основными метриками
появляются alias-версии EM и F1: у 2Wiki алиасы лежат в ``id_aliases.json`` и
адресуются полем ``answer_id``, у MuSiQue — в ``answer_aliases`` самого
примера. Официальные эвалы обоих датасетов считают EM максимумом по алиасам,
поэтому без этой колонки наши числа систематически ниже опубликованных.

Основные метрики не пересчитываются, а **проверяются**: ``EM`` и ``F1``,
посчитанные здешним кодом по основному ответу, обязаны совпасть с теми, что
лежат в ``answer_judge.json``. Расхождение означает, что нормализация
разъехалась с судейской, и тогда alias-версиям верить нельзя — скрипт падает.

Пример:

    python src/export_eval_json.py --run 2026-08-05-lineA-best-2wiki
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import string
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

import runlib
from fullwiki_qrag import load_input_samples, sample_id, wiki_title


LOG = logging.getLogger("export-eval-json")

EXPORT_FILE = "export.json"
ALIAS_FILE = "id_aliases.json"


# --------------------------------------------------------------------------
# метрики
#
# Копия нормализации из read-only ``answer_judge_llms.py``. Копия, а не
# импорт: тот модуль при импорте тянет vLLM-клиента и промпты, а править его
# нельзя. Совпадение гарантируется не чтением глазами, а сверкой на каждой
# записи в :func:`export_record`.


def normalize_answer(value: str) -> str:
    value = value.lower()
    value = re.sub(r"\b(a|an|the)\b", " ", value)
    value = "".join(char for char in value if char not in string.punctuation)
    return " ".join(value.split())


def exact_match(prediction: str, target: str) -> int:
    return int(normalize_answer(prediction) == normalize_answer(target))


def f1_score(prediction: str, target: str) -> float:
    prediction_tokens = normalize_answer(prediction).split()
    target_tokens = normalize_answer(target).split()
    if not prediction_tokens or not target_tokens:
        return float(prediction_tokens == target_tokens)
    overlap = sum((Counter(prediction_tokens) & Counter(target_tokens)).values())
    if not overlap:
        return 0.0
    precision = overlap / len(prediction_tokens)
    recall = overlap / len(target_tokens)
    return 2 * precision * recall / (precision + recall)


def best_over_aliases(
    prediction: str, targets: Sequence[str]
) -> tuple[int, float]:
    """Максимум EM и F1 по списку допустимых ответов, как в официальных эвалах."""
    em = max((exact_match(prediction, target) for target in targets), default=0)
    f1 = max((f1_score(prediction, target) for target in targets), default=0.0)
    return em, f1


# --------------------------------------------------------------------------
# алиасы


def load_id_aliases(path: Path) -> dict[str, list[str]]:
    """``id_aliases.json`` 2WikiMultiHopQA: строка на сущность Wikidata."""
    aliases: dict[str, list[str]] = {}
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            record = json.loads(line)
            aliases[str(record["Q_id"])] = [
                str(value) for value in record.get("aliases", [])
            ]
    return aliases


def collect_aliases(dataset: Path, alias_source: Path | None) -> dict[str, list[str]]:
    """Отображение id примера в допустимые ответы помимо основного.

    Форма зависит от датасета: MuSiQue носит алиасы в самом примере, 2Wiki
    адресует их через ``answer_id`` во внешнюю таблицу сущностей, HotpotQA не
    имеет их вовсе. Все три случая разбираются здесь, чтобы вызывающему коду
    не приходилось знать, какой датасет он экспортирует.
    """
    samples = load_input_samples(dataset)
    inline = {
        sample_id(sample, index): [
            str(value) for value in sample.get("answer_aliases") or []
        ]
        for index, sample in enumerate(samples)
        if sample.get("answer_aliases")
    }
    if inline:
        LOG.info("Алиасы из самого датасета: %d примеров", len(inline))
        return inline

    if alias_source is None:
        candidate = dataset.parent / ALIAS_FILE
        alias_source = candidate if candidate.is_file() else None
    if alias_source is None:
        LOG.info("Алиасов у датасета нет: alias-метрики совпадут с основными")
        return {}

    table = load_id_aliases(alias_source)
    LOG.info("Таблица алиасов: %d сущностей, %s", len(table), alias_source)
    by_sample = {}
    for index, sample in enumerate(samples):
        answer_id = sample.get("answer_id")
        if not answer_id:
            continue
        found = table.get(str(answer_id))
        if found:
            by_sample[sample_id(sample, index)] = found
    LOG.info("Алиасы разрешены у %d примеров", len(by_sample))
    return by_sample


# --------------------------------------------------------------------------
# экспорт


def split_chunk(text: str) -> tuple[str, str]:
    """Чанк корпуса хранится как ``"Титул"\\nтекст``; разложить обратно."""
    _, _, body = text.partition("\n")
    return wiki_title(text), body


def export_record(
    record: dict[str, Any],
    aliases: Sequence[str],
    index: int,
) -> dict[str, Any]:
    prediction = record.get("prediction") or ""
    answer = record.get("answer") or ""

    # Сверка с судейскими числами: она и есть гарантия, что alias-версии
    # посчитаны той же нормализацией, что и колонка EM в таблице.
    if exact_match(prediction, answer) != int(record["EM"]):
        raise RuntimeError(
            f"Запись {record.get('id', index)!r}: EM разошёлся с "
            f"answer_judge.json ({exact_match(prediction, answer)} против "
            f"{record['EM']}); нормализация ответа не совпадает с судейской"
        )
    if abs(f1_score(prediction, answer) - float(record["F1"])) > 1e-9:
        raise RuntimeError(
            f"Запись {record.get('id', index)!r}: F1 разошёлся с answer_judge.json"
        )

    targets = [answer, *aliases]
    em_alias, f1_alias = best_over_aliases(prediction, targets)

    hops = record.get("retrieval_hops") or []
    chunks = []
    for position, text in enumerate(record.get("pred_texts", [])):
        title, body = split_chunk(text)
        hop = hops[position] if position < len(hops) else {}
        chunks.append(
            {
                "step": hop.get("step", position),
                "title": title,
                "text": body,
                "score": hop.get("selected_score"),
                "row": (record.get("pred_idx") or [None] * (position + 1))[position],
            }
        )

    return {
        "id": record.get("id", index),
        "question": record.get("question"),
        "gold_answer": answer,
        "gold_aliases": list(aliases),
        "prediction": prediction,
        "gold_titles": record.get("gold_titles", []),
        "chunks": chunks,
        "metrics": {
            "em": int(record["EM"]),
            "f1": float(record["F1"]),
            "judge": float(record["LLM_Judge_Score"]),
            "em_alias": em_alias,
            "f1_alias": f1_alias,
            "title_em": record.get("title_em"),
            "title_recall": record.get("title_recall"),
        },
    }


def summarize(exported: Sequence[dict[str, Any]]) -> dict[str, Any]:
    def mean(key: str) -> float:
        values = [item["metrics"][key] for item in exported]
        return sum(values) / len(values) if values else 0.0

    return {
        "examples": len(exported),
        "em": mean("em"),
        "f1": mean("f1"),
        "judge": mean("judge"),
        "em_alias": mean("em_alias"),
        "f1_alias": mean("f1_alias"),
        "with_aliases": sum(1 for item in exported if item["gold_aliases"]),
    }


def build_export(
    run_id: str,
    manifest: dict[str, Any],
    records: Iterable[dict[str, Any]],
    aliases: dict[str, list[str]],
) -> dict[str, Any]:
    config = manifest.get("config", {})
    retrieve = config.get("retrieve", {})
    exported = [
        export_record(record, aliases.get(str(record.get("id")), []), index)
        for index, record in enumerate(records)
    ]
    return {
        "run_id": run_id,
        "dataset": runlib.dataset_key(config),
        "label": config.get("label"),
        "input": retrieve.get("input"),
        "checkpoint": retrieve.get("checkpoint"),
        "steps": retrieve.get("steps"),
        "max_chunks_per_title": retrieve.get("max_chunks_per_title"),
        "reader_model": (config.get("judge") or {}).get("answer_model"),
        "summary": summarize(exported),
        "examples": exported,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="run_id из реестра")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="файл вопросов; по умолчанию retrieve.input из манифеста",
    )
    parser.add_argument(
        "--alias-source",
        type=Path,
        default=None,
        help=f"таблица алиасов; по умолчанию {ALIAS_FILE} рядом с датасетом",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--log-level", choices=("DEBUG", "INFO", "WARNING"), default="INFO"
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    run = runlib.run_dir(args.run)
    manifest = runlib.read_manifest(run)
    judge_file = run / runlib.JUDGE_FILE
    if not judge_file.exists():
        raise SystemExit(f"Нет {judge_file}: экспортировать нечего")

    dataset = args.dataset or Path(manifest["config"]["retrieve"]["input"])
    aliases = collect_aliases(dataset.expanduser().resolve(), args.alias_source)

    with judge_file.open("r", encoding="utf-8") as stream:
        records = json.load(stream)
    export = build_export(args.run, manifest, records, aliases)

    destination = args.output or run / EXPORT_FILE
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(export, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    temporary.replace(destination)
    summary = export["summary"]
    LOG.info(
        "%s: %d примеров, EM=%.4f (alias %.4f), F1=%.4f, judge=%.4f → %s",
        export["dataset"],
        summary["examples"],
        summary["em"],
        summary["em_alias"],
        summary["f1"],
        summary["judge"],
        destination,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
