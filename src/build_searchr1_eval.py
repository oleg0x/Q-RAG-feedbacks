#!/usr/bin/env python3
"""Разложить eval-таблицу Search-R1 на семь входов нашего пайплайна.

`PeterJinGo/nq_hotpotqa_train` называется train-датасетом, но его
``test.parquet`` — это все семь бенчмарков Search-R1 в одной таблице, по
колонке ``data_source``: nq, triviaqa, popqa, hotpotqa, 2wikimultihopqa,
musique, bamboogle. Здесь она режется на семь JSONL, каждый из которых
``fullwiki_qrag.py`` читает как обычный вход.

Две вещи, которые тут делаются намеренно и без которых числа будут неверны.

**``supporting_facts`` не пишется вовсе.** Gold-титулов в parquet нет ни у
одного из семи датасетов, а пустой список — не то же самое, что отсутствие
поля: ``title_metrics`` на пустом голде возвращает ``title_recall = title_em
= 1.0``, то есть ран отчитался бы стопроцентным попаданием по титулам. Без
поля ``add_eval_fields`` title-метрики просто не считает, и ``score_run``
проставляет им ``null``.

**Алиасы ответа сохраняются все.** Search-R1 считает EM максимумом по всему
``golden_answers``, а ``answer_judge_llms.py`` (read-only) сравнивает с
единственным ``answer``. Поэтому первый ответ идёт в ``answer``, остальные —
в ``answer_aliases``, откуда их берут ``export_eval_json.py`` и
``report_phase0.score_run`` для alias-версий EM и F1. Без этого TriviaQA (в
среднем 14 допустимых ответов на вопрос, 88% вопросов с несколькими) окажется
занижен в разы.

Пример:

    python src/build_searchr1_eval.py \
      --input runs/shared/searchr1/test.parquet \
      --output-dir runs/shared/searchr1
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Iterator, Sequence

from fullwiki_qrag import load_input_samples, sample_id


LOG = logging.getLogger("build-searchr1-eval")

# Ожидаемый состав таблицы. Проверяется при конвертации: сдвиг в числе
# примеров означает, что скачался другой снимок датасета, и тогда наши числа
# нельзя ставить рядом с опубликованными Search-R1.
EXPECTED_ROWS: dict[str, int] = {
    "nq": 3610,
    "triviaqa": 11313,
    "popqa": 14267,
    "hotpotqa": 7405,
    "2wikimultihopqa": 12576,
    "musique": 2417,
    "bamboogle": 125,
}

# Поля, которых в результате быть не должно. Список существует ради
# ``supporting_facts``: пустой список молча превращает title-метрики в 100%,
# см. заголовок модуля.
FORBIDDEN_FIELDS = ("supporting_facts", "context", "sf_idx")


def convert_sample(row: dict[str, Any], source: str) -> dict[str, Any]:
    question = row.get("question")
    if not isinstance(question, str) or not question.strip():
        raise ValueError(f"{source}: пример {row.get('id')!r} без вопроса")
    # Не `or []`: pandas отдаёт golden_answers массивом numpy, и проверка на
    # истинность массива длиннее одного элемента — исключение, а не пустота.
    golden = row.get("golden_answers")
    answers = [] if golden is None else [str(value) for value in golden]
    answers = [value for value in answers if value.strip()]
    if not answers:
        raise ValueError(f"{source}: у примера {row.get('id')!r} нет ответов")
    return {
        # Идентификатор в parquet уникален только внутри своего
        # data_source: и у nq, и у popqa первая строка называется `test_0`.
        # Префикс делает id уникальным во всём реестре ранов.
        "_id": f"{source}_{row['id']}",
        "data_source": source,
        "question": question,
        "answer": answers[0],
        "answer_aliases": answers[1:],
    }


def convert(rows: Sequence[dict[str, Any]], source: str) -> Iterator[dict[str, Any]]:
    for row in rows:
        record = convert_sample(row, source)
        leaked = [field for field in FORBIDDEN_FIELDS if field in record]
        if leaked:
            raise RuntimeError(
                f"{source}: в записи оказались поля {leaked}; пустой "
                "supporting_facts даёт title EM = 100% на пустом голде"
            )
        yield record


def verify_output(path: Path, expected: Sequence[dict[str, Any]]) -> None:
    """Перечитать результат тем же кодом, которым его прочитает пайплайн.

    Проверяется не содержимое файла как текста, а то, что из него достанет
    ``fullwiki_qrag.load_input_samples``: тот же порядок, те же вопросы, те же
    идентификаторы и по-прежнему отсутствующий ``supporting_facts``.
    """
    reread = load_input_samples(path)
    if len(reread) != len(expected):
        raise RuntimeError(
            f"{path.name}: перечитано {len(reread)} примеров вместо {len(expected)}"
        )
    for index, (actual, want) in enumerate(zip(reread, expected)):
        if actual["question"] != want["question"]:
            raise RuntimeError(f"{path.name}: вопрос {index} не совпал после записи")
        if sample_id(actual, index) != want["_id"]:
            raise RuntimeError(f"{path.name}: id {index} не совпал после записи")
        if actual.get("supporting_facts") is not None:
            raise RuntimeError(
                f"{path.name}: у примера {index} появился supporting_facts"
            )


def write_source(
    rows: Sequence[dict[str, Any]], source: str, directory: Path
) -> tuple[Path, int]:
    destination = directory / f"{source}.jsonl"
    records = list(convert(rows, source))
    # Через временный файл: оборванная конвертация не должна оставить
    # правдоподобный, но неполный вход для многочасового рана.
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(destination)
    verify_output(destination, records)
    with_aliases = sum(1 for record in records if record["answer_aliases"])
    LOG.info(
        "%-16s %6d примеров, с алиасами %5d (%.0f%%) → %s",
        source,
        len(records),
        with_aliases,
        100 * with_aliases / len(records),
        destination.name,
    )
    return destination, len(records)


def load_table(path: Path) -> dict[str, list[dict[str, Any]]]:
    import pandas as pd

    frame = pd.read_parquet(path)
    missing = {"id", "question", "golden_answers", "data_source"} - set(frame.columns)
    if missing:
        raise ValueError(f"{path}: в таблице нет колонок {sorted(missing)}")
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in frame.to_dict("records"):
        grouped.setdefault(str(row["data_source"]), []).append(row)
    return grouped


def check_composition(grouped: dict[str, list[dict[str, Any]]]) -> None:
    actual = {source: len(rows) for source, rows in grouped.items()}
    if actual != EXPECTED_ROWS:
        raise ValueError(
            "Состав таблицы разошёлся с ожидаемым; наши числа нельзя ставить "
            f"рядом с опубликованными Search-R1.\nожидалось: {EXPECTED_ROWS}\n"
            f"получено:  {actual}"
        )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
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
    source_path = args.input.expanduser().resolve()
    directory = args.output_dir.expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)

    grouped = load_table(source_path)
    LOG.info("Прочитано %d датасетов: %s", len(grouped), source_path)
    check_composition(grouped)

    total = 0
    # Порядок обхода — из EXPECTED_ROWS, а не из таблицы: лог должен читаться
    # одинаково от запуска к запуску.
    for source in EXPECTED_ROWS:
        total += write_source(grouped[source], source, directory)[1]
    LOG.info("Записано %d примеров в %s", total, directory)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
