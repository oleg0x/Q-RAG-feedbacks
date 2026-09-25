#!/usr/bin/env python3
"""Переложить MuSiQue в схему HotpotQA, которую понимает весь пайплайн.

``fullwiki_qrag.py`` читает вход через ``load_input_samples`` и достаёт из
него ``question``, ``answer`` и пару ``context`` / ``supporting_facts``. У
MuSiQue вместо этой пары — плоский список ``paragraphs`` с флагом
``is_supporting``, поэтому сырой файл проходит через пайплайн молча и без
gold-титулов. Молчание тут дороже падения: ``title_metrics`` на пустом
списке gold-титулов возвращает ``title_recall = title_em = 1.0``, то есть
``metrics.json`` показал бы стопроцентное попадание по титулам и долю
потолка больше единицы.

Абзацы группируются по титулу, а не раскладываются один к одному: у 1 293 из
2 417 вопросов dev-сплита в списке есть два абзаца одной статьи, а у 116 оба
опорные. Группировка воспроизводит семантику HotpotQA буквально — ``context``
это ``[титул, [фрагменты статьи]]``, а ``supporting_facts`` это
``[титул, номер фрагмента внутри статьи]``, — и заодно даёт правильное число
различных gold-титулов, по которым считается title EM.

``answer_aliases`` переносятся как есть: сам пайплайн их не использует
(``answer_judge_llms.py`` сравнивает с одним ``answer``), но
``export_eval_json.py`` считает по ним alias-метрики.

Пример:

    python src/build_musique_eval.py \
      --input datasets/data_sources/musique/musique_ans_v1.0_dev.jsonl \
      --output runs/shared/musique_ans_dev_hotpot_schema.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Iterator, Sequence

from fullwiki_qrag import load_input_samples, supporting_fact_texts


LOG = logging.getLogger("build-musique-eval")


def hop_type(sample_id: str) -> str:
    """Класс вопроса из его идентификатора: ``2hop__1234_5678`` → ``2hop``.

    Поле нужно тем же, чем ``type`` у 2WikiMultiHopQA: разрезом результатов
    по числу переходов. У MuSiQue оно закодировано в id и больше нигде.
    """
    return sample_id.partition("__")[0] or "unknown"


def convert_sample(sample: dict[str, Any], index: int) -> dict[str, Any]:
    sample_id = str(sample.get("id", index))
    question = sample.get("question")
    if not isinstance(question, str) or not question.strip():
        raise ValueError(f"У примера {sample_id!r} пустой вопрос")
    answer = sample.get("answer")
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError(f"У примера {sample_id!r} пустой ответ")
    paragraphs = sample.get("paragraphs")
    if not isinstance(paragraphs, list) or not paragraphs:
        raise ValueError(f"У примера {sample_id!r} нет абзацев")

    # Порядок титулов — порядок первого появления, порядок фрагментов внутри
    # титула — порядок в исходном списке. Оба нужны, чтобы конвертация была
    # детерминированной и сравнимой между запусками.
    context: list[list[Any]] = []
    positions: dict[str, int] = {}
    supporting: list[list[Any]] = []
    for paragraph in paragraphs:
        title = str(paragraph.get("title", "")).strip()
        text = paragraph.get("paragraph_text")
        if not title:
            raise ValueError(f"У примера {sample_id!r} абзац без титула")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"У примера {sample_id!r} пустой абзац {title!r}")
        if title not in positions:
            positions[title] = len(context)
            context.append([title, []])
        fragments = context[positions[title]][1]
        if paragraph.get("is_supporting"):
            supporting.append([title, len(fragments)])
        fragments.append(text)
    if not supporting:
        raise ValueError(f"У примера {sample_id!r} нет опорных абзацев")

    return {
        "_id": sample_id,
        "type": hop_type(sample_id),
        "question": question,
        "answer": answer,
        "answer_aliases": list(sample.get("answer_aliases") or []),
        "context": context,
        "supporting_facts": supporting,
    }


def verify_sample(source: dict[str, Any], converted: dict[str, Any]) -> None:
    """Сверить конвертацию тем же кодом, которым её будет читать пайплайн.

    Проверяется ровно то, ради чего конвертация затевалась: gold-предложения,
    которые достаёт ``supporting_fact_texts``, — это тексты опорных абзацев
    исходного примера, в том же порядке и без потерь на группировке титулов.
    """
    expected = [
        f"{str(paragraph['title']).strip()} {paragraph['paragraph_text']}"
        for paragraph in source["paragraphs"]
        if paragraph.get("is_supporting")
    ]
    actual = supporting_fact_texts(converted)
    if actual != expected:
        raise RuntimeError(
            f"Пример {converted['_id']!r} после конвертации даёт другие "
            f"gold-предложения: {len(actual)} против {len(expected)}"
        )


def convert(samples: Sequence[dict[str, Any]], verify_every: int) -> Iterator[dict[str, Any]]:
    for index, sample in enumerate(samples):
        converted = convert_sample(sample, index)
        if verify_every and index % verify_every == 0:
            verify_sample(sample, converted)
        yield converted


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--verify-every",
        type=int,
        default=1,
        help="как часто сверять конвертацию через supporting_fact_texts; 0 отключает",
    )
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
    source = args.input.expanduser().resolve()
    destination = args.output.expanduser().resolve()
    samples = load_input_samples(source)
    LOG.info("Прочитано %d примеров: %s", len(samples), source)

    destination.parent.mkdir(parents=True, exist_ok=True)
    # Пишем через временный файл: оборванная конвертация не должна оставить
    # правдоподобный, но неполный вход для многочасового рана.
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    written = 0
    titles = 0
    with temporary.open("w", encoding="utf-8") as stream:
        for record in convert(samples, args.verify_every):
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1
            titles += len({fact[0] for fact in record["supporting_facts"]})
    temporary.replace(destination)
    LOG.info(
        "Записано %d примеров, gold-титулов в среднем %.2f: %s",
        written,
        titles / written if written else 0.0,
        destination,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
