#!/usr/bin/env python3
"""Convert MuSiQue into the HotpotQA schema understood by the whole pipeline.

``fullwiki_qrag.py`` reads its input via ``load_input_samples`` and takes
``question``, ``answer`` and the ``context`` / ``supporting_facts`` pair from
it. MuSiQue has a flat ``paragraphs`` list with an ``is_supporting`` flag
instead of that pair, so the raw file passes through the pipeline silently
and without gold titles. Silence is worse than a crash here: on an empty
gold-title list ``title_metrics`` returns ``title_recall = title_em = 1.0``,
so ``metrics.json`` would report a perfect title hit rate and a share of the
ceiling above one.

Paragraphs are grouped by title rather than mapped one to one: 1,293 of the
2,417 dev questions list two paragraphs of the same article, and in 116 both
are supporting. Grouping reproduces the HotpotQA semantics literally
(``context`` is ``[title, [article fragments]]`` and ``supporting_facts`` is
``[title, fragment index within the article]``) and also gives the correct
number of distinct gold titles for title EM.

``answer_aliases`` are carried over as is: the pipeline itself does not use
them (``answer_judge_llms.py`` compares against a single ``answer``), but
``export_eval_json.py`` computes alias metrics from them.

Example:

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
    """Question class from its identifier: ``2hop__1234_5678`` -> ``2hop``.

    The field serves the same purpose as ``type`` in 2WikiMultiHopQA: slicing
    results by the number of hops. MuSiQue encodes it only in the id.
    """
    return sample_id.partition("__")[0] or "unknown"


def convert_sample(sample: dict[str, Any], index: int) -> dict[str, Any]:
    sample_id = str(sample.get("id", index))
    question = sample.get("question")
    if not isinstance(question, str) or not question.strip():
        raise ValueError(f"Sample {sample_id!r} has an empty question")
    answer = sample.get("answer")
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError(f"Sample {sample_id!r} has an empty answer")
    paragraphs = sample.get("paragraphs")
    if not isinstance(paragraphs, list) or not paragraphs:
        raise ValueError(f"Sample {sample_id!r} has no paragraphs")

    # Titles keep the order of first appearance, fragments within a title keep
    # the order of the source list. Both keep the conversion deterministic and
    # comparable across launches.
    context: list[list[Any]] = []
    positions: dict[str, int] = {}
    supporting: list[list[Any]] = []
    for paragraph in paragraphs:
        title = str(paragraph.get("title", "")).strip()
        text = paragraph.get("paragraph_text")
        if not title:
            raise ValueError(f"Sample {sample_id!r} has a paragraph without a title")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Sample {sample_id!r} has an empty paragraph {title!r}")
        if title not in positions:
            positions[title] = len(context)
            context.append([title, []])
        fragments = context[positions[title]][1]
        if paragraph.get("is_supporting"):
            supporting.append([title, len(fragments)])
        fragments.append(text)
    if not supporting:
        raise ValueError(f"Sample {sample_id!r} has no supporting paragraphs")

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
    """Check the conversion with the same code the pipeline will read it with.

    It checks exactly what the conversion is for: the gold sentences returned
    by ``supporting_fact_texts`` are the texts of the source sample's
    supporting paragraphs, in the same order and with nothing lost to title
    grouping.
    """
    expected = [
        f"{str(paragraph['title']).strip()} {paragraph['paragraph_text']}"
        for paragraph in source["paragraphs"]
        if paragraph.get("is_supporting")
    ]
    actual = supporting_fact_texts(converted)
    if actual != expected:
        raise RuntimeError(
            f"Sample {converted['_id']!r} yields different gold sentences "
            f"after conversion: {len(actual)} vs {len(expected)}"
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
        help="how often to check the conversion via supporting_fact_texts; 0 disables",
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
    LOG.info("Read %d samples: %s", len(samples), source)

    destination.parent.mkdir(parents=True, exist_ok=True)
    # Write through a temporary file: an interrupted conversion must not leave
    # a plausible but incomplete input for a multi-hour run.
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
        "Wrote %d samples, %.2f gold titles on average: %s",
        written,
        titles / written if written else 0.0,
        destination,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
